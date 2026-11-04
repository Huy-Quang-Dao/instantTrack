"""Evaluate many checkpoints of a run at once (same protocol as instanttrack.evaluate, App. F.1).

Each checkpoint gets its own block of C worlds (one per clip) in a single simulator, so M checkpoints
cost about as much wall-clock as one: evaluation is latency-bound at C = 40 worlds. Results are
appended to ``<run_dir>/eval.csv`` (``eval_r<reps>.csv`` with --reps) in the format of ``instanttrack.evaluate``.

    python tools/eval_batch.py --run_dir logs/<run> --group 26 --reps 4
"""

import argparse
import glob
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--robot", type=str, default="g1")
parser.add_argument("--group", type=int, default=13, help="checkpoints evaluated together")
parser.add_argument("--reps", type=int, default=1,
                    help="independent replicas of every clip per checkpoint (the simulator is not bitwise "
                         "reproducible, so borderline clips can flip); results are averaged and written to eval_r<reps>.csv")
args = parser.parse_args()

import torch

import instanttrack.networks as networks
from instanttrack.dr import XiLayout
from instanttrack.env import TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import MAX_STEP_REWARD, MDP
from instanttrack.motion import MotionLib
from instanttrack.networks import Policy, check_pruned, policy_kwargs
from instanttrack.robot import get_robot

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
robot = get_robot(args.robot)
kin = Kinematics(robot, dev)
_run_args = json.load(open(os.path.join(args.run_dir, "args.json"))) if os.path.exists(os.path.join(args.run_dir, "args.json")) else {}
motion = MotionLib.from_dir(_run_args.get("motion_dir") or robot.motion_dir, robot, kin, dev)  # the run's own motion set
check_pruned(_run_args, nets_only=True, what=args.run_dir)
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
C = len(motion.names)


def load_policy(ck):
    a_ = ck["args"]
    widths = lambda key: tuple(int(x) for x in a_.get(key, "2048,1024,512,256").split(","))  # noqa: E731
    pi = Policy(mdp.obs_dim, motion.goal_dim(a_["goal_frames"]), robot.num_joints, hist=a_["hist"],
                enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"), **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
    pi.load_state_dict(ck["pi"])
    return pi.eval()


@torch.no_grad()
def evaluate_group(env: TrackingSim, cks: list[dict]) -> list[dict]:
    M = len(cks)
    N = env.N
    pis = [load_policy(ck) for ck in cks]
    a0 = cks[0]["args"]
    K, Hh = a0["goal_frames"], a0["hist"]
    assert all(ck["args"]["goal_frames"] == K and ck["args"]["hist"] == Hh for ck in cks)
    networks.AMP_DTYPE = torch.bfloat16 if a0.get("amp", "none") == "bf16" else None
    clip_of = torch.arange(N, device=dev) % C
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(torch.arange(N, device=dev), motion.clip_start[clip_of].clone())
    clip_end = motion.clip_end[clip_of]
    ret = torch.zeros(N, device=dev)
    hist = None
    a = torch.zeros(N, robot.num_joints, device=dev)
    while not bool(env.done_eval.all()):
        o = env.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        g = motion.goal(env.f, K)
        for m, pi in enumerate(pis):
            sl = slice(m * C, (m + 1) * C)
            a[sl] = pi(hist[sl], g[sl])
        alive = ~env.done_eval.clone()
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
    clip_len = motion.clip_len[clip_of]
    norm = ret / ((clip_len - 1).float() * MAX_STEP_REWARD)
    res = []
    for m in range(M):
        sl = slice(m * C, (m + 1) * C)
        failed = env.failed_eval[sl]
        res.append({
            "norm_return": norm[sl].mean().item(),
            "norm_return_framew": (ret[sl].sum() / ((clip_len[sl] - 1).sum() * MAX_STEP_REWARD)).item(),
            "n_failed": int(failed.sum().item()),
            "failed": [motion.names[i] for i in failed.nonzero().flatten().tolist()],
            "per_clip": {motion.names[i]: round(norm[m * C + i].item(), 4) for i in range(C)},
        })  # fmt: skip
    return res


def merge_reps(res: list[dict]) -> list[dict]:
    """Average every ``args.reps`` consecutive results (replicas of one checkpoint); n_failed becomes the mean
    number of failed clips per replica and ``failed`` lists every clip that failed in at least one replica."""
    R = args.reps
    if R == 1:
        return res
    out = []
    for k in range(0, len(res), R):
        rs = res[k : k + R]
        out.append({
            "norm_return": sum(r["norm_return"] for r in rs) / R,
            "norm_return_framew": sum(r["norm_return_framew"] for r in rs) / R,
            "n_failed": sum(r["n_failed"] for r in rs) / R,
            "failed": sorted({n for r in rs for n in r["failed"]}),
            "per_clip": {n: round(sum(r["per_clip"][n] for r in rs) / R, 4) for n in rs[0]["per_clip"]},
            "reps": [{"norm_return": r["norm_return"], "n_failed": r["n_failed"], "failed": r["failed"]} for r in rs],
        })  # fmt: skip
    return out


SUFFIX = "" if args.reps == 1 else f"_r{args.reps}"
out_csv = os.path.join(args.run_dir, f"eval{SUFFIX}.csv")
done = set()
if os.path.exists(out_csv):
    done = {int(line.split(",")[0]) for line in open(out_csv).read().strip().splitlines()[1:]}
else:
    open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
todo = [c for c in sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt"))) if int(os.path.basename(c)[5:11]) not in done]
print(f"{len(todo)} checkpoints to evaluate in groups of {args.group}", flush=True)
if done and todo:  # the simulator is not bitwise reproducible: a different group layout can flip borderline clips
    print(f"note: {len(done)} checkpoints already in {out_csv}; the remaining ones are grouped anew (layout in the JSON files)", flush=True)
MOTION_ID = {"clips": C, "frames": motion.num_frames, "names": motion.names}
env = None
for s in range(0, len(todo), args.group):
    paths = todo[s : s + args.group]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]  # full checkpoints also hold the world model: keep them off the GPU
    if env is None or env.N != C * len(paths) * args.reps:
        env = TrackingSim(make_sim(robot), C * len(paths) * args.reps, robot, motion, mdp, dev, eval_mode=True, randomize=False)
    t0 = time.time()
    for p, ck, res in zip(paths, cks, merge_reps(evaluate_group(env, [c for c in cks for _ in range(args.reps)]))):
        it = int(os.path.basename(p)[5:11])
        # 6 decimals: the 0.82 threshold is applied to these values
        line = (f"{it},{ck['wall_time'] / 3600:.6f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.6f},"
                f"{res['norm_return_framew']:.6f},{res['n_failed']:g},{'|'.join(res['failed'])}\n")  # fmt: skip
        res["layout"] = {"n_worlds": env.N, "group": [int(os.path.basename(q)[5:11]) for q in paths], "reps": args.reps,
                         "motion": MOTION_ID}  # fmt: skip
        open(out_csv, "a").write(line)
        json.dump(res, open(p.replace(".pt", f"_eval{SUFFIX}.json"), "w"), indent=1)
        print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C}", flush=True)
    print(f"group of {len(paths)} in {time.time() - t0:.0f}s", flush=True)
