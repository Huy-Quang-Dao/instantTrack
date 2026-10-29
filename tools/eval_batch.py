"""Evaluate many checkpoints of a run at once (same protocol as instanttrack.evaluate, App. F.1).

Each checkpoint gets its own block of C worlds (one per clip) in a single simulator, so M checkpoints
cost about as much wall-clock as one: evaluation is latency-bound at C = 40 worlds. Results are
appended to ``<run_dir>/eval.csv`` in the format of ``instanttrack.evaluate``.

    uv run python tools/eval_batch.py --run_dir logs/<run> [--group 13]
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
args = parser.parse_args()

import torch

import instanttrack.networks as networks
from instanttrack.dr import XiLayout
from instanttrack.env import TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import MAX_STEP_REWARD, MDP
from instanttrack.motion import MotionLib
from instanttrack.networks import Policy
from instanttrack.robot import get_robot

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
robot = get_robot(args.robot)
kin = Kinematics(robot, dev)
motion = MotionLib.from_dir(robot.motion_dir, robot, kin, dev)
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
C = len(motion.names)


def load_policy(ck):
    a_ = ck["args"]
    widths = lambda key: tuple(int(x) for x in a_.get(key, "2048,1024,512,256").split(","))  # noqa: E731
    pi = Policy(mdp.obs_dim, motion.goal_dim(a_["goal_frames"]), robot.num_joints, hist=a_["hist"],
                enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden")).to(dev)  # fmt: skip
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


out_csv = os.path.join(args.run_dir, "eval.csv")
done = set()
if os.path.exists(out_csv):
    done = {int(line.split(",")[0]) for line in open(out_csv).read().strip().splitlines()[1:]}
else:
    open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
todo = [c for c in sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt"))) if int(os.path.basename(c)[5:11]) not in done]
print(f"{len(todo)} checkpoints to evaluate in groups of {args.group}", flush=True)
env = None
for s in range(0, len(todo), args.group):
    paths = todo[s : s + args.group]
    cks = [torch.load(p, map_location=dev) for p in paths]
    if env is None or env.N != C * len(paths):
        env = TrackingSim(make_sim(robot), C * len(paths), robot, motion, mdp, dev, eval_mode=True, randomize=False)
    t0 = time.time()
    for p, ck, res in zip(paths, cks, evaluate_group(env, cks)):
        it = int(os.path.basename(p)[5:11])
        line = (f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},"
                f"{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n")  # fmt: skip
        open(out_csv, "a").write(line)
        json.dump(res, open(p.replace(".pt", "_eval.json"), "w"), indent=1)
        print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C}", flush=True)
    print(f"group of {len(paths)} in {time.time() - t0:.0f}s", flush=True)
