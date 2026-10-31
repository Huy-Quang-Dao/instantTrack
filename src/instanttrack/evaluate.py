"""Checkpoint evaluation (paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.

Run as a watcher: ``python -m instanttrack.evaluate --run_dir logs/<run> --watch``.

Robustness: ``--dr_from <robot>`` evaluates under that robot's training randomization and sensing
(``self`` for the evaluated robot's own), still one start per clip from its first frame;
``--repeats n`` runs every clip n times to average the randomness.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="g1", help="robot package under instanttrack/ (g1)")
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
parser.add_argument("--motion_dir", type=str, default=None, help="defaults to the robot's motion directory")
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
parser.add_argument("--dr_from", type=str, default=None, help="evaluate under this robot's training randomization ('self': own)")
parser.add_argument("--dr_parts", type=str, default="all", choices=["all", "dynamics", "sensing"],
                    help="with --dr_from: take only its simulator randomization or only its sensing")
parser.add_argument("--repeats", type=int, default=1, help="rollouts per clip")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

import dataclasses
import re

import numpy as np
import torch

import instanttrack.networks as networks
from instanttrack.dr import XiLayout
from instanttrack.env import TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import MAX_STEP_REWARD, MDP
from instanttrack.motion import MotionLib
from instanttrack.networks import Policy
from instanttrack.robot import RobotSpec, get_robot


def transfer_dr(target: RobotSpec, source: RobotSpec, parts: str = "all") -> RobotSpec:
    """``source``'s randomization and/or sensing on ``target``. PD-gain ranges are given per actuator group,
    so each target group takes the range of the source entry that covers most of its joints."""
    if parts == "sensing":
        return dataclasses.replace(target, sensing=source.sensing)
    if parts == "dynamics":
        return dataclasses.replace(transfer_dr(target, source), sensing=target.sensing)
    if target.name == source.name:
        return target
    src_acts = source.robot_cfg().articulation.actuators
    tgt_acts = target.robot_cfg().articulation.actuators
    joints = lambda act, names: {n for n in names if any(re.fullmatch(p, n) for p in act.target_names_expr)}  # noqa: E731
    entries = [(set().union(*(joints(src_acts[i], source.joint_names) for i in ids)), kp, kd) for ids, kp, kd in source.dr.pd_gain_scale]
    groups: dict[int, list[int]] = {}
    for i, act in enumerate(tgt_acts):
        tj = joints(act, target.joint_names)
        best = max(range(len(entries)), key=lambda e: len(tj & entries[e][0]))
        groups.setdefault(best, []).append(i)
    pd = tuple((tuple(ids), entries[e][1], entries[e][2]) for e, ids in sorted(groups.items()))
    dr = dataclasses.replace(source.dr, pd_gain_scale=pd)
    return dataclasses.replace(target, dr=dr, sensing=source.sensing, com_range=source.com_range)


torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(args.seed)
dev = "cuda:0"
robot = get_robot(args.robot)
if args.dr_from:
    robot = transfer_dr(robot, robot if args.dr_from == "self" else get_robot(args.dr_from), args.dr_parts)
    print("pd_gain_scale on this robot:", robot.dr.pd_gain_scale)
kin = Kinematics(robot, dev)
motion = MotionLib.from_dir(args.motion_dir or robot.motion_dir, robot, kin, dev)
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
C = len(motion.names)
N = C * args.repeats
clip_of = torch.arange(N, device=dev) % C


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    widths = lambda key, default: tuple(int(x) for x in a_.get(key, default).split(","))  # noqa: E731
    pi = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=Hh, enc_hidden=widths("enc_hidden", "2048,1024,512,256"),
                dec_hidden=widths("dec_hidden", "2048,1024,512,256"), ref_residual=a_.get("ref_residual", 0.0), experts=a_.get("pi_experts", 1), simba_arch=a_.get("pi_simba", "")).to(dev)  # fmt: skip
    pi.load_state_dict(ck["pi"])
    pi.eval()
    ids = torch.arange(N, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start[clip_of].clone())
    clip_end = motion.clip_end[clip_of]
    ret = torch.zeros(N, device=dev)
    steps = torch.zeros(N, device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    while not bool(env.done_eval.all()):
        S = env.state()
        o = env.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        if record:
            from instanttrack.rotations import mat_to_quat

            traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        steps += alive.float()
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
    clip_len = motion.clip_len[clip_of]
    norm = ret / ((clip_len - 1).float() * MAX_STEP_REWARD)
    norm_c = norm.view(args.repeats, C).mean(0)
    fail_c = env.failed_eval.view(args.repeats, C).float().mean(0)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((clip_len - 1).sum() * MAX_STEP_REWARD)).item(),
        "n_failed": int(env.failed_eval.sum().item()),
        "fail_rate": env.failed_eval.float().mean().item(),
        "failed": [motion.names[i] for i in (fail_c > 0).nonzero().flatten().tolist()],
        "per_clip": {motion.names[i]: round(norm_c[i].item(), 4) for i in range(C)},
        "per_clip_fail_rate": {motion.names[i]: round(fail_c[i].item(), 3) for i in range(C) if fail_c[i] > 0},
    }
    if record:
        np.savez_compressed(record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res


sim_cfg = make_sim(robot)
env = TrackingSim(sim_cfg, N, robot, motion, mdp, dev, eval_mode=True, randomize=args.dr_from is not None)
if args.ckpt:
    ck = torch.load(args.ckpt, map_location=dev)
    res = evaluate(env, ck, args.record)
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
    if args.record is None and (args.dr_from or args.repeats > 1):
        tag = f"dr-{args.dr_from or 'none'}{'' if args.dr_parts == 'all' else '-' + args.dr_parts}_x{args.repeats}"
        json.dump(res, open(args.ckpt.replace(".pt", f"_eval_{tag}.json"), "w"), indent=1)
    sys.exit(0)
out_csv = os.path.join(args.run_dir, "eval.csv")
done = set()
if os.path.exists(out_csv):
    done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
else:
    open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
while True:
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
    todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
    todo = [c for c in todo if (int(os.path.basename(c)[5:11]) // 2000) % args.every == 0]
    for c in todo:
        it = int(os.path.basename(c)[5:11])
        ck = torch.load(c, map_location=dev)
        t0 = time.time()
        res = evaluate(env, ck)
        line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"
        open(out_csv, "a").write(line)
        json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
        print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} ({time.time() - t0:.0f}s)", flush=True)
        done.add(it)
    if not args.watch:
        break
    time.sleep(60)
