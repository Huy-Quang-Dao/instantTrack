"""Check FK vs simulator, state round-trip, and reward on reference playback."""

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="g1")
parser.add_argument("--motion_dir", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=64)
args = parser.parse_args()

import torch

from instanttrack.dr import XiLayout
from instanttrack.env import TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import MDP
from instanttrack.motion import MotionLib
from instanttrack.robot import get_robot
from instanttrack.rotations import quat_to_mat, rot_angle_sq

dev = "cuda:0"
robot = get_robot(args.robot)
kin = Kinematics(robot, dev)
t0 = time.time()
motion = MotionLib.from_dir(args.motion_dir or robot.motion_dir, robot, kin, dev)
print(f"motion: {len(motion.names)} clips, {motion.num_frames} frames ({motion.num_frames * 0.02 / 3600:.2f} h), bins {motion.num_bins}, load {time.time() - t0:.1f}s")
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
env = TrackingSim(make_sim(robot), args.num_envs, robot, motion, mdp, dev, eval_mode=True)
frames = motion.clip_start[torch.arange(args.num_envs, device=dev) % len(motion.names)] + 100
env.reset(torch.arange(args.num_envs, device=dev), frames)
env.refresh()
S = env.state()
ref = motion.ref_state(frames)
names = ["p", "R", "v", "w", "q", "qd"]
for n, a, b in zip(names, S, ref):
    print(f"roundtrip {n}: max err {(a - b).abs().max().item():.2e}")
# FK vs sim body frames
d = env.robot.data
ids = [env.robot.body_names.index(b) for b in robot.tracked_bodies]
sp = d.body_link_pos_w[:, ids]
sR = quat_to_mat(d.body_link_quat_w[:, ids])  # wxyz
bp, bR, bv, bw = kin.forward(*S)
print("FK pos err", (bp - sp).norm(dim=-1).max().item())
print("FK rot err", rot_angle_sq(bR, sR).sqrt().max().item())
sv = d.body_link_vel_w[:, ids]
print("FK linvel err", (bv - sv[..., :3]).norm(dim=-1).max().item(), " angvel err", (bw - sv[..., 3:]).norm(dim=-1).max().item())
print("ref body vs FK(sim) pos err", (motion.body_p[frames] - bp).norm(dim=-1).max().item())
# open-loop reference playback via joint targets
rs = []
for t in range(150):
    f1 = env.f + 1
    a = (motion.q[f1] - env.default_q) / env.alpha
    out = env.step(a)
    rs.append(out["r"].mean().item() / 0.1)
    if t % 25 == 0:
        print(t, "norm reward", rs[-1], "term frac", env.failed_eval.float().mean().item(), "e_body_pos", out["terms"]["e_body_pos"].mean().item())
