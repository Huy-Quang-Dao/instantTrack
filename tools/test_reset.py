"""Check that state() reflects a reset immediately (no stale data)."""
import argparse
parser = argparse.ArgumentParser(); parser.add_argument("--robot", type=str, default="g1"); args = parser.parse_args()
import torch
from instanttrack.dr import XiLayout
from instanttrack.env import TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import MDP, unflatten_state
from instanttrack.motion import MotionLib
from instanttrack.robot import get_robot
dev = "cuda:0"
robot = get_robot(args.robot)
kin = Kinematics(robot, dev); motion = MotionLib.from_dir(robot.motion_dir, robot, kin, dev)
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
env = TrackingSim(make_sim(robot), 64, robot, motion, mdp, dev, eval_mode=True)
ids = torch.arange(64, device=dev)
for _ in range(30):
    env.step(torch.zeros(64, robot.num_joints, device=dev))
fr = motion.clip_start[ids % len(motion.names)] + 200
env.reset(ids, fr)
S = env.state(); ref = motion.ref_state(fr)
print("no refresh:", [round((a - b).abs().max().item(), 4) for a, b in zip(S, ref)])
env.refresh()
S = env.state()
print("after refresh:", [round((a - b).abs().max().item(), 4) for a, b in zip(S, ref)])
out = env.step(torch.zeros(64, robot.num_joints, device=dev))
print("step s vs ref:", [round((a - b).abs().max().item(), 4) for a, b in zip(unflatten_state(out["s"]), ref)])
