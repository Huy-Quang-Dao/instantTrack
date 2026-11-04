"""Equivalence of TrackingSim.reset_masked / step_tail_masked with reset / step_tail.

From one simulator state (512 training worlds after some random-action steps, some episodes ended): (1) reset(ids,
frames[ids]) and, from the same restored state, reset_masked(mask, frames) with the same frames and reset noise off;
every compared field must be identical, except the joint offsets of the reset worlds (random, checked for range).
(2) the bin statistics of step_tail and step_tail_masked on the same transition (resets stubbed out).

    PYTHONPATH=src python tools/reset_check.py
"""
import torch

from instanttrack.dr import XiLayout
from instanttrack.env import TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import MDP
from instanttrack.motion import MotionLib
from instanttrack.robot import get_robot

dev = "cuda:0"
torch.manual_seed(0)
robot = get_robot("g1")
kin = Kinematics(robot, dev)
motion = MotionLib.from_dir(robot.motion_dir, robot, kin, dev)
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
N = 512
env = TrackingSim(make_sim(robot), N, robot, motion, mdp, dev, fail_reset=0.3, bin_fix=True, reset_noise=False)
for _ in range(30):
    env.observe()
    env.step(0.5 * torch.randn(N, robot.num_joints, device=dev))
SIM = ("qpos", "qvel", "ctrl", "qacc_warmstart", "act", "qacc", "xfrc_applied")
ED = ("joint_pos_target", "joint_vel_target", "joint_effort_target")
ENV = ("f", "ep_step", "a_prev", "start_bin", "ep_disc_ret", "ep_ret", "xi", "obs_fresh", "bin_G", "bin_fail", "bin_w_avg")


def snap():
    out = {k: getattr(env.sim.data, k)[:].clone() for k in SIM}
    out.update({k: getattr(env.robot.data, k).clone() for k in ED})
    out.update({k: getattr(env, k).clone() for k in ENV})
    return out


def restore(st):
    for k in SIM:
        getattr(env.sim.data, k)[:] = st[k]
    for k in ED:
        getattr(env.robot.data, k).copy_(st[k])
    for k in ENV:
        getattr(env, k).copy_(st[k])
    torch.cuda.synchronize()


ok = True
mask = torch.rand(N, device=dev) < 0.3
ids = mask.nonzero().squeeze(-1)
frames, _ = env.sample_frames(N)
s0 = snap()
env.reset(ids, frames[ids])
a = snap()
restore(s0)
env.reset_masked(mask, frames)
b = snap()
jo = env.layout.slices["joint_offset"]
for k in a:
    x, y = a[k], b[k]
    if k == "xi":
        x, y = x.clone(), y.clone()
        half = env.randomizer.joint_offset_half
        inr = bool(((y[mask][:, jo].abs() <= half + 1e-7).all()))
        print(f"  xi joint_offset of reset worlds within range: {inr}; unchanged elsewhere: "
              f"{torch.equal(y[~mask][:, jo], s0['xi'][~mask][:, jo])}")  # fmt: skip
        ok &= inr
        x[:, jo] = 0; y[:, jo] = 0
    same = torch.equal(x, y)
    if same:
        print(f"  {k:22s} identical")
        continue
    # per-world fields: worlds outside the mask must be bitwise unchanged, reset worlds equal up to float rounding
    keep = torch.equal(x[~mask], y[~mask]) if x.shape[:1] == (N,) else False
    d = float((x.float() - y.float()).abs().max())
    print(f"  {k:22s} differs: max |diff| {d:.3g}; worlds outside the mask identical: {keep}")
    ok &= keep and d <= 1e-6
print(f"(1) reset vs reset_masked on {int(mask.sum())}/{N} worlds: {'PASS' if ok else 'FAIL'}")

# (2) bin statistics of the tails on one transition (resets stubbed out)
restore(s0)
env.observe()
out = env.step_core(0.5 * torch.randn(N, robot.num_joints, device=dev))
print(f"  ended episodes in the transition: {int(out['done'].sum())}")
s1 = snap()
r, rm = env.reset, env.reset_masked
env.reset = lambda ids, frames=None: None
env.reset_masked = lambda m, frames=None: None
o1 = env.step_tail(dict(out))
a = snap()
restore(s1)
o2 = env.step_tail_masked(dict(out))
b = snap()
env.reset, env.reset_masked = r, rm
ok2 = True
for k in ("bin_G", "bin_fail"):
    d = float((a[k] - b[k]).abs().max())
    print(f"  {k:10s} max |diff| {d:.3g}")
    ok2 &= d <= 1e-6
n = int(out["done"].sum())
if n:
    for k_e, k_m in (("ep_len", "ep_len_sum"), ("ep_ret_norm", "ep_ret_norm_sum")):
        d = abs(float(o1[k_e].mean()) - float(o2[k_m]) / n)
        print(f"  {k_e:10s} mean {float(o1[k_e].mean()):.6f} vs {float(o2[k_m]) / n:.6f}")
        ok2 &= d <= 1e-5
print(f"(2) step_tail vs step_tail_masked: {'PASS' if ok2 else 'FAIL'}")

# (3) the device-side focus (use_static_focus / set_focus) that a captured reset_masked reads
restore(s0)
w0 = env.bin_w_avg.clone()
torch.manual_seed(5)
fa, _ = env.sample_frames(N)
env.bin_w_avg.copy_(w0)
env.use_static_focus()
torch.manual_seed(5)
fb, _ = env.sample_frames(N)
same0 = torch.equal(fa, fb)
q = torch.zeros(motion.num_bins, device=dev)
q[123] = 1.0
env.set_focus(q, 1.0)
_, bins = env.sample_frames(N)
all_focus = bool((bins == 123).all())
print(f"(3) static focus: share 0 gives the same frames as no focus: {same0}; share 1 on one bin: all resets there: "
      f"{all_focus} -> {'PASS' if same0 and all_focus else 'FAIL'}")  # fmt: skip
