"""Offline benchmark/diagnostics of world-model and policy updates on a synthetic buffer (no simulator)."""
import sys, time
import torch
from instanttrack.kinematics import Kinematics
from instanttrack.motion import MotionLib
from instanttrack.dr import XiLayout
from instanttrack.mdp import MDP
from instanttrack.networks import WorldModel
from instanttrack.robot import get_robot
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
robot = get_robot(sys.argv[1] if len(sys.argv) > 1 else "g1")
J = robot.num_joints
kin = Kinematics(robot, dev)
motion = MotionLib.from_dir(robot.motion_dir, robot, kin, dev)
layout = XiLayout(robot, dev)
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), layout.dim, robot.sensing)
wm = WorldModel(J, layout.dim, use_prev_action=robot.has_action_delay).to(dev)
normalize_xi = layout.normalize
B = 2048
f = torch.randint(0, motion.num_frames - 20, (B * 8,), device=dev)
S = motion.ref_state(f); S1 = motion.ref_state(f + 1)
a = torch.randn(B * 8, J, device=dev); xi = torch.zeros(B * 8, mdp.xi_dim, device=dev); xi[:, 0] = 0.75
wm.in_norm.update(wm.inputs(S, a, a, normalize_xi(xi))); wm.out_norm.update(wm.targets(S, S1))
std = wm.out_norm.std()
names = ["dp"] * 3 + ["dth"] * 3 + ["dv"] * 3 + ["dw"] * 3 + ["dq"] * J + ["dqd"] * J
print("out std per group:", {n: f"{std[[i for i, m in enumerate(names) if m == n]].min().item():.2e}..{std[[i for i, m in enumerate(names) if m == n]].max().item():.2e}" for n in dict.fromkeys(names)})
def timeit(fn, n=5):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n
Sb = tuple(x[:B] for x in S)
def wm_step_fb():
    out = wm.step(Sb, a[:B], a[:B], normalize_xi(xi[:B]))
    sum(o.sum() for o in out).backward()
print("wm step fwd+bwd (B=2048): %.1f ms" % (1000 * timeit(wm_step_fb)))
Sb4 = tuple(x[:4096].clone().requires_grad_(True) for x in motion.ref_state(f[:4096]))
fb = f[:4096]
print("FK fwd (B=4096): %.2f ms" % (1000 * timeit(lambda: kin.forward(*Sb4))))
print("reward fwd+bwd: %.2f ms" % (1000 * timeit(lambda: mdp.reward(Sb4, fb, a[:4096], a[:4096]).sum().backward())))
print("obs fwd: %.2f ms" % (1000 * timeit(lambda: mdp.obs(Sb4, a[:4096], xi[:4096], fb))))
print("goal: %.2f ms" % (1000 * timeit(lambda: motion.goal(fb, 10))))
print("critic feats: %.2f ms" % (1000 * timeit(lambda: mdp.critic_features(Sb4, fb, normalize_xi(xi[:4096])))))
print("---- torch.compile ----")
kin_fwd_c = torch.compile(kin.forward)
t0 = time.time(); kin_fwd_c(*Sb4); torch.cuda.synchronize(); print("compile FK %.1fs" % (time.time() - t0))
print("FK compiled fwd: %.2f ms" % (1000 * timeit(lambda: kin_fwd_c(*Sb4))))
rew_c = torch.compile(mdp.reward)
t0 = time.time(); rew_c(Sb4, fb, a[:4096], a[:4096]).sum().backward(); torch.cuda.synchronize(); print("compile reward %.1fs" % (time.time() - t0))
print("reward compiled fwd+bwd: %.2f ms" % (1000 * timeit(lambda: rew_c(Sb4, fb, a[:4096], a[:4096]).sum().backward())))
