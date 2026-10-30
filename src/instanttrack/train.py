"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="g1", help="robot package under instanttrack/ (g1)")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
# ---- InstantTrack options (defaults reproduce SuperMimic) ----
parser.add_argument("--wm_hidden", type=str, default="2048,2048,2048", help="world-model MLP widths")
parser.add_argument("--enc_hidden", type=str, default="2048,1024,512,256", help="motion-encoder MLP widths")
parser.add_argument("--dec_hidden", type=str, default="2048,1024,512,256", help="shared-decoder MLP widths")
parser.add_argument("--no_sync", action="store_true", help="remove host-device synchronizations from the update loop")
parser.add_argument("--demo_replay", action="store_true",
                    help="DiffMimic demonstration replay: imagined states that fail are replaced by the reference state")  # fmt: skip
parser.add_argument("--demo_replay_thr", type=float, default=0.0,
                    help="also replay when the mean squared tracked-body position error exceeds this (m^2; DiffMimic's pose threshold)")  # fmt: skip
parser.add_argument("--fail_reset", type=float, default=0.0,
                    help="share of resets sampled one bin before recent termination points (anti-forgetting of hard clips)")
parser.add_argument("--imag_action_noise", type=float, default=0.0, help="reparameterized Gaussian action noise in imagination")
parser.add_argument("--fused_adam", action="store_true", help="fused Adam kernels and a compiled critic update")
parser.add_argument("--wm_foot_weight", type=float, default=0.0,
                    help="contact-aware world-model loss: weight of foot position/velocity errors at frames where the foot is on the ground")  # fmt: skip
parser.add_argument("--anchored", action="store_true",
                    help="imagination anchored to logged trajectories: s_{t+1} = s^real_{t+1} + f(s_t, pi) - f(s^real_t, a^data)")  # fmt: skip
parser.add_argument("--pi_beta1", type=float, default=0.9, help="Adam beta1 of the policy (SHAC uses 0.7)")
parser.add_argument("--pi_clip_value", type=float, default=0.1, help="element-wise policy gradient clip after the norm clip (<= 0: off)")
parser.add_argument("--torch_profile", type=int, default=0, help="profile this many iterations after warm-up with torch.profiler, report, exit")
parser.add_argument("--fused_rot", action="store_true", help="elementwise 3x3 products (fused by torch.compile) instead of cuBLAS bmm")
parser.add_argument("--compile_more", action="store_true",
                    help="also compile the per-sim-step MDP heads (reward, termination, obs, goal), the WM normalizer inputs, and the critic bootstrap")  # fmt: skip
parser.add_argument("--cudagraphs", action="store_true", help="torch.compile mode='reduce-overhead' (CUDA graphs) for the update loop")
parser.add_argument("--pi_ema", type=float, default=0.0,
                    help="Polyak average of the policy weights (decay per update); checkpoints store the average as 'pi'")  # fmt: skip
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_contact", type=float, default=0.0,
                    help="contact-mode world model (per-mode increment heads mixed by predicted foot contact); weight of the contact BCE loss")  # fmt: skip
parser.add_argument("--contact_reward", type=float, default=0.0,
                    help="weight of the contact-matching / anti-slip reward term in imagination (evaluation reward unchanged)")
parser.add_argument("--root_reward", type=float, default=0.0,
                    help="weight of a wide-kernel global root position term in imagination (evaluation reward unchanged)")
parser.add_argument("--ref_residual", type=float, default=0.0,
                    help="policy outputs a correction on top of beta x the next reference joint pose (0 = paper: offset from the default pose)")
parser.add_argument("--pi_experts", type=int, default=1, help="policy decoders mixed by a gate on the motion latent (1 = paper)")
parser.add_argument("--wm_ens", type=int, default=1, help="world-model heads on a shared trunk (1 = paper)")
parser.add_argument("--wm_ens_pen", type=float, default=0.0,
                    help="imagination reward penalty on the heads' disagreement (normalized increment variance)")
parser.add_argument("--root_reward_std", type=float, default=1.0, help="kernel width (m) of --root_reward")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--ckpt_minutes", type=float, default=3.0, help="also checkpoint every n minutes of training wall-clock")
parser.add_argument("--max_hours", type=float, default=None, help="stop after this much training wall-clock")
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--motion_dir", type=str, default=None, help="defaults to the robot's motion directory")
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
args = parser.parse_args()

import torch
import torch.nn as nn

from instanttrack.buffer import ReplayBuffer
from instanttrack.dr import XiLayout
from instanttrack.env import PHYSICS_DT, TrackingSim, make_sim
from instanttrack.kinematics import Kinematics
from instanttrack.mdp import MDP, inv_dim, invariant_features, unflatten_state
from instanttrack.motion import CONTROL_DT, MotionLib
import instanttrack.networks as networks
import instanttrack.rotations as rotations
from instanttrack.rotations import mm3
from instanttrack.networks import Critic, Policy, WorldModel
from instanttrack.robot import get_robot
from instanttrack.sensing import ObsPipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

robot = get_robot(args.robot)
J = robot.num_joints
kin = Kinematics(robot, dev)
motion = MotionLib.from_dir(args.motion_dir or robot.motion_dir, robot, kin, dev)
layout = XiLayout(robot, dev)
normalize_xi = layout.normalize
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), layout.dim, robot.sensing)
pipe = ObsPipeline(robot, mdp, layout, dev, PHYSICS_DT)
assert not args.anchored or (args.no_sync and not pipe.stateful), "--anchored needs --no_sync and a stateless sensing model"
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = motion.goal_dim(K)
CRITIC_IN = inv_dim(J) + mdp.critic_extra + GOAL_DIM

widths = lambda s: tuple(int(x) for x in s.split(","))  # noqa: E731
networks.RunningNorm.no_sync = args.no_sync
rotations.ELEMENTWISE = args.fused_rot
wm = WorldModel(J, layout.dim, use_prev_action=robot.has_action_delay, hidden=widths(args.wm_hidden),
                contact_modes=args.wm_contact > 0, ens_heads=args.wm_ens).to(dev)  # fmt: skip
pi = Policy(mdp.obs_dim, GOAL_DIM, J, hist=Hh, enc_hidden=widths(args.enc_hidden), dec_hidden=widths(args.dec_hidden),
            ref_residual=args.ref_residual, experts=args.pi_experts).to(dev)  # fmt: skip
if args.ref_residual > 0:
    pi.set_reference_action(mdp.default_q, torch.tensor(robot.action_scale_list(), device=dev))
critic = Critic(CRITIC_IN).to(dev)
fused = {"fused": True} if args.fused_adam else {}
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr, **fused)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(args.pi_beta1, 0.95), **fused)
pi_avg = torch.optim.swa_utils.AveragedModel(pi, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.pi_ema),
                                             use_buffers=True) if args.pi_ema > 0 else None  # fmt: skip
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr, **fused)
start_it, wall_offset, sim_steps = 0, 0.0, 0
if args.resume:
    ck = torch.load(args.resume, map_location=dev)
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((J,), 3e-3), torch.full((J,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, a_prev, xi):
    S = unflatten_state(s[:, 0])
    preds, logits, heads0 = [], [], None
    for j in range(args.wm_window):
        S, c, hd = wm.step_full(S, a[:, j], a_prev[:, j], normalize_xi(xi[:, j]))
        preds.append(S)
        logits.append(c)
        if j == 0:
            heads0 = hd
    S_hat = tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)
    return S_hat, torch.stack(logits, 1) if wm.contact_modes else None, heads0


def _ens_loss(heads0, s, s1, mask):
    """One-step loss of each head from the logged state, on a random half of the batch per head
    (bootstrap), in normalized increment units; the multi-step loss trains the heads' mean."""
    S0, S1 = unflatten_state(s[:, 0]), unflatten_state(s1[:, 0])
    y = (wm.targets(S0, S1) - wm.out_norm.mean) / wm.out_norm.std()
    e = ((heads0 - y.unsqueeze(1)) ** 2).mean(-1)  # (B, K)
    boot = (torch.rand_like(e) < 0.5).float()
    w = boot * mask[:, :1]
    return (e * w).sum() / w.sum().clamp(min=1)


FEET = [robot.tracked_bodies.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")] if robot.name == "g1" else []
FOOT_CONTACT_Z = 0.045  # ankle-roll link origin below this height: the foot is on the ground (flat foot ~0.035 m)
mdp.feet = FEET
assert args.contact_reward == 0 or FEET, "--contact_reward needs foot bodies (g1)"


def _foot_err(S_hat, S1):
    """Contact-masked foot position (1 cm units) and velocity (0.1 m/s units) errors, summed over feet: (B, k)."""
    B, k = S1[0].shape[:2]
    fl = lambda S: tuple(x.flatten(0, 1) for x in S)  # noqa: E731
    bp_h, _, bv_h, _ = kin.forward(*fl(S_hat))
    bp, _, bv, _ = kin.forward(*fl(S1))
    on_ground = (bp[:, FEET, 2] < FOOT_CONTACT_Z).float()
    e = ((bp_h[:, FEET] - bp[:, FEET]) / 0.01).pow(2).sum(-1) + ((bv_h[:, FEET] - bv[:, FEET]) / 0.1).pow(2).sum(-1)
    return (e * on_ground).sum(-1).view(B, k)


def _contact_bce(logits, S1):
    """BCE of the predicted next-step foot contact against the logged next state: (B, k)."""
    B, k = S1[0].shape[:2]
    bp = kin.forward(*(x.flatten(0, 1) for x in S1))[0]
    on_ground = (bp[:, FEET, 2] < FOOT_CONTACT_Z).float().view(B, k, 2)
    return nn.functional.binary_cross_entropy_with_logits(logits, on_ground, reduction="none").sum(-1)


def _wm_err(S_hat, logits, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    S1 = unflatten_state(s1)
    e = (wm.state_error(S_hat, S1) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    loss = e
    if args.wm_foot_weight > 0:
        loss = e + args.wm_foot_weight * _foot_err(S_hat, S1)
    if logits is not None:
        loss = loss + args.wm_contact * _contact_bce(logits, S1)
    return (loss * mask).sum() / s1.shape[0], e[:, -1]


ENS_U = [None]  # last imagined mean head disagreement (logged with --wm_ens > 1)


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g):
    a = pi(o_hist, g)
    if args.imag_action_noise > 0:
        a = a + args.imag_action_noise * torch.randn_like(a)  # reparameterized: the gradient reaches the mean action
    S1, _, heads = wm.step_full(S, a, a_prev, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    u = wm.disagreement(heads) if heads is not None else torch.zeros_like(S1[0][..., 0])
    return S1, a, o_hist, f1, motion.goal(f1, K), u


def _anchor(P, Q, R1, m):
    """Real next state R1 corrected by the model's predicted effect of the action change, P - Q; where
    the logged trajectory is cut (m = 0) the pure model prediction P is used."""
    mb = m.bool()
    out = []
    for i, (p, q, r) in enumerate(zip(P, Q, R1)):
        if i == 1:  # rotation: left-multiply the predicted world-frame rotation difference
            v = mm3(mm3(p, q.transpose(-1, -2)), r)
        else:
            v = r + p - q
        out.append(torch.where(mb.view(-1, *([1] * (p.dim() - 1))), v, p))
    return tuple(out)


def _imag_step_anchored(S, o_hist, a_prev, f, xi, xi_n, g, Sr, ar, apr, Sr1, m):
    a = pi(o_hist, g)
    if args.imag_action_noise > 0:
        a = a + args.imag_action_noise * torch.randn_like(a)
    P = wm.step(S, a, a_prev, xi_n)
    with torch.no_grad():
        Q = wm.step(Sr, ar, apr, xi_n)
    S1 = _anchor(P, Q, Sr1, m)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K)


def _imag_step_sensed(S, o_hist, a_prev, f, xi, xi_n, g, st):
    """Imagination step with the robot's stateful signal chain (filters, delay) at the control rate."""
    a = pi(o_hist, g)
    S1 = wm.step(S, a, a_prev, xi_n)
    f1 = f + 1
    gyro, accel = pipe.imu_from_states(S, S1, CONTROL_DT) if robot.sensing.imu is not None else (None, None)
    st = pipe.filter_update(st, S1[5], gyro, accel, xi, CONTROL_DT, noise=True, ctrl_rate=True)
    o1, st = pipe.observe(st, pipe.raw_obs(st, S1, a, xi, f1), noise=True, delay=True)
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K), st


def _reward_term(S, f, a, a_prev):
    r = mdp.reward(S, f, a, a_prev, w_contact=args.contact_reward, w_root=args.root_reward, root_std=args.root_reward_std)
    return r, mdp.soft_termination(S, f)


def _gae_targets(V, rews, dones):
    """GAE(lambda) returns over the imagined transitions; V is (H+1, B), bootstrapped from V[-1]."""
    R = V[-1]
    targets = []
    for t in range(HZ - 1, -1, -1):
        R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
        targets.append(R)
    return torch.stack(targets[::-1])


def _critic_loss(X, Y):
    return ((critic(X) - Y) ** 2).mean()


def _wm_norm_data(S0, S1, a, a_prev, xi):
    return wm.inputs(S0, a, a_prev, normalize_xi(xi)), wm.targets(S0, S1)


def _critic_input(S, f, xi_n, g):
    return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)


compile_ = (lambda fn: torch.compile(fn, mode="reduce-overhead")) if args.cudagraphs else torch.compile
wm_rollout = compile_(_wm_rollout)
wm_err = compile_(_wm_err)
ens_loss = compile_(_ens_loss)
imag_step = compile_(_imag_step)
imag_step_sensed = compile_(_imag_step_sensed)
imag_step_anchored = compile_(_imag_step_anchored)
reward_term = compile_(_reward_term)
critic_input_c = compile_(_critic_input)
wm_norm_data_c = torch.compile(_wm_norm_data)
gae_targets_c = compile_(_gae_targets)
critic_loss_c = torch.compile(_critic_loss)  # called repeatedly with backward in between: no CUDA graphs


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat, logits, heads0 = wm_rollout(w["s"], w["a"], w["a_prev"], w["xi"])
    loss, e_last = wm_err(S_hat, logits, w["s1"], w["mask"])
    if heads0 is not None:
        loss = loss + ens_loss(heads0, w["s"], w["s1"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


def sample_valid_starts(B: int):
    if args.no_sync:
        # fixed-size selection without reading the valid count on the host: stable-sort the valid
        # candidates first (only starts within H frames of a clip end are invalid, < 1%)
        s, h, *w = buffer.sample_starts(2 * B, Hh, HZ if args.anchored else 0)
        f = s["f"]
        ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
        idx = torch.sort((~ok).to(torch.uint8), stable=True).indices[:B]
        return ({k: v[idx] for k, v in s.items()}, {k: v[idx] for k, v in h.items()}) + tuple({k: v[idx] for k, v in d.items()} for d in w)
    starts, hists, n = [], [], 0
    while n < B:
        s, h = buffer.sample_starts(2 * B, Hh)
        f = s["f"]
        ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    return cat(starts), cat(hists)


PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))


def policy_update():
    if PROFILE:
        torch.cuda.synchronize(); tq0 = time.time()
    s, h, *win = sample_valid_starts(args.pi_batch)
    if args.anchored:
        win = win[0]
        Wr, Wr1 = unflatten_state(win["s"]), unflatten_state(win["s1"])  # (B, H, ...)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    if pipe.stateful:  # the observations the policy actually saw, and the sensing state after the last one
        o_hist, st = h["o"], pipe.unpack(s["obs_state"])
    else:
        with torch.no_grad():
            o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        o_hist = mdp.add_noise(o_hist)
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    g0 = motion.goal(f, K)
    g = g0
    S0 = S
    Ss, As, Aps, Fs, Gs, Reps, Us = [], [], [], [], [], [], []
    for t in range(HZ):
        Aps.append(a_prev)
        if pipe.stateful:
            S, a_prev, o_hist, f, g, st = imag_step_sensed(S, o_hist, a_prev, f, xi, xi_n, g, st)
        elif args.anchored:
            sel = lambda X: tuple(x[:, t] for x in X)  # noqa: E731
            S, a_prev, o_hist, f, g = imag_step_anchored(S, o_hist, a_prev, f, xi, xi_n, g, sel(Wr), win["a"][:, t], win["a_prev"][:, t],
                                                          sel(Wr1), win["mask"][:, t])  # fmt: skip
        else:
            S, a_prev, o_hist, f, g, u = imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
            Us.append(u)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        if args.demo_replay:
            # DiffMimic-style demonstration replay: a failed imagined state is scored (above) and then
            # replaced by the reference state, so the rest of the horizon still yields gradients
            with torch.no_grad():
                rep = mdp.soft_termination(S, f) > 0.5
                if args.demo_replay_thr > 0:
                    e_pos = mdp.reward(S, f, a_prev, Aps[-1], terms=True)[1]["e_body_pos"]
                    rep = rep | (e_pos > args.demo_replay_thr)
                ref = motion.ref_state(f)
                o_ref = mdp.add_noise(mdp.obs(ref, a_prev, xi, f))
            S = tuple(torch.where(rep.view(-1, *([1] * (x.dim() - 1))), r_.detach(), x) for x, r_ in zip(S, ref))
            o_hist = torch.cat((o_hist[:, :-1], torch.where(rep[:, None], o_ref, o_hist[:, -1]).unsqueeze(1)), 1)
            Reps.append(rep)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
    r, d = r.view(HZ, B), d.view(HZ, B)
    if Us and wm.ens_heads > 1:
        U = torch.stack(Us)
        ENS_U[0] = U.detach().mean()
        if args.wm_ens_pen > 0:
            r = r - args.wm_ens_pen * U * CONTROL_DT
    if args.demo_replay:
        # survival weight restarts at 1 after a replay (the replayed state is a fresh episode start)
        c, cs = torch.ones(B, device=dev), []
        for t in range(HZ):
            cs.append(c)
            c = torch.where(Reps[t], 1.0, c * (1 - d[t]))
        c_t = torch.stack(cs)
        alive = torch.cat((c_t[1:], c.unsqueeze(0)), 0)
    else:
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    if not args.no_critic:
        J = J + (args.gamma**HZ) * c * critic((critic_input_c if args.compile_more else _critic_input)(S, f, xi_n, g))
    loss = -J.mean()
    if PROFILE:
        torch.cuda.synchronize(); tq1 = time.time()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    if PROFILE:
        torch.cuda.synchronize(); tq2 = time.time()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    if args.pi_clip_value > 0:
        nn.utils.clip_grad_value_(pi.parameters(), args.pi_clip_value)
    if args.no_sync:
        # skip a non-finite update without a host read: zero the gradients on the GPU instead
        finite = torch.isfinite(gn)
        for p in pi.parameters():
            if p.grad is not None:
                p.grad.copy_(torch.where(finite, p.grad, 0.0))
        opt_pi.step()
    elif torch.isfinite(gn):
        opt_pi.step()
    if pi_avg is not None:
        pi_avg.update_parameters(pi)
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_loss = torch.zeros(())
    if not args.no_critic:
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        with torch.no_grad():
            targets = (gae_targets_c if args.fused_adam else _gae_targets)(critic(X), rews, dones)  # (H, B)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = (critic_loss_c if args.fused_adam else _critic_loss)(Xf[idx], Yf[idx])
                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_v.step()
    if PROFILE:
        torch.cuda.synchronize()
        print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
    return loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), c.mean().detach()


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim(robot)
if args.compile_more:
    # the simulator wrapper calls these once per control step in eager mode otherwise (FK + rotations: thousands of kernels)
    mdp.reward, mdp.termination, mdp.obs = torch.compile(mdp.reward), torch.compile(mdp.termination), torch.compile(mdp.obs)
    motion.goal = torch.compile(motion.goal)
env = TrackingSim(sim_cfg, args.num_envs, robot, motion, mdp, dev, gamma=args.gamma, fail_reset=args.fail_reset)
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, J, layout.dim,
                      mdp.obs_dim if pipe.stateful else None, pipe.state_dim if pipe.stateful else None)  # fmt: skip
obs_hist = torch.zeros(args.num_envs, Hh, mdp.obs_dim, device=dev)


@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global obs_hist, sim_steps
    for _ in range(n_steps):
        o = env.observe()
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        g = motion.goal(env.f, K)
        pi.obs_norm.update(o)
        pi.goal_norm.update(g)
        a = pi(obs_hist, g)
        a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
        out = env.step(a)
        buffer.add(out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        wm_in, wm_out = (wm_norm_data_c if args.compile_more else _wm_norm_data)(S0, S1, out["a"], out["a_prev"], out["xi"])
        wm.in_norm.update(wm_in)
        wm.out_norm.update(wm_out)
        sim_steps += args.num_envs
        stats["r"].append(out["r"].mean() / 0.1)
        if "ep_len" in out:
            stats["ep_len"].append(out["ep_len"].mean())
            stats["ep_ret"].append(out["ep_ret_norm"].mean())
            stats["term"].append(out["term"].float().sum())
            stats["done"].append(out["done"].float().sum())

stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
collect(args.warmup_steps, stats)  # the replay buffer is not checkpointed, so a resumed run refills it too
log_f = open(os.path.join(args.log_dir, "train_log.csv"), "a")
if start_it == 0:
    log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr\n")
t_start = time.time() - wall_offset
next_ckpt_s = wall_offset + args.ckpt_minutes * 60
t_log = time.time()
step_acc = 0.0
agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
TORCH_PROF = args.torch_profile or int(os.environ.get("SM_TORCH_PROFILE", "0"))
prof_ctx, prof_start = None, args.wm_only_iters + 30
for it in range(start_it, args.iterations):
    if args.cudagraphs:
        torch.compiler.cudagraph_mark_step_begin()
    if TORCH_PROF and it == prof_start:
        torch.cuda.synchronize()
        prof_ctx = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        prof_ctx.__enter__()
        prof_t0 = time.time()
    if TORCH_PROF and it == prof_start + TORCH_PROF:
        torch.cuda.synchronize()
        wall_ms = 1e3 * (time.time() - prof_t0) / TORCH_PROF
        prof_ctx.__exit__(None, None, None)
        ka = prof_ctx.key_averages()
        dev_ms = lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) / 1e3 / TORCH_PROF  # noqa: E731
        kern = [(e.key, dev_ms(e), e.count / TORCH_PROF) for e in ka if dev_ms(e) > 0]
        total = sum(k[1] for k in kern)
        is_gemm = lambda n: any(t in n.lower() for t in ("gemm", "cutlass", "sm90", "sm100", "sm120", "matmul", "nvjet", "ampere", "cublas"))  # noqa: E731
        gemm = sum(k[1] for k in kern if is_gemm(k[0]))
        launches = sum(k[2] for k in kern)
        print(f"[torch-profile] per iteration: wall {wall_ms:.1f} ms | GPU busy {total:.1f} ms ({100 * total / wall_ms:.0f} %) | "
              f"GEMM {gemm:.1f} ms ({100 * gemm / max(total, 1e-9):.0f} % of GPU) | other kernels {total - gemm:.1f} ms | "
              f"kernel launches {launches:.0f}", flush=True)  # fmt: skip
        for name, ms, cnt in sorted(kern, key=lambda k: -k[1])[:25]:
            print(f"[torch-profile] {ms:7.2f} ms  x{cnt:6.0f}  {'GEMM ' if is_gemm(name) else '     '}{name[:110]}", flush=True)
        prof_ctx.export_chrome_trace(os.path.join(args.log_dir, "trace.json"))
        break
    lr = lr_at(it)
    for o in (opt_wm, opt_pi, opt_v):
        for gparam in o.param_groups:
            gparam["lr"] = lr
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
    if prof:
        torch.cuda.synchronize(); tp0 = time.time()
    collect(n, stats)
    if prof:
        torch.cuda.synchronize(); tp1 = time.time()
    for _ in range(args.wm_batches):
        l, e8 = wm_update()
    if prof:
        torch.cuda.synchronize(); tp2 = time.time()
    agg["wm"].append(l), agg["wm8"].append(e8)
    if it >= args.wm_only_iters:
        pl, vl, ir, gn, surv = policy_update()
        if prof:
            torch.cuda.synchronize(); tp3 = time.time()
            print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
        agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
    if (it + 1) % args.log_every == 0:
        mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall_h = (time.time() - t_start) / 3600
        term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
        row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
               mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr]  # fmt: skip
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        log_f.flush()
        print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
              f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s"
              + (f" | ens_u {float(ENS_U[0]):.4g}" if ENS_U[0] is not None else ""), flush=True)  # fmt: skip
        stats = {k: [] for k in stats}
        agg = {k: [] for k in agg}
    wall = time.time() - t_start
    out_of_time = args.max_hours is not None and wall >= args.max_hours * 3600
    timed = args.ckpt_minutes > 0 and wall >= next_ckpt_s
    if timed:
        next_ckpt_s += args.ckpt_minutes * 60
    if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations or timed or out_of_time:
        ck = {"wm": wm.state_dict(), "pi": pi_avg.module.state_dict() if pi_avg is not None else pi.state_dict(), "pi_live": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
              "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
              "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        small = {k: ck[k] for k in ("pi", "pi_live", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        torch.save(small, path + ".tmp")
        os.replace(path + ".tmp", path)
    if out_of_time:
        print(f"time budget of {args.max_hours} h reached at it {it + 1}", flush=True)
        break
