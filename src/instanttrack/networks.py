"""World model, policy (motion encoder + shared decoder), and critic (SuperMimic paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mdp import inv_dim, invariant_features
from .rotations import mm3, mv3, so3_exp, yaw_mat


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    # InstantTrack: when true, the freeze test runs on the GPU (torch.where) instead of a host-side `if`
    no_sync: bool = False

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if not RunningNorm.no_sync and self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        mean = self.mean + delta * n / tot
        var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        if RunningNorm.no_sync:
            live = self.count < self.max_count
            self.mean.copy_(torch.where(live, mean, self.mean))
            self.var.copy_(torch.where(live, var, self.var))
            self.count.copy_(torch.where(live, tot, self.count))
        else:
            self.mean.copy_(mean)
            self.var = var
            self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments.

    With ``use_prev_action`` the model also sees a_{t-1}: delayed actuators still track the previous
    action's target during the first physics steps of a control step.
    """

    def __init__(self, num_j: int, xi_dim: int, use_prev_action: bool = False, hidden=(2048, 2048, 2048),
                 contact_modes: bool = False, structured: bool = False, dt: float = 0.02):  # fmt: skip
        """``contact_modes`` (InstantTrack): the last layer predicts the two feet's next-step ground contact
        and one increment head per contact mode (flight, left, right, double support); the increment is the
        mixture of the heads weighted by the predicted mode probabilities, so each head stays smooth within
        its mode instead of one MLP smoothing across touch-down and lift-off."""
        super().__init__()
        self.num_j = num_j
        self.use_prev_action = use_prev_action
        self.contact_modes = contact_modes
        # ``structured`` = semi-implicit integration (positions follow the predicted velocities,
        # the net predicts velocity increments plus small position corrections)
        self.structured, self.dt = structured, dt
        self.in_dim = inv_dim(num_j) + num_j * (2 if use_prev_action else 1) + xi_dim
        self.out_dim = 3 + 3 + 3 + 3 + num_j + num_j
        n_out = 4 * self.out_dim + 2 if contact_modes else self.out_dim
        self.net = mlp(self.in_dim, list(hidden), n_out, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)
        # scale of the plain frame-local increments, used to normalize the training loss (equal to out_norm
        # unless ``structured`` changes what the network outputs)
        self.inc_norm = RunningNorm(self.out_dim, eps=1e-4)

    def inputs(self, S, a, a_prev, xi_n):
        if self.use_prev_action:
            return torch.cat((invariant_features(S), a, a_prev, xi_n), -1)
        return torch.cat((invariant_features(S), a, xi_n), -1)

    # weight of the new velocity in the position drift: n semi-implicit substeps with a linearly changing
    # velocity give sum_k k/n / n = (n + 1) / (2 n); n = 4 substeps -> 0.625 (measured residual std of the
    # joint-position increment: 16 % with 0.5, 11 % with 0.625, 17 % with 1.0)
    DRIFT_W = 0.625

    def _drift(self, S, v1, w1, qd1):
        """Semi-implicit position update implied by the old and new velocities, in the frames of the
        increments: yaw frame for the root position, body frame for the rotation vector."""
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        a, b = self.dt * (1 - self.DRIFT_W), self.dt * self.DRIFT_W
        return (mv3(H.transpose(-1, -2), a * v + b * v1), mv3(R.transpose(-1, -2), a * w + b * w1), a * qd + b * qd1)

    def targets(self, S, S1):
        """Regression targets of the network: the frame-local increments, or with ``structured`` the
        velocity increments plus the position corrections on top of the semi-implicit drift."""
        inc = self.increments(S, S1)
        if not self.structured:
            return inc
        J = self.num_j
        dp, dth, dv, dw, dq, dqd = inc.split((3, 3, 3, 3, J, J), -1)
        gp, gth, gq = self._drift(S, S1[2], S1[3], S1[5])
        return torch.cat((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)

    @staticmethod
    def increments(S, S1):
        """Frame-local increments from S to S1."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                mv3(H.transpose(-1, -2), p1 - p),
                so3_log(mm3(Rt, R1)),
                mv3(Rt, v1 - v),
                mv3(Rt, w1 - w),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, a_prev, xi_n):
        return self.step_c(S, a, a_prev, xi_n)[0]

    def step_c(self, S, a, a_prev, xi_n):
        """Next state and next-step foot-contact logits (B, 2), or None without ``contact_modes``."""
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, a_prev, xi_n)))
        logits = None
        if self.contact_modes:
            logits = y[..., -2:]
            cl, cr = torch.sigmoid(logits).unbind(-1)
            mode_w = torch.stack(((1 - cl) * (1 - cr), cl * (1 - cr), (1 - cl) * cr, cl * cr), -1)
            y = (y[..., :-2].unflatten(-1, (4, self.out_dim)) * mode_w.unsqueeze(-1)).sum(-2)
        return self.integrate(S, y * self.out_norm.std() + self.out_norm.mean), logits

    def integrate(self, S, d):
        """Next state from the denormalized network output ``d`` (the layout of ``targets``)."""
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.num_j, self.num_j), -1)
        v1 = v + mv3(R, dv)
        w1 = w + mv3(R, dw)
        qd1 = qd + dqd
        if self.structured:
            gp, gth, gq = self._drift(S, v1, w1, qd1)
            dp, dth, dq = dp + gp, dth + gth, dq + gq
        p1 = p + mv3(H, dp)
        R1 = mm3(R, so3_exp(dth))
        return (p1, R1, v1, w1, q + dq, qd1)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                mv3(H.transpose(-1, -2), ph - p),
                so3_log(mm3(Rt, Rh)),
                mv3(Rt, vh - v),
                mv3(Rt, wh - w),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh.

    ``ref_skip`` = (offset, q0, alpha) (--pi_ref_skip): reference-action skip connection,
    a = (q_ref(f+1) - q0) / alpha + dec(...), with q_ref(f+1) read from the raw (unnormalized) goal vector at
    ``offset``; the network only learns the correction to tracking the reference with the PD controller."""

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hist: int = 10, z_dim: int = 32,
                 enc_hidden=(2048, 1024, 512, 256), dec_hidden=(2048, 1024, 512, 256), ref_skip=None):  # fmt: skip
        super().__init__()
        self.hist = hist
        self.ref_off = None
        if ref_skip is not None:
            self.ref_off = ref_skip[0]
            self.register_buffer("ref_q0", torch.tensor(ref_skip[1], dtype=torch.float32))
            self.register_buffer("ref_alpha", torch.tensor(ref_skip[2], dtype=torch.float32))
        self.enc = mlp(goal_dim, list(enc_hidden), z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, list(dec_hidden), act_dim)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last. ``z`` (optional) is
        ``encode(goal)`` computed beforehand."""
        o = self.obs_norm(obs_hist).flatten(-2)
        if z is None:
            z = self.encode(goal)
        a = run_mlp(self.dec, torch.cat((z, o), -1))
        if self.ref_off is not None:
            q_ref = goal[..., self.ref_off : self.ref_off + a.shape[-1]]
            a = a + (q_ref - self.ref_q0) / self.ref_alpha
        return a

    def encode(self, goal: torch.Tensor) -> torch.Tensor:
        """Motion latent z from the goal features (reference only, independent of the robot state)."""
        return torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))


# Switches of the development code that this release does not have, with their "off" value. Run arguments (args.json,
# the "args" of a checkpoint) of that code still carry them; at the off value they are ignored, any other value
# describes a run that this code cannot rebuild or continue.
PRUNED_OFF = {
    "actor_lam": 1.0, "grad_decay": 1.0, "anchored": False, "demo_replay": False,
    "imag_action_noise": 0.0, "wm_prio": 0.0, "imag_prio": 0.0, "critic_alive": False, "critic_prior": False,
    "critic_grad": 0.0, "critic_ema": 0.0, "lookahead_k": 0, "wm_batches_early": 0, "wm_foot_weight": 0.0,
    "bc_sharp_weight": 0.0, "pi_grad_batch": 0, "no_critic": False, "wm_imag_ema": 0.0, "anchor_coef": 0.0,
    "boot_coef": 1.0, "grad_clip_step": 0.0, "det_frac": 0.0, "wm_load": None, "wm_freeze": False, "wm_ready": 0.0,
    # these also change the networks (inputs or architecture)
    "wm_pd": False, "pi_arch": "mlp", "pi_fourier": None, "pi_goal_feet": False, "wm_contact": 0.0,
}  # fmt: skip
PRUNED_NET = ("wm_pd", "pi_arch", "pi_fourier", "pi_goal_feet", "wm_contact")


def check_pruned(a_: dict, nets_only: bool = False, what: str = "this run") -> None:
    """Raise if the run arguments ``a_`` use a removed switch. ``nets_only``: only the switches that change the
    networks (enough to rebuild and evaluate a policy / world model); otherwise also every removed training switch
    (for tools that recompute the training objective)."""
    keys = PRUNED_NET if nets_only else PRUNED_OFF
    bad = {k: a_[k] for k in keys if k in a_ and a_[k] != PRUNED_OFF[k] and not (PRUNED_OFF[k] is None and not a_[k])}
    if bad:
        raise ValueError(f"{what} uses switches that this code does not have: {bad}")


def policy_kwargs(a_: dict, robot) -> dict:
    """Policy keyword arguments of a run with the training arguments ``a_``."""
    check_pruned(a_, nets_only=True)
    return {"ref_skip": policy_ref_skip(a_, robot)}


def policy_ref_skip(a_: dict, robot):
    """Policy(ref_skip=...) argument of a run with the training arguments ``a_`` (None without --pi_ref_skip)."""
    if not a_.get("pi_ref_skip", False):
        return None
    from .motion import GOAL_Q_OFFSET

    return (GOAL_Q_OFFSET, robot.default_joint_pos_list(), robot.action_scale_list())


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
