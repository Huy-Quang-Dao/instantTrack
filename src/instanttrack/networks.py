"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4)."""

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
                 contact_modes: bool = False):  # fmt: skip
        """``contact_modes`` (InstantTrack): the last layer predicts the two feet's next-step ground contact
        and one increment head per contact mode (flight, left, right, double support); the increment is the
        mixture of the heads weighted by the predicted mode probabilities, so each head stays smooth within
        its mode instead of one MLP smoothing across touch-down and lift-off."""
        super().__init__()
        self.num_j = num_j
        self.use_prev_action = use_prev_action
        self.contact_modes = contact_modes
        self.in_dim = inv_dim(num_j) + num_j * (2 if use_prev_action else 1) + xi_dim
        self.out_dim = 3 + 3 + 3 + 3 + num_j + num_j
        n_out = 4 * self.out_dim + 2 if contact_modes else self.out_dim
        self.net = mlp(self.in_dim, list(hidden), n_out, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)

    def inputs(self, S, a, a_prev, xi_n):
        if self.use_prev_action:
            return torch.cat((invariant_features(S), a, a_prev, xi_n), -1)
        return torch.cat((invariant_features(S), a, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
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
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.num_j, self.num_j), -1)
        p1 = p + mv3(H, dp)
        R1 = mm3(R, so3_exp(dth))
        v1 = v + mv3(R, dv)
        w1 = w + mv3(R, dw)
        return (p1, R1, v1, w1, q + dq, qd + dqd), logits

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
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh."""

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hist: int = 10, z_dim: int = 32,
                 enc_hidden=(2048, 1024, 512, 256), dec_hidden=(2048, 1024, 512, 256)):  # fmt: skip
        super().__init__()
        self.hist = hist
        self.enc = mlp(goal_dim, list(enc_hidden), z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, list(dec_hidden), act_dim)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        o = self.obs_norm(obs_hist).flatten(-2)
        return run_mlp(self.dec, torch.cat((z, o), -1))


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
