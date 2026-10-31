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


class SimBaBlock(nn.Module):
    """Pre-LayerNorm inverted-bottleneck residual block (SimBa, Lee et al., ICLR 2025)."""

    def __init__(self, h: int, expand: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(h)
        self.fc1, self.fc2 = nn.Linear(h, expand * h), nn.Linear(expand * h, h)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(self.ln(x))))


def simba(inp: int, h: int, blocks: int, out: int) -> nn.Sequential:
    """SimBa stack: linear embedding, residual blocks, post-LayerNorm, linear head."""
    return nn.Sequential(nn.Linear(inp, h), *[SimBaBlock(h) for _ in range(blocks)], nn.LayerNorm(h), nn.Linear(h, out))


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
                 contact_modes: bool = False, ens_heads: int = 1):  # fmt: skip
        """``contact_modes`` (InstantTrack): the last layer predicts the two feet's next-step ground contact
        and one increment head per contact mode (flight, left, right, double support); the increment is the
        mixture of the heads weighted by the predicted mode probabilities, so each head stays smooth within
        its mode instead of one MLP smoothing across touch-down and lift-off.

        ``ens_heads`` K > 1 (InstantTrack): a shared trunk with K increment heads; the model predicts their
        mean, and their spread (variance in normalized increment units) measures epistemic uncertainty."""
        super().__init__()
        self.num_j = num_j
        self.use_prev_action = use_prev_action
        self.contact_modes = contact_modes
        self.ens_heads = ens_heads
        assert not (contact_modes and ens_heads > 1)
        self.in_dim = inv_dim(num_j) + num_j * (2 if use_prev_action else 1) + xi_dim
        self.out_dim = 3 + 3 + 3 + 3 + num_j + num_j
        n_out = 4 * self.out_dim + 2 if contact_modes else ens_heads * self.out_dim
        self.net = mlp(self.in_dim, list(hidden), n_out, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if ens_heads > 1:  # distinct heads (the mean still starts near zero)
            nn.init.normal_(self.net[-1].weight, std=1e-2)
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
        S1, logits, _ = self.step_full(S, a, a_prev, xi_n)
        return S1, logits

    def step_full(self, S, a, a_prev, xi_n):
        """Next state, contact logits (or None) and the per-head normalized increments (B, K, D) (or None)."""
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, a_prev, xi_n)))
        heads = None
        if self.ens_heads > 1:
            heads = y.unflatten(-1, (self.ens_heads, self.out_dim))
            y = heads.mean(-2)
        S1, logits = self.integrate(S, y)
        return S1, logits, heads

    @staticmethod
    def disagreement(heads):
        """Mean over dimensions of the across-head variance, in normalized increment units: (B,)."""
        return heads.var(-2, unbiased=False).mean(-1)

    def integrate(self, S, y):
        p, R, v, w, q, qd = S
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
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh.

    ``ref_residual`` > 0 (InstantTrack): the decoder outputs a correction on top of the next reference
    joint pose read from the goal, a = dec(...) + beta * (q_ref(f+1) - q_default) / alpha, so the
    untrained policy already commands the reference. ``q_default`` and ``alpha`` are buffers (set by
    ``set_reference_action``, then saved in the state dict)."""

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, hist: int = 10, z_dim: int = 32,
                 enc_hidden=(2048, 1024, 512, 256), dec_hidden=(2048, 1024, 512, 256), ref_residual: float = 0.0,
                 experts: int = 1, simba_arch: str = ""):  # fmt: skip
        """``experts`` E > 1 (InstantTrack): E decoders mixed by a softmax gate on the motion latent z, so
        dissimilar motions (walking, dancing, getting up) need not share one decoder.

        ``simba_arch`` "<width>x<blocks>" (InstantTrack): encoder and decoder are SimBa residual stacks instead of
        plain MLPs (the observation / goal running normalizers stay as in SimBa)."""
        super().__init__()
        self.hist = hist
        self.ref_residual = ref_residual
        self.experts = experts
        if ref_residual > 0:
            self.register_buffer("q_default", torch.zeros(act_dim))
            self.register_buffer("alpha", torch.ones(act_dim))
        if simba_arch:
            sw, sb = (int(x) for x in simba_arch.split("x"))
            self.enc = simba(goal_dim, sw, sb, z_dim)
            self.dec = simba(z_dim + obs_dim * hist, sw, sb, act_dim)
        else:
            self.enc = mlp(goal_dim, list(enc_hidden), z_dim)
            self.dec = mlp(z_dim + obs_dim * hist, list(dec_hidden), act_dim)
        if experts > 1:
            assert not simba_arch
            self.decs = nn.ModuleList([self.dec] + [mlp(z_dim + obs_dim * hist, list(dec_hidden), act_dim) for _ in range(experts - 1)])
            self.gate = nn.Linear(z_dim, experts)
            nn.init.zeros_(self.gate.weight), nn.init.zeros_(self.gate.bias)
        with torch.no_grad():
            for d in self.decs if experts > 1 else [self.dec]:
                d[-1].weight.mul_(0.01)
                d[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last. ``z`` (optional) is ``encode(goal)``
        computed beforehand, e.g. for a whole imagined horizon at once."""
        if z is None:
            z = self.encode(goal)
        o = self.obs_norm(obs_hist).flatten(-2)
        x = torch.cat((z, o), -1)
        if self.experts > 1:
            gw = torch.softmax(self.gate(z), -1)
            a = sum(gw[..., e : e + 1] * run_mlp(d, x) for e, d in enumerate(self.decs))
        else:
            a = run_mlp(self.dec, x)
        if self.ref_residual > 0:  # goal frame 1 = [dp 3, rot 6, v 3, w 3, q J, qd J]
            q_next = goal[..., 15 : 15 + a.shape[-1]]
            a = a + self.ref_residual * (q_next - self.q_default) / self.alpha
        return a

    def encode(self, goal: torch.Tensor) -> torch.Tensor:
        """Motion latent z from goal features (reference only, independent of the robot state)."""
        return torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))

    def set_reference_action(self, q_default: torch.Tensor, alpha: torch.Tensor):
        self.q_default.copy_(q_default), self.alpha.copy_(alpha)


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
