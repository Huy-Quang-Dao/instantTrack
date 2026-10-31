"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from .mdp import state_dim


def fields(num_j: int, xi_dim: int, obs_dim: int | None = None, obs_state_dim: int | None = None) -> dict:
    out = {
        "s": (state_dim(num_j), torch.float32),
        "s1": (state_dim(num_j), torch.float32),
        "a": (num_j, torch.float32),
        "a_prev": (num_j, torch.float32),
        "xi": (xi_dim, torch.float32),
        "r": ((), torch.float32),
        "f": ((), torch.long),
        "ep_step": ((), torch.int32),
        "done": ((), torch.bool),
        "cut": ((), torch.bool),
    }
    if obs_dim is not None:  # stateful sensing: the observation as seen and the sensing state after it
        out["o"] = (obs_dim, torch.float32)
        out["obs_state"] = (obs_state_dim, torch.float32)
    return out


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str, num_j: int, xi_dim: int, obs_dim: int | None = None,
                 obs_state_dim: int | None = None):  # fmt: skip
        self.T, self.N, self.device = T, N, device
        self.fields = fields(num_j, xi_dim, obs_dim, obs_state_dim)
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0
        # InstantTrack (CUDA graphs): a device copy of ``ptr`` so that captured sampling follows the write head
        self.ptr_t = torch.zeros((), dtype=torch.long, device=device) if device != "cpu" else None

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)
        if self.ptr_t is not None:
            self.ptr_t.fill_(self.ptr)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        ptr = self.ptr_t if self.ptr_t is not None else self.ptr
        return (ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "a_prev", "xi", "cut")}
        # transition j is valid if the state stayed continuous (no episode end, no push) after 0..j-1
        d = out["cut"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int, fwd: int = 0):
        """Start points for imagination plus the indices of their observation history (oldest first).

        With ``fwd > 0`` also returns the logged ``fwd``-step window that follows each start (keys ``s``,
        ``s1``, ``a``, ``a_prev`` of shape (B, fwd, ...) and ``mask``: transition j is valid if the
        state stayed continuous after 0..j-1), for imagination anchored to real trajectories."""
        age = torch.randint(0, self.size - fwd + (1 if fwd else 0), (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        stateful = "o" in self.data
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f") + (("obs_state",) if stateful else ())}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f") + (("o",) if stateful else ())}
        if not fwd:
            return start, hist_d
        wrows = self._row(age.unsqueeze(-1) + torch.arange(fwd, device=self.device))
        wenv = env.unsqueeze(-1).expand_as(wrows)
        win = {k: self.data[k][wrows, wenv] for k in ("s", "s1", "a", "a_prev", "cut")}
        d = win.pop("cut").float()
        win["mask"] = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        return start, hist_d, win
