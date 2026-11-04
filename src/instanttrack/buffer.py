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
        # --cuda_graph: a device copy of ``ptr``, so that sampling captured in a CUDA graph follows the write head
        self.ptr_t = None
        # --graph_early: a device copy of ``size`` too; sampling bounds follow it (uniform draws scaled by the fill level
        # instead of randint with a host bound), so the captured updates stay valid while the buffer fills
        self.size_t = None

    def use_device_ptr(self, device_size: bool = False):
        self.ptr_t = torch.full((), self.ptr, dtype=torch.long, device=self.device)
        if device_size:
            self.size_t = torch.full((), self.size, dtype=torch.long, device=self.device)

    def _randint(self, n, B: int) -> torch.Tensor:
        """Uniform integers in [0, n) with n a host int, or (--graph_early) a device tensor."""
        if self.size_t is None:
            return torch.randint(0, n, (B,), device=self.device)
        return torch.minimum((torch.rand(B, device=self.device) * n).long(), n - 1)

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)
        if self.ptr_t is not None:
            self.ptr_t.fill_(self.ptr)
        if self.size_t is not None:
            self.size_t.fill_(self.size)

    def add_device(self, tr: dict):
        """add() at the device-side write head (train.py --collect_graph: capturable). Advances ptr_t / size_t on the
        device; the host-side ptr / size are advanced by advance_host() after each replay."""
        row = self.ptr_t.view(1)
        for k in self.fields:
            self.data[k].index_copy_(0, row, tr[k].unsqueeze(0).to(self.data[k].dtype))  # add() casts the same way
        self.ptr_t.copy_((self.ptr_t + 1) % self.T)
        self.size_t.copy_(torch.clamp(self.size_t + 1, max=self.T))

    def advance_host(self):
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        ptr = self.ptr_t if self.ptr_t is not None else self.ptr
        size = self.size_t if self.size_t is not None else self.size
        return (ptr - size + age) % self.T

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = self._randint((self.size_t if self.size_t is not None else self.size) - k + 1, B)
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

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        age = self._randint(self.size_t if self.size_t is not None else self.size, B)
        env = torch.randint(0, self.N, (B,), device=self.device)
        return self.starts_at(age, env, hist)

    def starts_at(self, age: torch.Tensor, env: torch.Tensor, hist: int):
        """Start points at the transitions (age, env) (age 0 = oldest) plus their observation history (oldest first)."""
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
        return start, hist_d
