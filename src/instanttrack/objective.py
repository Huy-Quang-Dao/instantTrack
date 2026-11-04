"""Actor objective over an imagined horizon."""

from __future__ import annotations

import torch


def actor_objective(r: torch.Tensor, d: torch.Tensor, v_boot: torch.Tensor | None, gamma: float, loop_cumprod: bool = False):
    """SHAC objective J = sum_{t<H} gamma^t c_t r_t + gamma^H c_H V(z_H), c_t = prod_{k<t} (1 - d_k).
    r, d: (H, B); v_boot: (B,) critic value of the last imagined state, or None (no bootstrap). Returns (J, c_t, c_H).
    ``loop_cumprod`` (--cuda_graph): the survival products as a loop of multiplications (same products; cumprod's
    backward tests for zeros on the host, which a CUDA graph cannot capture)."""
    H, B = r.shape
    if loop_cumprod:
        acc, cs = torch.ones_like(d[0]), []
        for t in range(H):
            acc = acc * (1 - d[t])
            cs.append(acc)
        alive = torch.stack(cs)
    else:
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=r.device), alive[:-1]), 0)
    disc = gamma ** torch.arange(H, device=r.device, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    if v_boot is not None:
        J = J + (gamma**H) * c * v_boot
    return J, c_t, c
