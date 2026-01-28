"""
Experimental Hyper-Connections (HC/mHC) utilities.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
import math

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch.compiler import is_compiling as _is_compiling  # torch>=2.1
except Exception:  # pragma: no cover
    try:
        from torch._dynamo import is_compiling as _is_compiling  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        def _is_compiling() -> bool:  # type: ignore[no-redef]
            return False


def sinkhorn_project(matrix: Tensor, iters: int = 10, eps: float = 1e-8) -> Tensor:
    """
    Approximate doubly-stochastic projection via Sinkhorn normalization.

    Expects a non-negative matrix; runs in float32 for stability.
    """
    if matrix.ndim != 2:
        raise ValueError(f"sinkhorn_project expects a 2D tensor, got shape={tuple(matrix.shape)}")

    steps = max(0, int(iters))
    eps = float(eps)
    p = matrix.float()
    p = p.clamp_min(eps)

    for _ in range(steps):
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        p = p / (p.sum(dim=-2, keepdim=True) + eps)
        p = p.clamp_min(eps)

    return p


def aggregate_streams(x_streams: Tensor) -> Tensor:
    """
    Aggregate multi-stream tensor to a single stream.
    """
    n = x_streams.shape[2]
    return x_streams.sum(dim=2) / math.sqrt(n)


def scatter_delta(x_streams: Tensor, delta: Tensor, stream_idx: int = 0) -> Tensor:
    """
    Scatter the block delta back into the multi-stream tensor.
    """
    out = x_streams.clone()
    out[:, :, stream_idx, :] = out[:, :, stream_idx, :] + delta
    return out


class MHCMixer(nn.Module):
    """
    Per-layer stream mixer with optional Sinkhorn projection (mHC).
    """

    def __init__(
        self,
        n_streams: int,
        mode: str = "mhc",
        use_sinkhorn: bool = True,
        sinkhorn_iters: int = 10,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if n_streams < 1:
            raise ValueError("n_streams must be >= 1")
        if mode not in ("mhc", "hc"):
            raise ValueError(f"mode must be 'mhc' or 'hc', got {mode}")

        self.n_streams = int(n_streams)
        self.mode = mode
        self.use_sinkhorn = bool(use_sinkhorn)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.eps = float(eps)
        self.last_P_used_stats: Optional[Dict[str, Any]] = None

        init = torch.eye(self.n_streams)
        self.matrix = nn.Parameter(init)

    def _mix_matrix(self) -> Tensor:
        eps = self.eps
        logits = self.matrix.float()
        p = torch.softmax(logits, dim=-1)
        p = p.clamp_min(eps)
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        if self.mode == "hc" or not self.use_sinkhorn:
            return p
        return sinkhorn_project(p, iters=self.sinkhorn_iters, eps=eps)

    def forward(self, x_streams: Tensor) -> Tensor:
        mix = self._mix_matrix()
        # Avoid torch.compile graph breaks from CPU transfers and scalar .item() logging.
        # These stats are debug-only; skip them while compiling.
        if not _is_compiling():
            with torch.no_grad():
                stats = mix.detach().float().cpu()
                nan = int(torch.isnan(stats).sum().item())
                inf = int(torch.isinf(stats).sum().item())
                finite = torch.isfinite(stats)
                if finite.any():
                    f = stats[finite]
                    mean = float(f.mean().item())
                    vmax = float(f.max().item())
                    vmin = float(f.min().item())
                else:
                    mean = float("nan")
                    vmax = float("nan")
                    vmin = float("nan")
                self.last_P_used_stats = {
                    "shape": tuple(mix.shape),
                    "nan": nan,
                    "inf": inf,
                    "finite": int(finite.sum().item()),
                    "total": int(stats.numel()),
                    "mean": mean,
                    "max": vmax,
                    "min": vmin,
                }
        x = x_streams.float()
        mixed = torch.einsum("b s n d, n m -> b s m d", x, mix)
        return mixed.to(dtype=x_streams.dtype)
