"""
Experimental Hyper-Connections (HC/mHC) utilities.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
import math
import os

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


def _maybe_sanitize(name: str, tensor: Tensor) -> Tensor:
    if os.getenv("ONYX_MHC_DEBUG_NONFINITE") != "1":
        return tensor
    if torch.isfinite(tensor).all():
        return tensor
    data = tensor.detach().float()
    nan = int(torch.isnan(data).sum().item())
    inf = int(torch.isinf(data).sum().item())
    finite = torch.isfinite(data)
    if finite.any():
        f = data[finite]
        mean = float(f.mean().item())
        vmax = float(f.max().item())
        vmin = float(f.min().item())
    else:
        mean = float("nan")
        vmax = float("nan")
        vmin = float("nan")
    print(f"[MHC][NonFinite] {name} nan={nan} inf={inf} mean={mean:.4e} max={vmax:.4e} min={vmin:.4e}")
    return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)


def sinkhorn_project(matrix: Tensor, iters: int = 10, eps: float = 1e-8) -> Tensor:
    """
    Approximate doubly-stochastic projection via Sinkhorn normalization.

    Expects a non-negative matrix; runs in float32 for stability.
    """
    if matrix.ndim != 2:
        raise ValueError(f"sinkhorn_project expects a 2D tensor, got shape={tuple(matrix.shape)}")

    steps = max(0, int(iters))
    eps = float(eps)
    device_type = matrix.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        p = matrix.float()
        p = p.clamp_min(eps)
        p = _maybe_sanitize("sinkhorn_init", p)

        for i in range(steps):
            p = p / (p.sum(dim=-1, keepdim=True) + eps)
            p = p / (p.sum(dim=-2, keepdim=True) + eps)
            p = p.clamp_min(eps)
            p = _maybe_sanitize(f"sinkhorn_iter_{i}", p)

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
        device_type = self.matrix.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            logits = self.matrix.float()
            logits = logits - logits.max(dim=-1, keepdim=True).values
            logits = logits.clamp(min=-50.0, max=50.0)
            p = torch.softmax(logits, dim=-1)
            p = p.clamp_min(eps)
            p = p / (p.sum(dim=-1, keepdim=True) + eps)
            if self.mode == "hc" or not self.use_sinkhorn:
                return _maybe_sanitize("mhc_mix", p)
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
        device_type = x_streams.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            x = x_streams.float()
            mixed = torch.einsum("b s n d, n m -> b s m d", x, mix.float())
        return mixed.to(dtype=x_streams.dtype)


def _build_lite_permutations(n_streams: int) -> Tensor:
    """
    Build a compact deterministic bank of permutation matrices.

    Includes identity, reverse, cyclic shifts, and adjacent swaps.
    """
    n = int(n_streams)
    perms = []
    seen = set()

    def add_perm(order: list[int]) -> None:
        key = tuple(order)
        if key in seen:
            return
        seen.add(key)
        perms.append(order)

    base = list(range(n))
    add_perm(base)
    add_perm(list(reversed(base)))
    for s in range(1, n):
        add_perm(base[s:] + base[:s])
    for i in range(n - 1):
        o = list(base)
        o[i], o[i + 1] = o[i + 1], o[i]
        add_perm(o)

    mats = []
    for order in perms:
        p = torch.zeros((n, n), dtype=torch.float32)
        for r, c in enumerate(order):
            p[r, c] = 1.0
        mats.append(p)
    return torch.stack(mats, dim=0)  # [K, N, N]


class MHCLiteMixer(nn.Module):
    """
    Lightweight doubly-stochastic mixer by construction.

    Uses a convex combination of fixed permutation matrices (Birkhoff subset),
    avoiding iterative Sinkhorn normalization in the forward path.
    """

    def __init__(self, n_streams: int, eps: float = 1e-8) -> None:
        super().__init__()
        if n_streams < 1:
            raise ValueError("n_streams must be >= 1")
        self.n_streams = int(n_streams)
        self.eps = float(eps)
        self.last_P_used_stats: Optional[Dict[str, Any]] = None

        perm_bank = _build_lite_permutations(self.n_streams)
        self.register_buffer("perm_bank", perm_bank, persistent=False)
        init_logits = torch.zeros((perm_bank.shape[0],), dtype=torch.float32)
        init_logits[0] = 1.0  # bias toward identity at init
        self.perm_logits = nn.Parameter(init_logits)

    def _mix_matrix(self) -> Tensor:
        eps = self.eps
        device_type = self.perm_logits.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            w = torch.softmax(self.perm_logits.float(), dim=0).clamp_min(eps)
            w = w / (w.sum() + eps)
            mix = torch.einsum("k,kij->ij", w, self.perm_bank.float())
            # Keep strict numerical hygiene.
            mix = mix.clamp_min(eps)
            mix = mix / (mix.sum(dim=-1, keepdim=True) + eps)
            mix = mix / (mix.sum(dim=-2, keepdim=True) + eps)
        return _maybe_sanitize("mhc_lite_mix", mix)

    def forward(self, x_streams: Tensor) -> Tensor:
        mix = self._mix_matrix()
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
        device_type = x_streams.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            x = x_streams.float()
            mixed = torch.einsum("b s n d, n m -> b s m d", x, mix.float())
        return mixed.to(dtype=x_streams.dtype)
