#!/usr/bin/env python3
import argparse
import contextlib
import os
import sys
from typing import List, Optional, Tuple

import torch

# Ensure repo root is importable
REPO_ROOT = "/Users/owner/Desktop/caiatech/models/onyx"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from onyx.model import Onyx, OnyxConfig, StandardAttention, build_packed_causal_mask


def _select_devices(requested: str) -> List[str]:
    if requested != "auto":
        return [requested]
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def _build_seg_id(boundaries: torch.Tensor, S: int, device: torch.device) -> torch.Tensor:
    b = boundaries.to(device=device)
    b = b[b >= 0]
    if b.numel() == 0 or int(b[0]) != 0:
        b = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), b])
    if int(b[-1]) != S:
        b = torch.cat([b, torch.tensor([S], device=device, dtype=torch.int32)])
    b = torch.clamp(b, 0, S)

    seg_id = torch.full((S,), -1, device=device, dtype=torch.int32)
    for i in range(b.numel() - 1):
        start = int(b[i].item())
        end = int(b[i + 1].item())
        if end <= start:
            continue
        seg_id[start:end] = i
    return seg_id


@contextlib.contextmanager
def _capture_softmax(target_shape: Tuple[int, int], device: torch.device):
    import onyx.model as onyx_model

    captured = {"probs": None}
    orig = onyx_model.F.softmax

    def wrapped(x, dim=-1):
        y = orig(x, dim=dim)
        if x.ndim == 4 and x.shape[-2:] == target_shape:
            # capture first matching attention softmax
            if captured["probs"] is None:
                captured["probs"] = y.detach().to(device)
        return y

    onyx_model.F.softmax = wrapped
    try:
        yield captured
    finally:
        onyx_model.F.softmax = orig


def _assert_masked_probs(attn_probs: torch.Tensor, boundaries: torch.Tensor, num_segs: Optional[torch.Tensor]) -> None:
    # attn_probs: [B, H, S, S]
    B, H, S, _ = attn_probs.shape
    for b in range(B):
        if num_segs is not None:
            n = int(num_segs[b].item())
            bnd = boundaries[b, :n]
        else:
            bnd = boundaries[b]
        seg_id = _build_seg_id(bnd, S, attn_probs.device)
        # cross-doc should be exactly zero
        cross = seg_id.unsqueeze(1) != seg_id.unsqueeze(0)
        cross_probs = attn_probs[b].masked_fill(~cross.unsqueeze(0), 0.0)
        max_cross = float(cross_probs.max().item())
        assert max_cross <= 1e-9, f"cross-doc attention prob > 0: {max_cross}"
        # causal within segment: no attention to future positions
        qpos = torch.arange(S, device=attn_probs.device).unsqueeze(1)
        kpos = torch.arange(S, device=attn_probs.device).unsqueeze(0)
        future = kpos > qpos
        future_same_seg = future & (seg_id.unsqueeze(1) == seg_id.unsqueeze(0))
        future_probs = attn_probs[b].masked_fill(~future_same_seg.unsqueeze(0), 0.0)
        max_future = float(future_probs.max().item())
        assert max_future <= 1e-9, f"causal attention prob > 0: {max_future}"


def _run_attention_checks(device: torch.device) -> None:
    torch.manual_seed(0)
    cfg = OnyxConfig(
        d_model=8,
        n_heads=2,
        n_kv_heads=1,
        head_dim=4,
        max_seq_len=16,
        use_flash_attention=False,
        dropout=0.0,
        attention_dropout=0.0,
        use_hope_attention=False,
    )
    attn = StandardAttention(cfg).to(device)
    attn.eval()

    B, S, D = 2, 6, 8
    x = torch.randn(B, S, D, device=device)

    # boundaries: missing 0 / missing S / repeated / -1 padding
    b0 = torch.tensor([3, 6], dtype=torch.int32, device=device)  # missing 0
    b1 = torch.tensor([0, 2, 2, 5], dtype=torch.int32, device=device)  # repeated, missing S
    max_len = 4
    boundaries = torch.full((B, max_len), -1, dtype=torch.int32, device=device)
    boundaries[0, : b0.numel()] = b0
    boundaries[1, : b1.numel()] = b1
    num_segs = torch.tensor([b0.numel(), b1.numel()], dtype=torch.int32, device=device)

    with _capture_softmax((S, S), device=device) as cap:
        out, _, _ = attn(
            x,
            packed_cu_seqlens=boundaries,
            packed_num_segs=num_segs,
            kv_cache=None,
            position_offset=0,
        )
        assert out.shape == (B, S, D)

    probs = cap["probs"]
    assert probs is not None, "failed to capture attention probs"
    _assert_masked_probs(probs, boundaries, num_segs)


def _run_mask_invariants(device: torch.device) -> None:
    S = 6
    boundaries = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
    mask = build_packed_causal_mask(boundaries, S, device)
    assert mask[4, 1].item() is True
    assert mask[4, 3].item() is False
    assert mask[2, 4].item() is True
    assert mask[2, 0].item() is False


def _run_packed_vs_unpacked(device: torch.device) -> Tuple[float, float]:
    torch.manual_seed(0)
    cfg = OnyxConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_ff=64,
        max_seq_len=8,
        train_seq_len=8,
        vocab_size=128,
        use_flash_attention=False,
        dropout=0.0,
        attention_dropout=0.0,
        use_hope_attention=False,
    )
    model = Onyx(cfg).to(device)
    model.eval()

    doc_a = torch.tensor([[5, 6, 7, 8]], dtype=torch.long, device=device)
    doc_b = torch.tensor([[9, 10, 11, 12]], dtype=torch.long, device=device)
    packed = torch.cat([doc_a, doc_b], dim=1)
    labels_packed = packed.clone()
    boundaries = torch.tensor([[0, 4, 8]], dtype=torch.int32, device=device)
    num_segs = torch.tensor([3], dtype=torch.int32, device=device)

    with torch.no_grad():
        out_packed = model(
            input_ids=packed,
            labels=labels_packed,
            packed_cu_seqlens=boundaries,
            packed_num_segs=num_segs,
        )
        input_unpacked = torch.cat([doc_a, doc_b], dim=0)
        labels_unpacked = input_unpacked.clone()
        out_unpacked = model(input_ids=input_unpacked, labels=labels_unpacked)

    loss_packed = float(out_packed["loss"].item())
    loss_unpacked = float(out_unpacked["loss"].item())
    diff = abs(loss_packed - loss_unpacked)
    assert diff < 1e-2, f"packed/unpacked loss diff too large: {diff}"
    return loss_packed, loss_unpacked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    devices = _select_devices(args.device)
    for dev in devices:
        device = torch.device(dev)
        print(f"== device={dev} ==")
        _run_mask_invariants(device)
        _run_attention_checks(device)
        lp, lu = _run_packed_vs_unpacked(device)
        print(f"packed_loss={lp:.6f} unpacked_loss={lu:.6f} diff={abs(lp-lu):.6f}")
        print("OK")


if __name__ == "__main__":
    main()
