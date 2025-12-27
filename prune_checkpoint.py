#!/usr/bin/env python3
import argparse
import math
from typing import Any, Dict, Iterable, List, Tuple

import torch


_SKIP_SUBSTRINGS = (
    ".norm",       # RMSNorm weights
    ".embed",      # embeddings
    ".lm_head",    # output head (often tied)
    ".memory.",    # delta memory internals
    ".k_memory.",  # self-referential memory
    ".v_memory.",  # self-referential memory
    ".conv.",      # short conv
)


def _extract_state_dict(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], bool]:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], True
    if isinstance(ckpt, dict):
        return ckpt, False
    raise ValueError("Checkpoint did not contain a state_dict-like mapping.")


def _is_prunable(name: str, tensor: torch.Tensor) -> bool:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return False
    if tensor.ndim < 2:
        return False
    if not name.startswith("layers."):
        return False
    if not name.endswith(".weight"):
        return False
    if any(s in name for s in _SKIP_SUBSTRINGS):
        return False
    return True


def _gather_prunable(
    sd: Dict[str, torch.Tensor],
) -> List[Tuple[str, torch.Tensor]]:
    return [(k, v) for k, v in sd.items() if _is_prunable(k, v)]


def _flatten_abs(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    flats = [t.abs().reshape(-1) for t in tensors]
    if not flats:
        return torch.tensor([])
    return torch.cat(flats, dim=0)


def _count_zeros(tensors: Iterable[torch.Tensor]) -> int:
    total = 0
    for t in tensors:
        total += int((t == 0).sum().item())
    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Prune an Onyx checkpoint by global magnitude.")
    p.add_argument("--input_ckpt", type=str, required=True)
    p.add_argument("--output_ckpt", type=str, required=True)
    p.add_argument("--sparsity", type=float, default=0.3)
    args = p.parse_args()

    if not (0.0 < args.sparsity < 1.0):
        raise ValueError("--sparsity must be in (0, 1).")

    ckpt = torch.load(args.input_ckpt, map_location="cpu", weights_only=False)
    sd, has_wrapper = _extract_state_dict(ckpt)

    prunable = _gather_prunable(sd)
    if not prunable:
        raise ValueError("No prunable tensors found with current rules.")

    tensors = [t for _, t in prunable]
    total_params = sum(t.numel() for t in tensors)
    to_prune = int(math.floor(total_params * args.sparsity))
    if to_prune <= 0:
        print("Sparsity too low; nothing to prune.")
        torch.save(ckpt, args.output_ckpt)
        return

    abs_flat = _flatten_abs(tensors)
    if abs_flat.numel() != total_params:
        raise RuntimeError("Unexpected tensor flattening mismatch.")

    kth = torch.kthvalue(abs_flat, k=to_prune).values.item()
    pre_zeros = _count_zeros(tensors)

    for name, t in prunable:
        mask = t.abs() <= kth
        t[mask] = 0

    post_zeros = _count_zeros(tensors)
    newly_pruned = post_zeros - pre_zeros
    actual = post_zeros / float(total_params)

    meta = {
        "sparsity_target": args.sparsity,
        "sparsity_actual": actual,
        "total_prunable": total_params,
        "zeros_before": pre_zeros,
        "zeros_after": post_zeros,
        "threshold": kth,
        "pruned_params": newly_pruned,
    }

    if has_wrapper:
        ckpt["prune_meta"] = meta
        torch.save(ckpt, args.output_ckpt)
    else:
        out = dict(sd)
        out["prune_meta"] = meta
        torch.save(out, args.output_ckpt)

    print("Pruning complete.")
    print(f"Target sparsity: {args.sparsity:.4f}")
    print(f"Actual sparsity: {actual:.4f}")
    print(f"Pruned params: {newly_pruned:,} / {total_params:,}")
    print(f"Threshold: {kth:.6g}")
    print(f"Saved pruned checkpoint: {args.output_ckpt}")


if __name__ == "__main__":
    main()
