#!/usr/bin/env python3
"""
Average the weights of the last N checkpoints for evaluation stability.
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict

import torch


def _step_num(path: Path) -> int:
    m = re.search(r"_step_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    raise ValueError(f"{path} does not contain a model_state_dict")


def _average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = list(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if sd.keys() != state_dicts[0].keys():
            raise ValueError("Checkpoint key sets do not match; cannot average.")

    averaged: Dict[str, torch.Tensor] = {}
    count = float(len(state_dicts))
    for k in keys:
        t0 = state_dicts[0][k]
        if not torch.is_tensor(t0):
            raise ValueError(f"Non-tensor key in state_dict: {k}")
        if torch.is_floating_point(t0) or torch.is_complex(t0):
            acc = t0.to(dtype=torch.float32)
            for sd in state_dicts[1:]:
                acc = acc + sd[k].to(dtype=torch.float32)
            averaged[k] = (acc / count).to(dtype=t0.dtype)
        else:
            averaged[k] = t0
    return averaged


def main() -> None:
    p = argparse.ArgumentParser(description="Average the last N checkpoints.")
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--pattern", type=str, default="checkpoint_step_*.pt")
    p.add_argument("--last_n", type=int, default=10)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    paths = sorted(ckpt_dir.glob(args.pattern), key=_step_num, reverse=True)
    if not paths:
        raise SystemExit(f"No checkpoints found in {ckpt_dir} with pattern {args.pattern}")

    selected = paths[: max(1, int(args.last_n))]
    print(f"Averaging {len(selected)} checkpoints:")
    for pth in selected:
        print(f"  - {pth}")

    state_dicts = [_load_state_dict(p) for p in selected]
    averaged = _average_state_dicts(state_dicts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": averaged,
            "num_averaged": len(selected),
            "averaged_from": [str(p) for p in selected],
        },
        out_path,
    )
    print(f"Saved averaged checkpoint to {out_path}")


if __name__ == "__main__":
    main()
