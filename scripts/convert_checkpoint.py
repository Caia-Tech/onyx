#!/usr/bin/env python3
"""
Checkpoint conversion utility for resizing Onyx architecture dimensions.

This keeps overlapping tensor regions and preserves random init for newly
introduced dimensions. When `init_strategy="copy_last"`, newly added layers
copy from the last source layer before overlap copy is applied.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from onyx.model import Onyx, OnyxConfig


_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


def _flatten_cfg(cfg_json: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in cfg_json.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def _load_arch_config(path: str) -> OnyxConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    flat = _flatten_cfg(raw.get("architecture", raw))
    if "rope_theta" in flat and "rope_base" not in flat:
        flat["rope_base"] = flat.pop("rope_theta")
    valid = {f.name for f in dataclasses.fields(OnyxConfig)}
    filtered = {k: v for k, v in flat.items() if k in valid}
    return OnyxConfig(**filtered)


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
        return ckpt_obj["model_state_dict"]
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format")


def _copy_overlap(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    out = dst.clone()
    src = src.to(dtype=out.dtype)
    common = tuple(min(a, b) for a, b in zip(out.shape, src.shape))
    if len(common) == 0:
        return out
    slices = tuple(slice(0, n) for n in common)
    out[slices] = src[slices]
    return out


def _source_key_for_target(
    key: str,
    old_state: Dict[str, torch.Tensor],
    old_layers: int,
    init_strategy: str,
) -> Optional[str]:
    if key in old_state:
        return key

    m = _LAYER_RE.match(key)
    if m is None:
        return None
    layer_idx = int(m.group(1))
    suffix = m.group(2)

    if init_strategy == "copy_last" and old_layers > 0 and layer_idx >= old_layers:
        src = f"layers.{old_layers - 1}.{suffix}"
        if src in old_state:
            return src
    return None


def convert_checkpoint(
    *,
    input_ckpt: str,
    output_ckpt: str,
    input_config: str,
    target_config: str,
    init_strategy: str = "random",
) -> None:
    strategy = str(init_strategy).lower()
    if strategy not in {"random", "copy_last"}:
        raise ValueError(f"init_strategy must be 'random' or 'copy_last', got {init_strategy}")

    old_cfg = _load_arch_config(input_config)
    new_cfg = _load_arch_config(target_config)

    ckpt = torch.load(input_ckpt, map_location="cpu", weights_only=False)
    old_state = _extract_state_dict(ckpt)

    new_model = Onyx(new_cfg)
    new_state = new_model.state_dict()

    old_layers = int(old_cfg.n_layers)
    converted: Dict[str, torch.Tensor] = {}

    for key, dst in new_state.items():
        src_key = _source_key_for_target(key, old_state, old_layers, strategy)
        if src_key is None:
            converted[key] = dst
            continue
        converted[key] = _copy_overlap(dst, old_state[src_key])

    Path(output_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": converted}, output_ckpt)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert/resize an Onyx checkpoint to a target config.")
    p.add_argument("--input_ckpt", required=True)
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--input_config", required=True)
    p.add_argument("--target_config", required=True)
    p.add_argument("--init_strategy", choices=["random", "copy_last"], default="random")
    args = p.parse_args()

    convert_checkpoint(
        input_ckpt=args.input_ckpt,
        output_ckpt=args.output_ckpt,
        input_config=args.input_config,
        target_config=args.target_config,
        init_strategy=args.init_strategy,
    )


if __name__ == "__main__":
    main()
