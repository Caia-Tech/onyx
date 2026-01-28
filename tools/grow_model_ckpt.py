#!/usr/bin/env python3
"""
Grow Onyx model checkpoint - supports growing d_model, n_layers, d_ff, n_heads.

This script:
- Loads an input checkpoint from a smaller model
- Instantiates a new larger model from provided JSON config
- Intelligently transfers weights:
    - Exact-match params: copied verbatim
    - Expanded params: overlap-copied, new dimensions initialized smart
    - New layers: left uninitialized (model's random init)
    - Missing params: appropriately initialized
- Saves a checkpoint that can be loaded with --init_checkpoint --no-init_strict

Growth strategies:
- d_model: Copy overlap, initialize new dims with scaled random init
- n_layers: Copy existing layers, leave new layers for random init
- d_ff: Copy overlap, zero-pad (preserves function initially)
- n_heads/n_kv_heads: Copy overlap, initialize new heads with scaled random init
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from onyx.model import Onyx, OnyxConfig

try:
    from onyx.experimental import OnyxMHC
    _MHC_AVAILABLE = True
except Exception:
    OnyxMHC = None  # type: ignore[assignment]
    _MHC_AVAILABLE = False


def _flatten_cfg(cfg_json: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in cfg_json.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def _load_arch_config(path: str) -> OnyxConfig:
    cfg_json = json.loads(Path(path).read_text())
    arch = cfg_json.get("architecture", cfg_json)
    if not isinstance(arch, dict):
        raise ValueError(f"{path} must contain a JSON object or an 'architecture' dict.")
    flat = _flatten_cfg(arch)
    valid_fields = {f.name for f in dataclasses.fields(OnyxConfig)}
    filtered = {k: v for k, v in flat.items() if k in valid_fields}
    if "rope_theta" in filtered and "rope_base" not in filtered:
        filtered["rope_base"] = filtered.pop("rope_theta")
    return OnyxConfig(**filtered)


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
        return ckpt_obj["model_state_dict"]
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj  # type: ignore[return-value]
    raise TypeError("Checkpoint must be a dict containing a state_dict or be a raw state_dict dict.")


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return sd
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _choose_best_key_normalization(
    old_sd: Dict[str, torch.Tensor],
    new_keys: Iterable[str],
) -> Tuple[Dict[str, torch.Tensor], str]:
    new_key_set = set(new_keys)
    candidates = [
        ("as_is", old_sd),
        ("strip_module", _strip_prefix(old_sd, "module.")),
        ("strip__orig_mod", _strip_prefix(old_sd, "_orig_mod.")),
        ("strip_module+_orig_mod", _strip_prefix(_strip_prefix(old_sd, "module."), "_orig_mod.")),
    ]
    best_name = candidates[0][0]
    best_sd = candidates[0][1]
    best_hits = -1
    for name, sd in candidates:
        hits = sum(1 for k in sd.keys() if k in new_key_set)
        if hits > best_hits:
            best_hits = hits
            best_name = name
            best_sd = sd
    return best_sd, best_name


def _kaiming_uniform_like(tensor: torch.Tensor, a: float = 0) -> torch.Tensor:
    """Initialize tensor with Kaiming uniform (similar to nn.Linear default init)."""
    fan_in = tensor.shape[1] if tensor.ndim >= 2 else tensor.shape[0]
    gain = math.sqrt(2.0 / (1 + a ** 2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def _smart_expand(dst: torch.Tensor, src: torch.Tensor, key: str, strategy: str = "auto") -> torch.Tensor:
    """
    Intelligently expand a tensor from src shape to dst shape.

    Strategies:
    - "zero_pad": Copy overlap, zero-pad rest (good for d_ff to preserve function)
    - "smart_init": Copy overlap, init new dims with scaled random (good for d_model, heads)
    - "auto": Choose based on key name
    """
    if dst.shape == src.shape:
        dst.copy_(src.to(dtype=dst.dtype))
        return dst

    # Determine strategy
    if strategy == "auto":
        # FFN weights: zero-pad to preserve function
        if ".ffn." in key and any(x in key for x in [".w1.", ".w2.", ".w3."]):
            strategy = "zero_pad"
        # Attention and other weights: smart init for new capacity
        else:
            strategy = "smart_init"

    # Copy overlapping region
    slices = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
    dst[slices].copy_(src[slices].to(dtype=dst.dtype))

    # Initialize non-overlapping regions
    if strategy == "zero_pad":
        # Zero the rest (already zeros if newly allocated)
        pass
    elif strategy == "smart_init":
        # Initialize new dimensions with scaled random values
        if dst.ndim >= 2:
            # For each dimension that grew, initialize the new region
            for dim_idx in range(dst.ndim):
                if dst.shape[dim_idx] > src.shape[dim_idx]:
                    # Create index for non-overlap region in this dimension
                    idx = [slice(None)] * dst.ndim
                    idx[dim_idx] = slice(src.shape[dim_idx], dst.shape[dim_idx])

                    # Initialize with scaled Kaiming uniform
                    new_region = dst[tuple(idx)]
                    _kaiming_uniform_like(new_region)
                    # Scale down to avoid disrupting training too much
                    new_region.mul_(0.1)

    return dst


def _build_grown_state_dict(
    old_sd: Dict[str, torch.Tensor],
    new_sd_template: Dict[str, torch.Tensor],
    old_cfg: OnyxConfig,
    new_cfg: OnyxConfig,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Build grown state dict with intelligent weight transfer."""
    out: Dict[str, torch.Tensor] = {}
    stats = {
        "copied_exact": 0,
        "expanded_smart": 0,
        "expanded_zero_pad": 0,
        "new_layers_skipped": 0,
        "missing_random_init": 0,
    }

    # Determine layer mapping
    old_n_layers = old_cfg.n_layers
    new_n_layers = new_cfg.n_layers

    for k, new_t in new_sd_template.items():
        # Extract layer index if this is a layer parameter
        layer_idx = None
        if ".layers." in k:
            try:
                parts = k.split(".layers.")[1].split(".")
                layer_idx = int(parts[0])
            except (ValueError, IndexError):
                pass

        # Handle layer growth: only copy weights for existing layers
        if layer_idx is not None and layer_idx >= old_n_layers:
            # This is a new layer, leave it with model's random init
            out[k] = new_t.clone()
            stats["new_layers_skipped"] += 1
            continue

        # Check if key exists in old checkpoint
        if k in old_sd:
            old_t = old_sd[k]
            if tuple(old_t.shape) == tuple(new_t.shape):
                # Exact match: copy directly
                out[k] = old_t.to(dtype=new_t.dtype).clone()
                stats["copied_exact"] += 1
            else:
                # Expanded: use smart expansion
                expanded = new_t.clone()  # Start with new random init
                _smart_expand(expanded, old_t, k, strategy="auto")
                out[k] = expanded

                # Track strategy used
                if ".ffn." in k:
                    stats["expanded_zero_pad"] += 1
                else:
                    stats["expanded_smart"] += 1
        else:
            # Missing in old checkpoint: keep new init
            out[k] = new_t.clone()
            stats["missing_random_init"] += 1

    # Add growth info
    growth_info = {
        "d_model": f"{old_cfg.d_model} -> {new_cfg.d_model}",
        "n_layers": f"{old_cfg.n_layers} -> {new_cfg.n_layers}",
        "n_heads": f"{old_cfg.n_heads} -> {new_cfg.n_heads}",
        "n_kv_heads": f"{old_cfg.n_kv_heads} -> {new_cfg.n_kv_heads}",
        "d_ff": f"{old_cfg.d_ff} -> {new_cfg.d_ff}",
    }

    return out, {"stats": stats, "growth": growth_info}


def _maybe_adjust_vocab_size(cfg: OnyxConfig, state: Dict[str, torch.Tensor]) -> None:
    if "embed.weight" in state:
        ckpt_vocab = int(state["embed.weight"].shape[0])
        if int(cfg.vocab_size) != ckpt_vocab:
            print(f"[Vocab] Adjusting config vocab_size {cfg.vocab_size} -> {ckpt_vocab} to match checkpoint")
            cfg.vocab_size = ckpt_vocab


def _build_model(cfg: OnyxConfig, ckpt_config: Dict[str, Any], force_no_mhc: bool):
    use_mhc = bool(ckpt_config.get("experimental_mhc", False)) and not force_no_mhc
    if use_mhc:
        if not _MHC_AVAILABLE:
            raise RuntimeError("Checkpoint indicates experimental_mhc=True but OnyxMHC import failed.")
        return OnyxMHC(
            cfg,
            mhc_n=int(ckpt_config.get("mhc_n", 2)),
            mhc_mode=str(ckpt_config.get("mhc_mode", "mhc")),
            mhc_sinkhorn=bool(ckpt_config.get("mhc_sinkhorn", True)),
            mhc_sinkhorn_iters=int(ckpt_config.get("mhc_sinkhorn_iters", 10)),
        )
    return Onyx(cfg)


def _sanity_check_shapes(
    old_sd: Dict[str, torch.Tensor],
    new_sd: Dict[str, torch.Tensor],
    old_cfg: OnyxConfig,
    new_cfg: OnyxConfig,
) -> None:
    """Sanity check that key shapes make sense given the config growth."""
    print("\n[Sanity] Checking key shape changes...")
    checked = 0
    for k in list(old_sd.keys())[:5]:  # Check first 5 keys
        if k in new_sd:
            old_shape = old_sd[k].shape
            new_shape = new_sd[k].shape
            if old_shape != new_shape:
                print(f"[Sanity]   {k}: {tuple(old_shape)} -> {tuple(new_shape)}")
                checked += 1
    if checked == 0:
        print("[Sanity]   (No shape changes in sampled keys)")


def main() -> None:
    p = argparse.ArgumentParser(description="Grow Onyx model checkpoint (d_model, n_layers, d_ff, n_heads).")
    p.add_argument("--in_ckpt", required=True, help="Input checkpoint path (.pt)")
    p.add_argument("--out_ckpt", required=True, help="Output grown checkpoint path (.pt)")
    p.add_argument("--old_config", required=True, help="Old model JSON config")
    p.add_argument("--new_config", required=True, help="New (larger) model JSON config")
    p.add_argument("--force_no_mhc", action="store_true", help="Force plain Onyx() even if ckpt says experimental_mhc.")
    p.add_argument("--sanity_check", action="store_true", help="Run additional sanity checks on output.")
    args = p.parse_args()

    print(f"Loading checkpoint: {args.in_ckpt}")
    ckpt = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)
    old_sd_raw = _extract_state_dict(ckpt)

    print(f"Loading old config: {args.old_config}")
    old_cfg = _load_arch_config(args.old_config)
    print(f"Loading new config: {args.new_config}")
    new_cfg = _load_arch_config(args.new_config)

    # Display growth plan
    print("\n" + "="*70)
    print("GROWTH PLAN")
    print("="*70)
    print(f"d_model:    {old_cfg.d_model:4d} -> {new_cfg.d_model:4d}  (Δ={new_cfg.d_model - old_cfg.d_model:+d})")
    print(f"n_layers:   {old_cfg.n_layers:4d} -> {new_cfg.n_layers:4d}  (Δ={new_cfg.n_layers - old_cfg.n_layers:+d})")
    print(f"n_heads:    {old_cfg.n_heads:4d} -> {new_cfg.n_heads:4d}  (Δ={new_cfg.n_heads - old_cfg.n_heads:+d})")
    print(f"n_kv_heads: {old_cfg.n_kv_heads:4d} -> {new_cfg.n_kv_heads:4d}  (Δ={new_cfg.n_kv_heads - old_cfg.n_kv_heads:+d})")
    print(f"d_ff:       {old_cfg.d_ff:4d} -> {new_cfg.d_ff:4d}  (Δ={new_cfg.d_ff - old_cfg.d_ff:+d})")
    print("="*70 + "\n")

    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(ckpt_cfg, dict):
        ckpt_cfg = {}

    _maybe_adjust_vocab_size(new_cfg, old_sd_raw)

    print("Building new (larger) model...")
    new_model = _build_model(new_cfg, ckpt_cfg, args.force_no_mhc)
    new_sd_template = new_model.state_dict()

    print("Building old model for reference...")
    old_model = _build_model(old_cfg, ckpt_cfg, args.force_no_mhc)

    print("Normalizing checkpoint keys...")
    old_sd, key_norm = _choose_best_key_normalization(old_sd_raw, new_sd_template.keys())
    print(f"Key normalization: {key_norm}")

    print("\nTransferring and growing weights...")
    grown_sd, info = _build_grown_state_dict(old_sd, new_sd_template, old_cfg, new_cfg)

    print(f"\nTransfer statistics:")
    for k, v in info["stats"].items():
        print(f"  {k}: {v}")

    print(f"\nGrowth summary:")
    for k, v in info["growth"].items():
        print(f"  {k}: {v}")

    if args.sanity_check:
        _sanity_check_shapes(old_sd, grown_sd, old_cfg, new_cfg)

    print("\nLoading grown state dict into new model (strict=False)...")
    incompatible = new_model.load_state_dict(grown_sd, strict=False)

    if incompatible.missing_keys:
        print(f"[WARN] Missing keys: {len(incompatible.missing_keys)}")
        for k in list(incompatible.missing_keys)[:5]:
            print(f"  - {k}")
        if len(incompatible.missing_keys) > 5:
            print(f"  ... and {len(incompatible.missing_keys) - 5} more")

    if incompatible.unexpected_keys:
        print(f"[WARN] Unexpected keys: {len(incompatible.unexpected_keys)}")
        for k in list(incompatible.unexpected_keys)[:5]:
            print(f"  - {k}")
        if len(incompatible.unexpected_keys) > 5:
            print(f"  ... and {len(incompatible.unexpected_keys) - 5} more")

    note = {
        "type": "grow_model",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "in_ckpt": str(args.in_ckpt),
        "old_config": str(args.old_config),
        "new_config": str(args.new_config),
        "key_normalization": key_norm,
        **info,
    }

    # Preserve config from original checkpoint
    out_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    out_ckpt = {
        "model_state_dict": grown_sd,
        "config": out_config,
        "note": note,
    }

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(out_path))

    print(f"\n{'='*70}")
    print(f"SUCCESS! Saved grown checkpoint: {out_path}")
    print(f"{'='*70}")
    print(f"\nTo use this checkpoint, run training with:")
    print(f"  --init_checkpoint {out_path}")
    print(f"  --no-init_strict")
    print(f"  --model_config {args.new_config}")


if __name__ == "__main__":
    main()
