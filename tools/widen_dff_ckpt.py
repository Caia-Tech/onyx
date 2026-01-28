#!/usr/bin/env python3
"""
Widen Onyx d_ff while preserving function at step 0 by zero-padding expanded weights.

This script:
- Loads an input checkpoint (expects "model_state_dict" or a raw state_dict).
- Instantiates a new model from a provided JSON config (d_ff widened).
- Builds a new state_dict:
    - exact-shape params: copied verbatim
    - expanded params: overlap-copied into top-left, remainder zero
    - missing params: zero (float) or keep new init (non-float)
- Loads with strict=True and saves a weights-only checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch

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
        # Could already be a raw state_dict
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


def _copy_overlap(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    if dst.shape == src.shape:
        dst.copy_(src.to(dtype=dst.dtype))
        return dst
    slices = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
    dst[slices].copy_(src[slices].to(dtype=dst.dtype))
    return dst


def _build_widened_state_dict(
    old_sd: Dict[str, torch.Tensor],
    new_sd_template: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    out: Dict[str, torch.Tensor] = {}
    stats = {"copied_exact": 0, "copied_overlap": 0, "missing_zeroed": 0, "missing_kept_init": 0}

    for k, new_t in new_sd_template.items():
        if k in old_sd:
            old_t = old_sd[k]
            if tuple(old_t.shape) == tuple(new_t.shape):
                out[k] = old_t.to(dtype=new_t.dtype).clone()
                stats["copied_exact"] += 1
            else:
                widened = torch.zeros_like(new_t)
                _copy_overlap(widened, old_t)
                out[k] = widened
                stats["copied_overlap"] += 1
        else:
            if torch.is_floating_point(new_t):
                out[k] = torch.zeros_like(new_t)
                stats["missing_zeroed"] += 1
            else:
                out[k] = new_t.clone()
                stats["missing_kept_init"] += 1

    return out, stats


def _assert_zero_padded_expansion(
    *,
    key: str,
    old_t: torch.Tensor,
    new_t: torch.Tensor,
    widened_t: torch.Tensor,
) -> None:
    if tuple(old_t.shape) == tuple(new_t.shape):
        return
    if widened_t.shape != new_t.shape:
        raise RuntimeError(f"{key}: widened tensor shape mismatch {tuple(widened_t.shape)} != {tuple(new_t.shape)}")

    overlap = tuple(min(o, n) for o, n in zip(old_t.shape, new_t.shape))
    slices = tuple(slice(0, m) for m in overlap)
    if not torch.allclose(
        widened_t[slices].float().cpu(),
        old_t[slices].to(dtype=widened_t.dtype).float().cpu(),
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError(f"{key}: overlap copy mismatch")

    # Check padded region is exactly zero for common CMSFFN expanded matrices.
    # For other expanded tensors (if any), zero-padding is still the default behavior.
    if torch.is_floating_point(widened_t):
        if old_t.shape != new_t.shape:
            # Any element outside the overlap must be zero.
            mask = torch.ones_like(widened_t, dtype=torch.bool)
            mask[slices] = False
            if mask.any():
                if widened_t[mask].abs().max().item() != 0.0:
                    raise RuntimeError(f"{key}: non-zero values in padded region")


def _maybe_adjust_vocab_size(cfg: OnyxConfig, state: Dict[str, torch.Tensor]) -> None:
    if "embed.weight" in state:
        ckpt_vocab = int(state["embed.weight"].shape[0])
        if int(cfg.vocab_size) != ckpt_vocab:
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


def _sanity_check_prompt(
    *,
    prompt: str,
    old_model,
    new_model,
    tokenizer_path: str = "tokenizer",
    device: str = "cpu",
) -> None:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        print("[Sanity] transformers not installed; skipping prompt sanity check.")
        return

    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    old_model = old_model.to(device).eval()
    new_model = new_model.to(device).eval()

    with torch.inference_mode():
        out_old = old_model(input_ids, update_memories=False, inference_mode=True)
        out_new = new_model(input_ids, update_memories=False, inference_mode=True)

    logits_old = out_old["logits"][0, -1].float().cpu()
    logits_new = out_new["logits"][0, -1].float().cpu()

    cos = torch.nn.functional.cosine_similarity(logits_old, logits_new, dim=0).item()
    rel_l2 = (logits_old - logits_new).norm().item() / (logits_old.norm().item() + 1e-12)
    topk = 20
    top_old = set(torch.topk(logits_old, k=topk).indices.tolist())
    top_new = set(torch.topk(logits_new, k=topk).indices.tolist())
    overlap = len(top_old & top_new)

    print("[Sanity] Next-token logits:")
    print(f"[Sanity] cosine={cos:.6f}  rel_l2={rel_l2:.6f}  top{topk}_overlap={overlap}/{topk}")


def main() -> None:
    p = argparse.ArgumentParser(description="Widen Onyx d_ff checkpoint with zero-padded expansion.")
    p.add_argument("--in_ckpt", required=True, help="Input checkpoint path (.pt)")
    p.add_argument("--out_ckpt", required=True, help="Output widened checkpoint path (.pt)")
    p.add_argument("--old_config", required=True, help="Old model JSON config (d_ff=512)")
    p.add_argument("--new_config", required=True, help="New model JSON config (d_ff=1024)")
    p.add_argument("--force_no_mhc", action="store_true", help="Force plain Onyx() even if ckpt says experimental_mhc.")
    p.add_argument("--sanity_prompt", default="", help="If set, run a quick prompt sanity check on CPU.")
    args = p.parse_args()

    ckpt = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)
    old_sd_raw = _extract_state_dict(ckpt)

    new_cfg = _load_arch_config(args.new_config)
    old_cfg = _load_arch_config(args.old_config)

    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(ckpt_cfg, dict):
        ckpt_cfg = {}

    _maybe_adjust_vocab_size(new_cfg, old_sd_raw)

    new_model = _build_model(new_cfg, ckpt_cfg, args.force_no_mhc)
    new_sd_template = new_model.state_dict()

    old_sd, key_norm = _choose_best_key_normalization(old_sd_raw, new_sd_template.keys())
    widened_sd, stats = _build_widened_state_dict(old_sd, new_sd_template)

    # Safety: assert CMSFFN expansions are strict top-left copies with zero padding.
    for k, new_t in new_sd_template.items():
        if k not in old_sd:
            continue
        if ".ffn.level_ffns." in k and k.endswith((".w1.weight", ".w2.weight", ".w3.weight")):
            _assert_zero_padded_expansion(key=k, old_t=old_sd[k], new_t=new_t, widened_t=widened_sd[k])

    incompatible = new_model.load_state_dict(widened_sd, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict=True load had incompatible keys: {incompatible}")

    note = {
        "type": "widen_dff",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "in_ckpt": str(args.in_ckpt),
        "old_config": str(args.old_config),
        "new_config": str(args.new_config),
        "old_d_ff": int(old_cfg.d_ff),
        "new_d_ff": int(new_cfg.d_ff),
        "cms_num_levels": int(new_cfg.cms_num_levels),
        "per_level_ff_old": int(max(1, old_cfg.d_ff // max(1, old_cfg.cms_num_levels))),
        "per_level_ff_new": int(max(1, new_cfg.d_ff // max(1, new_cfg.cms_num_levels))),
        "key_normalization": key_norm,
        "stats": stats,
        "missing_policy": "zero_float_keep_init_nonfloat",
    }

    out_ckpt = {
        "model_state_dict": widened_sd,
        "config": ckpt.get("config", {}) if isinstance(ckpt, dict) else {},
        "note": note,
    }

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(out_path))
    print(f"Saved widened checkpoint: {out_path}")
    print(f"Key normalization: {key_norm}  Stats: {stats}")

    if args.sanity_prompt.strip():
        old_model = _build_model(old_cfg, ckpt_cfg, args.force_no_mhc)
        old_sd_norm, _ = _choose_best_key_normalization(old_sd_raw, old_model.state_dict().keys())
        incompatible_old = old_model.load_state_dict(old_sd_norm, strict=True)
        if incompatible_old.missing_keys or incompatible_old.unexpected_keys:
            raise RuntimeError(f"Old strict=True load had incompatible keys: {incompatible_old}")
        _sanity_check_prompt(prompt=args.sanity_prompt, old_model=old_model, new_model=new_model)


if __name__ == "__main__":
    main()
