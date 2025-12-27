#!/usr/bin/env python3
import argparse
import dataclasses
import json
from typing import Any, Dict, Optional, Tuple

import torch

from onyx_model import Onyx, OnyxConfig


def _load_config(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(path, "r") as f:
        cfg = json.load(f)
    arch = cfg.get("architecture", cfg)
    if not isinstance(arch, dict):
        raise ValueError("Model config must be a dict or contain an 'architecture' dict.")
    return cfg, dict(arch)


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Checkpoint did not contain a state_dict-like mapping.")


def _infer_layers_from_state(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    indices = set()
    for k in sd.keys():
        if not k.startswith("layers."):
            continue
        parts = k.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            indices.add(int(parts[1]))
    if not indices:
        return None
    return max(indices) + 1


def _normalize_arch(arch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(arch)
    if "rope_base" not in out and "rope_theta" in out:
        out["rope_base"] = out["rope_theta"]
    if "train_seq_len" not in out and "max_seq_len" in out:
        out["train_seq_len"] = out["max_seq_len"]
    return out


def _build_model_config(arch: Dict[str, Any]) -> OnyxConfig:
    valid = {f.name for f in dataclasses.fields(OnyxConfig)}
    normalized = _normalize_arch(arch)
    cfg_kwargs = {k: normalized[k] for k in normalized.keys() if k in valid}

    if "head_dim" not in cfg_kwargs:
        d_model = cfg_kwargs.get("d_model")
        n_heads = cfg_kwargs.get("n_heads")
        if d_model is not None and n_heads is not None:
            if int(d_model) % int(n_heads) != 0:
                raise ValueError("d_model must be divisible by n_heads or specify head_dim in config.")
            cfg_kwargs["head_dim"] = int(d_model) // int(n_heads)

    return OnyxConfig(**cfg_kwargs)


def _infer_head_dim(total: int, n_heads: int) -> Optional[int]:
    if n_heads <= 0:
        return None
    if total % n_heads != 0:
        return None
    return total // n_heads


def _copy_overlap(old_t: torch.Tensor, new_t: torch.Tensor) -> torch.Tensor:
    if old_t.shape == new_t.shape:
        return old_t.clone()
    if old_t.ndim != new_t.ndim:
        return new_t
    if old_t.ndim == 0:
        return old_t.clone()
    out = new_t.clone()
    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_t.shape, new_t.shape))
    out[slices] = old_t[slices].to(dtype=out.dtype)
    return out


def _remap_qkv_weight(
    old_t: torch.Tensor,
    new_t: torch.Tensor,
    old_heads: int,
    new_heads: int,
) -> torch.Tensor:
    if old_t.ndim != 2 or new_t.ndim != 2:
        return _copy_overlap(old_t, new_t)
    old_out, old_in = old_t.shape
    new_out, new_in = new_t.shape
    old_head_dim = _infer_head_dim(old_out, old_heads)
    new_head_dim = _infer_head_dim(new_out, new_heads)
    if old_head_dim is None or new_head_dim is None:
        return _copy_overlap(old_t, new_t)
    old_view = old_t.view(old_heads, old_head_dim, old_in)
    new_view = new_t.view(new_heads, new_head_dim, new_in).clone()
    h = min(old_heads, new_heads)
    hd = min(old_head_dim, new_head_dim)
    d = min(old_in, new_in)
    new_view[:h, :hd, :d] = old_view[:h, :hd, :d].to(dtype=new_view.dtype)
    return new_view.reshape(new_t.shape)


def _remap_o_proj_weight(
    old_t: torch.Tensor,
    new_t: torch.Tensor,
    old_heads: int,
    new_heads: int,
) -> torch.Tensor:
    if old_t.ndim != 2 or new_t.ndim != 2:
        return _copy_overlap(old_t, new_t)
    old_out, old_in = old_t.shape
    new_out, new_in = new_t.shape
    old_head_dim = _infer_head_dim(old_in, old_heads)
    new_head_dim = _infer_head_dim(new_in, new_heads)
    if old_head_dim is None or new_head_dim is None:
        return _copy_overlap(old_t, new_t)
    old_view = old_t.view(old_out, old_heads, old_head_dim)
    new_view = new_t.view(new_out, new_heads, new_head_dim).clone()
    d = min(old_out, new_out)
    h = min(old_heads, new_heads)
    hd = min(old_head_dim, new_head_dim)
    new_view[:d, :h, :hd] = old_view[:d, :h, :hd].to(dtype=new_view.dtype)
    return new_view.reshape(new_t.shape)


def _remap_value_gen_weight(
    old_t: torch.Tensor,
    new_t: torch.Tensor,
    old_heads: int,
    new_heads: int,
) -> torch.Tensor:
    if old_t.ndim != 2 or new_t.ndim != 2:
        return _copy_overlap(old_t, new_t)
    old_out, old_in = old_t.shape
    new_out, new_in = new_t.shape
    old_head_dim = _infer_head_dim(old_out, old_heads)
    new_head_dim = _infer_head_dim(new_out, new_heads)
    if old_head_dim is None or new_head_dim is None:
        return _copy_overlap(old_t, new_t)
    if old_out != old_in or new_out != new_in:
        return _copy_overlap(old_t, new_t)
    old_view = old_t.view(old_heads, old_head_dim, old_heads, old_head_dim)
    new_view = new_t.view(new_heads, new_head_dim, new_heads, new_head_dim).clone()
    h = min(old_heads, new_heads)
    hd = min(old_head_dim, new_head_dim)
    new_view[:h, :hd, :h, :hd] = old_view[:h, :hd, :h, :hd].to(dtype=new_view.dtype)
    return new_view.reshape(new_t.shape)


def _remap_tensor(
    name: str,
    old_t: torch.Tensor,
    new_t: torch.Tensor,
    old_cfg: OnyxConfig,
    new_cfg: OnyxConfig,
) -> torch.Tensor:
    if name.endswith(".q_proj.weight"):
        return _remap_qkv_weight(old_t, new_t, int(old_cfg.n_heads), int(new_cfg.n_heads))
    if name.endswith(".k_proj.weight") or name.endswith(".v_proj.weight"):
        return _remap_qkv_weight(old_t, new_t, int(old_cfg.n_kv_heads), int(new_cfg.n_kv_heads))
    if name.endswith(".o_proj.weight"):
        return _remap_o_proj_weight(old_t, new_t, int(old_cfg.n_heads), int(new_cfg.n_heads))
    if name.endswith(".memory.M_init"):
        return _remap_qkv_weight(old_t, new_t, int(old_cfg.n_kv_heads), int(new_cfg.n_kv_heads))
    if ".value_gen." in name and name.endswith(".weight"):
        return _remap_value_gen_weight(old_t, new_t, int(old_cfg.n_kv_heads), int(new_cfg.n_kv_heads))
    return _copy_overlap(old_t, new_t)


def _apply_zero_layer(new_sd: Dict[str, torch.Tensor], dst_idx: int) -> int:
    zeroed = 0
    dst_prefix = f"layers.{dst_idx}."
    keep_norm_suffixes = ("norm1.weight", "norm2.weight")
    for k, v in new_sd.items():
        if not k.startswith(dst_prefix):
            continue
        if k.endswith(keep_norm_suffixes):
            continue
        if not torch.is_tensor(v):
            continue
        v.zero_()
        zeroed += 1
    return zeroed


def _apply_copy_layer(
    old_sd: Dict[str, torch.Tensor],
    new_sd: Dict[str, torch.Tensor],
    old_cfg: OnyxConfig,
    new_cfg: OnyxConfig,
    src_idx: int,
    dst_idx: int,
) -> int:
    copied = 0
    src_prefix = f"layers.{src_idx}."
    dst_prefix = f"layers.{dst_idx}."
    for k, v in new_sd.items():
        if not k.startswith(dst_prefix):
            continue
        src_k = src_prefix + k[len(dst_prefix):]
        if src_k not in old_sd:
            continue
        mapped = _remap_tensor(k, old_sd[src_k], v, old_cfg, new_cfg)
        new_sd[k] = mapped
        copied += 1
    return copied


def convert_checkpoint(
    input_ckpt: str,
    output_ckpt: str,
    input_config: str,
    target_config: str,
    init_strategy: str = "copy_last",
) -> Dict[str, Any]:
    if init_strategy not in ("copy_last", "zero", "random"):
        raise ValueError("init_strategy must be one of: copy_last, zero, random")

    ckpt = torch.load(input_ckpt, map_location="cpu", weights_only=False)
    old_sd = _extract_state_dict(ckpt)

    _, old_arch = _load_config(input_config)
    _, new_arch = _load_config(target_config)

    inferred_layers = _infer_layers_from_state(old_sd)
    if inferred_layers is not None:
        cfg_layers = int(old_arch.get("n_layers", inferred_layers))
        if cfg_layers != inferred_layers:
            print(f"[WARN] Config n_layers={cfg_layers} but checkpoint has {inferred_layers}; using checkpoint.")
        old_arch["n_layers"] = inferred_layers

    old_cfg = _build_model_config(old_arch)
    new_cfg = _build_model_config(new_arch)

    new_model = Onyx(new_cfg)
    new_sd = new_model.state_dict()

    exact = 0
    remapped = 0
    missing = 0

    for k, v in new_sd.items():
        if k not in old_sd:
            missing += 1
            continue
        old_t = old_sd[k]
        if torch.is_tensor(old_t) and old_t.shape == v.shape:
            new_sd[k] = old_t.clone()
            exact += 1
        elif torch.is_tensor(old_t):
            new_sd[k] = _remap_tensor(k, old_t, v, old_cfg, new_cfg)
            remapped += 1
        else:
            missing += 1

    old_layers = int(old_cfg.n_layers)
    new_layers = int(new_cfg.n_layers)
    if new_layers > old_layers:
        if init_strategy == "copy_last":
            total = 0
            for idx in range(old_layers, new_layers):
                total += _apply_copy_layer(old_sd, new_sd, old_cfg, new_cfg, old_layers - 1, idx)
            print(f"Initialized {new_layers - old_layers} layer(s) from last layer ({total} tensors).")
        elif init_strategy == "zero":
            total = 0
            for idx in range(old_layers, new_layers):
                total += _apply_zero_layer(new_sd, idx)
            print(f"Zero-initialized {new_layers - old_layers} layer(s) ({total} tensors).")
        else:
            print(f"Using random init for {new_layers - old_layers} layer(s).")

    if bool(getattr(new_cfg, "tie_embeddings", True)) and "embed.weight" in new_sd:
        new_sd["lm_head.weight"] = new_sd["embed.weight"].clone()

    try:
        new_model.load_state_dict(new_sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"Converted state_dict failed strict load: {e}") from e

    out = {
        "model_state_dict": new_sd,
        "source_checkpoint": input_ckpt,
        "notes": f"converted init_strategy={init_strategy}",
        "config": {"model_config_path": target_config},
        "conversion_meta": {
            "exact_copied": exact,
            "remapped": remapped,
            "missing_in_old": missing,
        },
    }
    torch.save(out, output_ckpt)
    print(f"Saved converted checkpoint: {output_ckpt}")
    return out["conversion_meta"]


def main() -> None:
    p = argparse.ArgumentParser(description="Convert an Onyx checkpoint to a new architecture.")
    p.add_argument("--input_ckpt", type=str, required=True)
    p.add_argument("--output_ckpt", type=str, required=True)
    p.add_argument("--input_config", type=str, required=True)
    p.add_argument("--target_config", type=str, required=True)
    p.add_argument("--init_strategy", choices=["copy_last", "zero", "random"], default="copy_last")
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
