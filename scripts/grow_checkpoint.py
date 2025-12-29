#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, Optional, Tuple

import torch

try:
    from _bootstrap import add_repo_root
except ImportError:
    from scripts._bootstrap import add_repo_root

add_repo_root()

from onyx.model import OnyxConfig, Onyx


def _load_config(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(path, "r") as f:
        cfg = json.load(f)
    arch = cfg.get("architecture", cfg)
    if not isinstance(arch, dict):
        raise ValueError("Model config must be a dict or contain an 'architecture' dict.")
    return cfg, dict(arch)


def _write_config(template_cfg: Dict[str, Any], arch: Dict[str, Any], path: str) -> None:
    if "architecture" in template_cfg:
        out_cfg = dict(template_cfg)
        out_cfg["architecture"] = arch
    else:
        out_cfg = arch
    with open(path, "w") as f:
        json.dump(out_cfg, f, indent=2)
        f.write("\n")


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


def _build_model_config(arch: Dict[str, Any]) -> OnyxConfig:
    d_model = int(arch.get("d_model", 384))
    n_heads = int(arch.get("n_heads", 6))
    head_dim = arch.get("head_dim")
    if head_dim is None:
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads or specify head_dim in config.")
        head_dim = d_model // n_heads
    head_dim = int(head_dim)
    max_seq_len = int(arch.get("max_seq_len", 4096))
    return OnyxConfig(
        d_model=d_model,
        n_layers=int(arch.get("n_layers", 6)),
        n_heads=n_heads,
        n_kv_heads=int(arch.get("n_kv_heads", 2)),
        head_dim=head_dim,
        d_ff=int(arch.get("d_ff", 4096)),
        vocab_size=int(arch.get("vocab_size", 128258)),
        max_seq_len=max_seq_len,
        train_seq_len=max_seq_len,
        rope_base=float(arch.get("rope_theta", arch.get("rope_base", 500000.0))),
        norm_eps=float(arch.get("norm_eps", 1e-5)),
        tie_embeddings=bool(arch.get("tie_embeddings", True)),
    )


def _copy_matching(old_sd: Dict[str, torch.Tensor], new_sd: Dict[str, torch.Tensor]) -> int:
    copied = 0
    for k, v in old_sd.items():
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v.clone()
            copied += 1
    return copied


def _copy_layer(
    old_sd: Dict[str, torch.Tensor],
    new_sd: Dict[str, torch.Tensor],
    src_idx: int,
    dst_idx: int,
) -> int:
    copied = 0
    src_prefix = f"layers.{src_idx}."
    dst_prefix = f"layers.{dst_idx}."
    for k in list(new_sd.keys()):
        if not k.startswith(dst_prefix):
            continue
        src_k = src_prefix + k[len(dst_prefix):]
        if src_k in old_sd and old_sd[src_k].shape == new_sd[k].shape:
            new_sd[k] = old_sd[src_k].clone()
            copied += 1
    return copied


def _zero_layer(
    new_sd: Dict[str, torch.Tensor],
    param_keys: set[str],
    dst_idx: int,
) -> int:
    zeroed = 0
    dst_prefix = f"layers.{dst_idx}."
    keep_norm_suffixes = ("norm1.weight", "norm2.weight")
    for k in list(new_sd.keys()):
        if k in param_keys and k.startswith(dst_prefix):
            if k.endswith(keep_norm_suffixes):
                continue
            new_sd[k].zero_()
            zeroed += 1
    return zeroed


def main() -> None:
    p = argparse.ArgumentParser(description="Grow an Onyx checkpoint by adding layers.")
    p.add_argument("--input_ckpt", type=str, required=True)
    p.add_argument("--output_ckpt", type=str, required=True)
    p.add_argument("--model_config", type=str, required=True)
    p.add_argument("--output_config", type=str, required=True)
    p.add_argument("--add_layers", type=int, default=1)
    p.add_argument("--init_strategy", choices=["copy_last", "zero", "random"], default="copy_last")
    args = p.parse_args()

    if args.add_layers <= 0:
        raise ValueError("--add_layers must be >= 1")

    ckpt = torch.load(args.input_ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        old_sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        old_sd = ckpt
    else:
        raise ValueError("Checkpoint did not contain a state_dict-like mapping.")

    cfg_json, arch = _load_config(args.model_config)
    inferred_layers = _infer_layers_from_state(old_sd)
    cfg_layers = int(arch.get("n_layers", inferred_layers or 0))
    if inferred_layers is not None and cfg_layers != inferred_layers:
        print(f"[WARN] Config n_layers={cfg_layers} but checkpoint has {inferred_layers} layers; using checkpoint.")
        cfg_layers = inferred_layers

    if cfg_layers <= 0:
        raise ValueError("Could not determine existing layer count.")

    new_layers = cfg_layers + int(args.add_layers)
    new_arch = dict(arch)
    new_arch["n_layers"] = new_layers

    _write_config(cfg_json, new_arch, args.output_config)
    new_cfg = _build_model_config(new_arch)
    new_model = Onyx(new_cfg)

    new_sd = new_model.state_dict()
    copied = _copy_matching(old_sd, new_sd)
    print(f"Copied {copied} tensors from source checkpoint.")

    if args.init_strategy == "copy_last":
        total_copied = 0
        for idx in range(cfg_layers, new_layers):
            total_copied += _copy_layer(old_sd, new_sd, cfg_layers - 1, idx)
        print(f"Initialized {new_layers - cfg_layers} new layer(s) from last layer ({total_copied} tensors).")
    elif args.init_strategy == "zero":
        param_keys = set(dict(new_model.named_parameters()).keys())
        total_zeroed = 0
        for idx in range(cfg_layers, new_layers):
            total_zeroed += _zero_layer(new_sd, param_keys, idx)
        print(f"Zero-initialized {new_layers - cfg_layers} new layer(s) ({total_zeroed} tensors).")
    else:
        print(f"Using random init for {new_layers - cfg_layers} new layer(s).")

    out = {
        "model_state_dict": new_sd,
        "source_checkpoint": args.input_ckpt,
        "notes": f"grown layers {cfg_layers}->{new_layers}, init_strategy={args.init_strategy}",
        "config_path": args.output_config,
    }
    torch.save(out, args.output_ckpt)
    print(f"Saved grown checkpoint: {args.output_ckpt}")


if __name__ == "__main__":
    main()
