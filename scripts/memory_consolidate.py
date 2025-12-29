#!/usr/bin/env python3
"""
Run OnyxMemoryConsolidator with a training-style dataset/dataloader.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from _bootstrap import add_repo_root
except ImportError:
    from scripts._bootstrap import add_repo_root

add_repo_root()

from onyx.memory_consolidator import OnyxMemoryConsolidator
from onyx.model import Onyx, OnyxConfig
from onyx.train import StreamingPackedDataset, collate_onyx


def _load_model_config(path: str, max_seq_len: int) -> OnyxConfig:
    cfg_json = json.loads(Path(path).read_text())
    arch = cfg_json.get("architecture", cfg_json)
    return OnyxConfig(
        d_model=arch.get("d_model", 384),
        n_layers=arch.get("n_layers", 6),
        n_heads=arch.get("n_heads", 6),
        n_kv_heads=arch.get("n_kv_heads", 2),
        d_ff=arch.get("d_ff", 4096),
        vocab_size=arch.get("vocab_size", 128258),
        max_seq_len=max_seq_len,
        train_seq_len=max_seq_len,
        rope_base=arch.get("rope_theta", arch.get("rope_base", 500000.0)),
        norm_eps=arch.get("norm_eps", 1e-5),
        use_flash_attention=torch.cuda.is_available(),
        gradient_checkpointing=torch.cuda.is_available(),
        memory_reg_weight=arch.get("memory_reg_weight", 0.0001),
    )


def _extract_ckpt_vocab(ckpt: Dict[str, Any]) -> int:
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
    if not isinstance(state, dict):
        return 0
    for k in ("embed.weight", "lm_head.weight"):
        if k in state and hasattr(state[k], "shape"):
            return int(state[k].shape[0])
    return 0


def _resize_vocab_tensor(state_dict: Dict[str, torch.Tensor], key: str, target: torch.Tensor) -> None:
    if key not in state_dict:
        return
    t = state_dict[key]
    if t.shape == target.shape:
        return
    if t.ndim != target.ndim or t.shape[1:] != target.shape[1:]:
        raise ValueError(f"Cannot resize {key}: ckpt {tuple(t.shape)} vs target {tuple(target.shape)}")
    new_t = torch.zeros_like(target)
    rows = min(t.shape[0], target.shape[0])
    new_t[:rows] = t[:rows]
    state_dict[key] = new_t
    print(f"Resized {key}: {tuple(t.shape)} -> {tuple(target.shape)}")


class LimitedDataloader:
    def __init__(self, dataloader: DataLoader, steps_per_epoch: int, max_batches: Optional[int]):
        self._dataloader = dataloader
        self._steps = steps_per_epoch
        self._max_batches = max_batches

    def __iter__(self):
        if self._max_batches is None:
            return iter(self._dataloader)
        return itertools.islice(self._dataloader, self._max_batches)

    def __len__(self):
        return self._steps


def _pick_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _estimate_steps(dataloader: DataLoader) -> int:
    steps = 0
    for _ in dataloader:
        steps += 1
    return max(1, steps)


def main() -> int:
    p = argparse.ArgumentParser(description="Consolidate Onyx memory parameters with EWC.")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path (training format).")
    p.add_argument("--model_config", required=True, help="Model config JSON path.")
    p.add_argument("--tokenizer", required=True, help="Tokenizer name or local path.")
    p.add_argument("--data_glob", required=True, help="Training data glob (jsonl).")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--no-pack", action="store_true", help="Disable sequence packing.")
    p.add_argument("--shuffle_buffer_docs", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)

    p.add_argument("--ewc_lambda", type=float, default=500.0)
    p.add_argument("--fisher_sample_size", type=int, default=0)
    p.add_argument("--importance_threshold", type=float, default=1e-5)
    p.add_argument("--init_fisher", action="store_true", help="Precompute Fisher/anchors before consolidation.")

    p.add_argument("--steps_per_epoch", type=int, default=0, help="Override steps per epoch.")
    p.add_argument("--max_batches", type=int, default=0, help="Limit batches per epoch.")

    p.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (default: auto).")
    p.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    p.add_argument("--memory_state", type=str, default=None, help="Optional memory state path.")
    p.add_argument("--save_model", type=str, default=None, help="Save consolidated model checkpoint.")
    p.add_argument("--save_distiller", type=str, default=None, help="Save consolidator state.")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _pick_device(args.device)
    dtype = _parse_dtype(args.dtype)
    print(f"Device: {device} | dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = _load_model_config(args.model_config, args.max_seq_len)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    tok_vs = len(tokenizer)
    ckpt_vs = _extract_ckpt_vocab(ckpt) if isinstance(ckpt, dict) else 0
    target_vs = max(int(config.vocab_size), tok_vs, ckpt_vs)
    if config.vocab_size != target_vs:
        print(f"Adjusting vocab_size: {config.vocab_size} -> {target_vs}")
        config.vocab_size = target_vs

    model = Onyx(config)
    if isinstance(state, dict):
        if hasattr(model, "embed"):
            _resize_vocab_tensor(state, "embed.weight", model.embed.weight)
        if hasattr(model, "lm_head"):
            _resize_vocab_tensor(state, "lm_head.weight", model.lm_head.weight)
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=dtype)

    dataset = StreamingPackedDataset(
        data_glob=args.data_glob,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        pack=not args.no_pack,
        seed=args.seed,
        drop_remainder=False,
        emit_cu_seqlens=(device.type == "cuda"),
        shuffle_buffer_docs=args.shuffle_buffer_docs,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_onyx,
    )

    max_batches = args.max_batches if args.max_batches > 0 else None
    if args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch
    elif max_batches is not None:
        steps_per_epoch = max_batches
    else:
        print("Estimating steps_per_epoch by one streaming pass...")
        steps_per_epoch = _estimate_steps(dataloader)
        print(f"Estimated steps_per_epoch={steps_per_epoch}")

    if max_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_batches)

    wrapped_loader = LimitedDataloader(dataloader, steps_per_epoch, max_batches)

    fisher_sample_size = args.fisher_sample_size if args.fisher_sample_size > 0 else None
    consolidator = OnyxMemoryConsolidator(
        model=model,
        dataloader=wrapped_loader,
        device=device,
        ewc_lambda=args.ewc_lambda,
        fisher_sample_size=fisher_sample_size,
        importance_threshold=args.importance_threshold,
    )

    memory_states = None
    if args.memory_state:
        print(f"Loading memory states: {args.memory_state}")
        memory_states = model.load_memory_states(args.memory_state, device=device, dtype=dtype)

    if args.init_fisher:
        print("Precomputing Fisher/anchors...")
        consolidator.fisher_diagonal = consolidator.estimate_importance(memory_states, verbose=True)
        consolidator.theta_star = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    metrics = consolidator.consolidate(
        memory_states=memory_states,
        epochs=args.epochs,
        lr=args.lr,
        gradient_clip=args.gradient_clip,
        log_interval=args.log_interval,
        validate_fn=None,
    )

    print("Final metrics:", metrics)

    if args.save_model:
        ckpt_out = {
            "model_state_dict": model.state_dict(),
            "config": dataclasses.asdict(config),
        }
        torch.save(ckpt_out, args.save_model)
        print(f"Saved consolidated model checkpoint: {args.save_model}")

    if args.save_distiller:
        consolidator.save_checkpoint(args.save_distiller)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
