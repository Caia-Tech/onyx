#!/usr/bin/env python3
"""
Onyx 11M Training Script (Improved)

Key fixes vs prior version:
- CLI flag for keep_last_n + milestone retention keep_every_steps
- Safe vocab alignment for tokenizer/resume/init_checkpoint
- Safe init_checkpoint load when vocab differs (resizes embed/lm_head)
- Iterable dataset is actually streaming (buffer shuffle, no list(all_docs))
- CUDA varlen cu_seqlens batching uses a collate_fn (ragged-safe)
- pin_memory only on CUDA
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator, List, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from onyx_model import (
    Onyx,
    OnyxConfig,
    M3Optimizer,
    get_param_groups,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

class CMSFrequencyManager:
    """
    Manages the multi-timescale updates for CMS layers.
    Paper Section 7.1, Eq 71.

    This masks gradients for CMS levels that are not scheduled to update at the
    current optimizer step.
    """

    def __init__(self, model: nn.Module, config: OnyxConfig):
        self.param_schedules: List[Dict[str, Any]] = []
        if config is None:
            return

        base = int(getattr(config, "cms_base_chunk", 1) or 1)
        mult = int(getattr(config, "cms_chunk_multiplier", 2) or 2)
        if base < 1:
            base = 1
        if mult < 1:
            mult = 1

        for name, param in model.named_parameters():
            if "ffn.level_ffns" not in name:
                continue
            parts = name.split(".")
            try:
                idx = parts.index("level_ffns") + 1
                level = int(parts[idx])
            except Exception:
                continue

            update_every = int(base * (mult ** level))
            update_every = max(1, update_every)
            self.param_schedules.append(
                {"param": param, "update_every": update_every, "name": name, "level": level}
            )

    def mask_gradients(self, global_step: int) -> None:
        """
        Call this before gradient clipping / optimizer.step().
        Zeroes out gradients for levels that shouldn't update this step.
        """
        step = int(global_step)
        for item in self.param_schedules:
            if step % item["update_every"] != 0:
                if item["param"].grad is not None:
                    item["param"].grad = None


@dataclass
class TrainingConfig:
    # === Data ===
    data_glob: str = "./data.jsonl"
    tokenizer_name: str = "NousResearch/Hermes-2-Pro-Llama-3-8B"

    # === Sequence ===
    max_seq_len: int = 256
    pack_sequences: bool = True
    drop_remainder: bool = False
    shuffle_buffer_docs: int = 2048  # buffer-shuffle docs without loading all into RAM

    # === Batch & Accumulation ===
    batch_size: int = 4
    tokens_per_step: int = 8192

    # === Training Duration ===
    num_epochs: int = 1
    train_tokens_target: Optional[int] = None
    max_steps: Optional[int] = None

    # === Learning Rate ===
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    memory_lr_scale: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.1

    # === Memory ===
    memory_reg_weight: float = 0.0001

    # === Optimizer ===
    use_m3_optimizer: bool = True
    m3_beta_slow: float = 0.99
    m3_slow_freq: int = 50
    m3_slow_weight: float = 0.1

    # === Precision ===
    use_amp: bool = True
    amp_dtype: str = "float16"
    amp_flag: Optional[bool] = None

    # === Compilation ===
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"

    # === Gradient ===
    gradient_clip: float = 1.0

    # === Benchmarking ===
    bench_steps: int = 0
    disable_saves_during_bench: bool = False

    # === Checkpointing ===
    save_dir: str = "./checkpoints"
    save_every_epoch: bool = True
    save_every_steps: int = 0
    keep_last_n: int = 5
    keep_every_steps: int = 0  # if >0, keep step checkpoints at multiples of this forever
    resume: Optional[str] = None
    init_checkpoint: Optional[str] = None  # load weights only
    model_config_path: Optional[str] = None

    # === Logging ===
    log_every: int = 10
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # === Debug ===
    dry_run: bool = False
    seed: int = 42


# =============================================================================
# Dataset
# =============================================================================

def _iter_jsonl_texts(filepath: str) -> Iterator[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue

            text = data.get("text") or data.get("content") or data.get("input") or ""
            if not text:
                system = data.get("system")
                user = data.get("user")
                assistant = data.get("assistant")
                if isinstance(user, str) and isinstance(assistant, str):
                    system = system.strip() if isinstance(system, str) else ""
                    user = user.strip()
                    assistant = assistant.strip()
                    if system:
                        text = f"System: {system}\n\nUser: {user}\nAssistant: {assistant}"
                    else:
                        text = f"User: {user}\nAssistant: {assistant}"

            if isinstance(text, str) and text:
                yield text


class StreamingPackedDataset(IterableDataset):
    """
    Streaming dataset that packs documents into fixed-length sequences.
    Uses buffer shuffle (bounded memory) rather than list(all_docs).
    """

    def __init__(
        self,
        data_glob: str,
        tokenizer,
        max_seq_len: int = 2048,
        pack: bool = True,
        seed: int = 42,
        drop_remainder: bool = False,
        emit_cu_seqlens: bool = True,
        shuffle_buffer_docs: int = 2048,
    ):
        self.data_glob = data_glob
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pack = pack
        self.seed = seed
        self.drop_remainder = drop_remainder
        self.emit_cu_seqlens = emit_cu_seqlens
        self.shuffle_buffer_docs = max(0, int(shuffle_buffer_docs))

        self.data_files = self._get_files()
        if not self.data_files:
            raise ValueError(f"No files found matching: {data_glob}")

        print(f"Found {len(self.data_files)} data file(s)")
        for f in self.data_files[:5]:
            print(f"  - {f}")

        self.eod_token_id = tokenizer.eos_token_id
        if self.eod_token_id is None:
            self.eod_token_id = tokenizer.pad_token_id or 0

    def _get_files(self) -> List[str]:
        from glob import glob
        g = self.data_glob
        if "*" not in g and "?" not in g:
            return [g] if os.path.exists(g) else []
        return sorted(glob(g))

    def _iter_documents(self) -> Iterator[str]:
        rng = random.Random(self.seed)

        files = list(self.data_files)
        rng.shuffle(files)  # cheap shuffle across files

        if self.shuffle_buffer_docs <= 1:
            for fp in files:
                yield from _iter_jsonl_texts(fp)
            return

        buf: List[str] = []
        for fp in files:
            for text in _iter_jsonl_texts(fp):
                buf.append(text)
                if len(buf) >= self.shuffle_buffer_docs:
                    rng.shuffle(buf)
                    for t in buf:
                        yield t
                    buf.clear()

        if buf:
            rng.shuffle(buf)
            for t in buf:
                yield t

    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _pack_sequences_with_boundaries(self) -> Iterator[Tuple[List[int], List[int], Optional[int]]]:
        """Pack documents into fixed-length sequences and track doc boundaries.

        The returned `boundaries` is a cu_seqlens-style list of start offsets for
        each packed document segment within the fixed-length chunk, always
        starting at 0 and ending at `max_seq_len`.
        """
        current_seq: List[int] = []
        current_boundaries: List[int] = [0]  # doc start offsets within current_seq

        def _dedup_consecutive(xs: List[int]) -> List[int]:
            if not xs:
                return []
            out = [xs[0]]
            for v in xs[1:]:
                if v != out[-1]:
                    out.append(v)
            return out

        for doc in self._iter_documents():
            toks = self._tokenize(doc)
            if not toks:
                continue

            toks.append(self.eod_token_id)
            doc_start_in_seq = len(current_seq)
            current_seq.extend(toks)

            # Track where the new document begins (if not already at 0)
            if doc_start_in_seq != 0:
                if current_boundaries[-1] != doc_start_in_seq:
                    current_boundaries.append(doc_start_in_seq)

            while len(current_seq) >= self.max_seq_len:
                seq = current_seq[: self.max_seq_len]

                b = [v for v in current_boundaries if v < self.max_seq_len]
                if not b or b[0] != 0:
                    b.insert(0, 0)
                if b[-1] != self.max_seq_len:
                    b.append(self.max_seq_len)
                b = _dedup_consecutive(b)

                yield seq, b, None

                # Shift leftovers and boundaries down by max_seq_len.
                current_seq = current_seq[self.max_seq_len :]
                shifted = [v - self.max_seq_len for v in current_boundaries if v >= self.max_seq_len]
                if not shifted or shifted[0] != 0:
                    shifted.insert(0, 0)  # doc continuation starts at 0 in the new chunk
                current_boundaries = _dedup_consecutive(shifted)

        if current_seq and not self.drop_remainder:
            pad_start = None
            if len(current_seq) < self.max_seq_len:
                pad_start = len(current_seq)
                current_seq.extend([self.eod_token_id] * (self.max_seq_len - len(current_seq)))

            b = [v for v in current_boundaries if v < self.max_seq_len]
            if not b or b[0] != 0:
                b.insert(0, 0)
            if b[-1] != self.max_seq_len:
                b.append(self.max_seq_len)
            b = _dedup_consecutive(b)

            yield current_seq[: self.max_seq_len], b, pad_start

    def _simple_sequences(self) -> Iterator[Tuple[List[int], Optional[int]]]:
        buf: List[int] = []
        for doc in self._iter_documents():
            toks = self._tokenize(doc)
            if not toks:
                continue
            toks.append(self.eod_token_id)
            buf.extend(toks)

            while len(buf) >= self.max_seq_len:
                yield buf[: self.max_seq_len], None
                buf = buf[self.max_seq_len :]

        if buf and not self.drop_remainder:
            pad_start = None
            if len(buf) < self.max_seq_len:
                pad_start = len(buf)
                buf.extend([self.eod_token_id] * (self.max_seq_len - len(buf)))
            yield buf[: self.max_seq_len], pad_start

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.pack:
            for seq, boundaries, pad_start in self._pack_sequences_with_boundaries():
                input_ids = torch.tensor(seq, dtype=torch.long)
                labels = input_ids.clone()
                if pad_start is not None and pad_start < labels.numel():
                    labels[pad_start:] = -100

                sample: Dict[str, Any] = {"input_ids": input_ids, "labels": labels}
                if self.emit_cu_seqlens:
                    sample["cu_seqlens"] = torch.tensor(boundaries, dtype=torch.int32)
                    sample["max_seqlen"] = self.max_seq_len
                yield sample
        else:
            for seq, pad_start in self._simple_sequences():
                input_ids = torch.tensor(seq, dtype=torch.long)
                labels = input_ids.clone()
                if pad_start is not None:
                    labels[pad_start:] = -100
                yield {"input_ids": input_ids, "labels": labels}


def collate_onyx(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Stack fixed tensors; keep cu_seqlens ragged-safe (list of tensors).
    out: Dict[str, Any] = {}
    out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
    out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)

    if "cu_seqlens" in batch[0]:
        out["cu_seqlens"] = [b["cu_seqlens"] for b in batch]  # ragged list
        out["max_seqlen"] = torch.tensor([b.get("max_seqlen", 0) for b in batch], dtype=torch.int32)
    return out


# =============================================================================
# LR
# =============================================================================

def get_lr(step: int, config: TrainingConfig, total_steps: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)

    decay_steps = max(1, total_steps - config.warmup_steps)
    progress = (step - config.warmup_steps) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    return config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        scale = pg.get("lr_scale", 1.0)
        pg["lr"] = lr * scale


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._interrupt_requested = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.resume_vocab_size: Optional[int] = None
        self.init_vocab_size: Optional[int] = None

    def _handle_signal(self, signum, frame):
        print(f"\n[SIGNAL] Received {signum}, will save and exit after current step...")
        self._interrupt_requested = True

    def _extract_vocab_size_from_ckpt(self, ckpt: Dict[str, Any]) -> Optional[int]:
        msd = ckpt.get("model_state_dict") or ckpt.get("model") or None
        if not isinstance(msd, dict):
            return None
        for k in ("embed.weight", "lm_head.weight"):
            if k in msd and hasattr(msd[k], "shape"):
                return int(msd[k].shape[0])
        return None

    def _resize_vocab_tensor(self, state_dict: Dict[str, torch.Tensor], key: str, target: torch.Tensor) -> None:
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

    def setup(self):
        cfg = self.config
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
            print(f"Using CUDA: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_type = "mps"
            print("Using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"
            print("Using CPU")
            cfg.use_amp = False

        if cfg.amp_dtype == "bfloat16":
            self.autocast_dtype = torch.bfloat16
        elif cfg.amp_dtype == "float16":
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = torch.float32

        # Default: no AMP on MPS/CPU unless explicitly requested
        if self.device_type in ("mps", "cpu") and cfg.amp_flag is None:
            cfg.use_amp = False

        self.use_autocast = cfg.use_amp and self.device_type in ("cuda", "mps")
        print(f"Autocast dtype: {self.autocast_dtype} | AMP: {cfg.use_amp}")

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")

        print(f"Loading tokenizer: {cfg.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Peek vocab sizes from resume/init checkpoints so we never accidentally shrink
        resume_meta = None
        init_meta = None
        if cfg.resume:
            resume_meta = torch.load(cfg.resume, map_location="cpu", weights_only=False)
            self.resume_vocab_size = self._extract_vocab_size_from_ckpt(resume_meta)
        if cfg.init_checkpoint:
            init_meta = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=False)
            self.init_vocab_size = self._extract_vocab_size_from_ckpt(init_meta)

        # Load model config
        if cfg.model_config_path:
            print(f"Loading model config from: {cfg.model_config_path}")
            with open(cfg.model_config_path, "r") as f:
                json_cfg = json.load(f)
            arch = json_cfg.get("architecture", json_cfg)
            model_config = OnyxConfig(
                d_model=arch.get("d_model", 384),
                n_layers=arch.get("n_layers", 6),
                n_heads=arch.get("n_heads", 6),
                n_kv_heads=arch.get("n_kv_heads", 2),
                d_ff=arch.get("d_ff", 4096),
                vocab_size=arch.get("vocab_size", 128258),
                max_seq_len=cfg.max_seq_len,
                train_seq_len=cfg.max_seq_len,
                rope_base=arch.get("rope_theta", arch.get("rope_base", 500000.0)),
                norm_eps=arch.get("norm_eps", 1e-5),
                use_flash_attention=(self.device_type == "cuda"),
                gradient_checkpointing=(self.device_type == "cuda"),
                memory_reg_weight=cfg.memory_reg_weight,
            )
        else:
            print("Creating Onyx model with default config...")
            model_config = OnyxConfig(
                use_flash_attention=(self.device_type == "cuda"),
                gradient_checkpointing=(self.device_type == "cuda"),
                memory_reg_weight=cfg.memory_reg_weight,
                max_seq_len=cfg.max_seq_len,
                train_seq_len=cfg.max_seq_len,
            )

        # Align vocab size (never shrink below any checkpoint vocab if provided)
        tok_vs = len(self.tokenizer)
        ckpt_vs = max([v for v in [self.resume_vocab_size, self.init_vocab_size] if v is not None], default=0)
        target_vs = max(tok_vs, ckpt_vs, model_config.vocab_size)
        if target_vs != model_config.vocab_size:
            print(f"Adjusting model vocab_size from {model_config.vocab_size} to {target_vs} to match tokenizer/checkpoint.")
            model_config.vocab_size = target_vs

        self.model = Onyx(model_config).to(self.device)

        # Load weights from init_checkpoint (weights only)
        if cfg.init_checkpoint:
            print(f"Loading model weights from init_checkpoint: {cfg.init_checkpoint}")
            sd = init_meta.get("model_state_dict") if isinstance(init_meta, dict) and "model_state_dict" in init_meta else init_meta
            if not isinstance(sd, dict):
                raise ValueError("init_checkpoint did not contain a state_dict-like mapping")

            # Resize vocab-dependent tensors if needed
            if hasattr(self.model, "embed"):
                self._resize_vocab_tensor(sd, "embed.weight", self.model.embed.weight)
            if hasattr(self.model, "lm_head"):
                self._resize_vocab_tensor(sd, "lm_head.weight", self.model.lm_head.weight)

            self.model.load_state_dict(sd, strict=True)
            print("Init weights loaded.")

        # Dataset + loader
        print(f"Loading dataset from: {cfg.data_glob}")
        self.dataset = StreamingPackedDataset(
            data_glob=cfg.data_glob,
            tokenizer=self.tokenizer,
            max_seq_len=cfg.max_seq_len,
            pack=cfg.pack_sequences,
            seed=cfg.seed,
            drop_remainder=cfg.drop_remainder,
            emit_cu_seqlens=(self.device_type == "cuda"),
            shuffle_buffer_docs=cfg.shuffle_buffer_docs,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            num_workers=0,
            pin_memory=(self.device_type == "cuda"),
            collate_fn=collate_onyx,
        )
        self.data_iter = iter(self.dataloader)

        # Optimizer
        param_groups = get_param_groups(
            self.model,
            weight_decay=cfg.weight_decay,
            memory_lr_scale=cfg.memory_lr_scale,
        )

        if cfg.use_m3_optimizer:
            print("Using M3 optimizer")
            self.optimizer = M3Optimizer(
                param_groups,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                beta_slow=cfg.m3_beta_slow,
                slow_freq=cfg.m3_slow_freq,
                slow_weight=cfg.m3_slow_weight,
            )
        else:
            print("Using AdamW optimizer")
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=cfg.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=cfg.weight_decay,
            )

        tokens_per_batch = cfg.batch_size * cfg.max_seq_len
        self.accumulation_steps = max(1, cfg.tokens_per_step // max(1, tokens_per_batch))
        print(f"Gradient accumulation: {self.accumulation_steps} steps")
        print(f"Tokens per optimizer step: {self.accumulation_steps * tokens_per_batch:,}")

        self.scaler = None
        if cfg.use_amp and cfg.amp_dtype == "float16":
            if self.device_type == "cuda":
                self.scaler = torch.amp.GradScaler("cuda")
            elif self.device_type == "mps":
                try:
                    self.scaler = torch.amp.GradScaler("mps")
                    print("Using GradScaler on MPS.")
                except Exception as e:
                    print(f"GradScaler not available on MPS: {e}")

        self.global_step = 0
        self.tokens_seen = 0
        self.best_loss = float("inf")
        self.current_epoch = 0

        os.makedirs(cfg.save_dir, exist_ok=True)

        # Resume full state if requested
        if cfg.resume:
            self.load_checkpoint(cfg.resume)

        if cfg.wandb_project and WANDB_AVAILABLE:
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config={**vars(cfg), "device": str(self.device)},
            )

        if cfg.use_torch_compile and self.device_type == "cuda" and hasattr(torch, "compile"):
            print(f"Compiling model with mode: {cfg.compile_mode}")
            self.model = torch.compile(self.model, mode=cfg.compile_mode)

        print("\nSetup complete!\n")

    def get_batch(self) -> Optional[Dict[str, Any]]:
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            try:
                batch = next(self.data_iter)
            except StopIteration:
                return None

        out: Dict[str, Any] = {}
        out["input_ids"] = batch["input_ids"].to(self.device)
        out["labels"] = batch["labels"].to(self.device)
        if self.device_type == "cuda" and "cu_seqlens" in batch:
            out["cu_seqlens"] = batch["cu_seqlens"]  # list[Tensor] on CPU ok
            out["max_seqlen"] = batch["max_seqlen"]  # Tensor on CPU
        return out

    def train_step(self, memory_states=None):
        cfg = self.config
        self.model.train()

        if memory_states is None:
            memory_states = self.model.init_memory_states(
                cfg.batch_size,
                self.device,
                self.autocast_dtype if self.use_autocast else torch.float32,
            )

        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_tokens = 0

        for _ in range(self.accumulation_steps):
            batch = self.get_batch()
            if batch is None:
                break

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            total_tokens += input_ids.numel()

            cu_seqlens = None
            max_seqlen = None
            if self.device_type == "cuda" and "cu_seqlens" in batch:
                # Build combined cu_seqlens across batch (flattened)
                B, S = input_ids.shape
                combined = [0]
                offset = 0
                for b in range(B):
                    sample_cu = batch["cu_seqlens"][b]
                    for i in range(1, len(sample_cu)):
                        combined.append(offset + int(sample_cu[i].item()))
                    offset += S
                cu_seqlens = torch.tensor(combined, dtype=torch.int32, device=self.device)
                max_seqlen = int(batch["max_seqlen"].max().item()) if torch.is_tensor(batch["max_seqlen"]) else S

            memory_states = self.model.detach_memory_states(memory_states)

            if self.use_autocast and self.device_type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        memory_states=memory_states,
                        update_memories=True,
                        return_memory_reg_loss=True,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )
            elif self.use_autocast and self.device_type == "mps":
                with torch.autocast(device_type="mps", dtype=self.autocast_dtype):
                    out = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        memory_states=memory_states,
                        update_memories=True,
                        return_memory_reg_loss=True,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )
            else:
                out = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    memory_states=memory_states,
                    update_memories=True,
                    return_memory_reg_loss=True,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

            memory_states = out["memory_states"]

            loss = out["loss"] / self.accumulation_steps
            mem_reg = out.get("memory_reg_loss", 0.0)
            if isinstance(mem_reg, torch.Tensor):
                mem_reg = mem_reg / self.accumulation_steps

            combined_loss = loss + mem_reg if isinstance(mem_reg, torch.Tensor) else loss

            if self.scaler is not None:
                self.scaler.scale(combined_loss).backward()
            else:
                combined_loss.backward()

            total_loss += float(loss.item()) * self.accumulation_steps

        if total_tokens == 0:
            return None, memory_states

        if not hasattr(self, "_cms_manager"):
            self._cms_manager = None
            model_cfg = getattr(self.model, "config", None)
            if model_cfg is not None and getattr(model_cfg, "use_cms_ffn", False):
                self._cms_manager = CMSFrequencyManager(self.model, model_cfg)

        if self._cms_manager is not None:
            self._cms_manager.mask_gradients(self.global_step)

        if cfg.gradient_clip > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
            grad_norm_val = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
        else:
            grad_norm_val = 0.0

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.global_step += 1
        self.tokens_seen += total_tokens

        return {
            "loss": total_loss / max(1, self.accumulation_steps),
            "grad_norm": grad_norm_val,
            "tokens": total_tokens,
        }, memory_states

    def train(self):
        if not hasattr(self, "model"):
            self.setup()

        cfg = self.config
        bench_mode = cfg.bench_steps and cfg.bench_steps > 0

        if cfg.max_steps is not None:
            total_steps = int(cfg.max_steps)
        elif cfg.train_tokens_target is not None:
            total_steps = int(cfg.train_tokens_target // max(1, cfg.tokens_per_step))
        else:
            # If neither is provided, run "epochs" based on a conservative estimate:
            # we approximate steps per epoch by counting batches for one pass.
            print("No --max_steps or --train_tokens_target provided; estimating steps_per_epoch by one streaming pass...")
            est_steps = 0
            # NOTE: this is a streaming pass; it will take time but not load all docs into RAM.
            for _ in self.dataloader:
                est_steps += 1
            est_steps = max(1, est_steps // max(1, self.accumulation_steps))
            total_steps = est_steps * max(1, int(cfg.num_epochs))
            # reset iterator
            self.data_iter = iter(self.dataloader)
            print(f"Estimated steps_per_epoch={est_steps:,} -> total_steps={total_steps:,}")

        if bench_mode:
            total_steps = self.global_step + int(cfg.bench_steps)

        print("=" * 70)
        print("Starting training")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Save every steps: {cfg.save_every_steps}")
        print(f"  keep_last_n: {cfg.keep_last_n} | keep_every_steps: {cfg.keep_every_steps}")
        print("=" * 70)

        memory_states = None
        start_time = time.time()
        log_loss = 0.0
        log_tokens = 0
        log_steps = 0

        while self.global_step < total_steps:
            if self._interrupt_requested:
                print("\n[INTERRUPT] Saving checkpoint before exit...")
                self.save_checkpoint(f"interrupt_step_{self.global_step}")
                print("Checkpoint saved. Exiting.")
                return

            metrics, memory_states = self.train_step(memory_states)
            if metrics is None:
                continue

            lr = get_lr(self.global_step, cfg, total_steps)
            set_lr(self.optimizer, lr)

            log_loss += metrics["loss"]
            log_tokens += metrics["tokens"]
            log_steps += 1

            if self.global_step % cfg.log_every == 0:
                elapsed = time.time() - start_time
                avg_loss = log_loss / max(1, log_steps)
                tokps = log_tokens / max(1e-8, elapsed)

                print(
                    f"Step {self.global_step:6d} | Loss {avg_loss:.4f} | LR {lr:.2e} | Tok/s {tokps:.0f}",
                    flush=True,
                )

                if cfg.wandb_project and WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/grad_norm": metrics["grad_norm"],
                        "train/tokps": tokps,
                        "train/tokens_seen": self.tokens_seen,
                        "train/step": self.global_step,
                    })

                log_loss = 0.0
                log_tokens = 0
                log_steps = 0
                start_time = time.time()

            if cfg.save_every_steps > 0 and (self.global_step % cfg.save_every_steps == 0) and not (bench_mode and cfg.disable_saves_during_bench):
                self.save_checkpoint(f"step_{self.global_step}")

            if bench_mode and (self.global_step >= total_steps):
                self.save_checkpoint(f"bench_step_{self.global_step}")
                print("Benchmark complete. Exiting.")
                return

        self.save_checkpoint(f"final_step_{self.global_step}")
        print("\nTraining complete!")
        print(f"  Total steps: {self.global_step:,}")
        print(f"  Total tokens: {self.tokens_seen:,}")

    def save_checkpoint(self, name: str):
        cfg = self.config
        os.makedirs(cfg.save_dir, exist_ok=True)

        path = os.path.join(cfg.save_dir, f"checkpoint_{name}.pt")
        tmp_path = path + ".tmp"

        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_loss": self.best_loss,
            "current_epoch": getattr(self, "current_epoch", 0),
            "config": vars(cfg),
        }
        if self.scaler is not None:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)  # atomic
        print(f"Saved checkpoint: {path}")

        self._cleanup_checkpoints()

    def load_checkpoint(self, path: str):
        print(f"Resuming from: {path}")
        map_location = getattr(self, "device", torch.device("cpu"))
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        sd = ckpt["model_state_dict"]
        # Resize vocab tensors if needed
        if hasattr(model, "embed"):
            self._resize_vocab_tensor(sd, "embed.weight", model.embed.weight)
        if hasattr(model, "lm_head"):
            self._resize_vocab_tensor(sd, "lm_head.weight", model.lm_head.weight)

        model.load_state_dict(sd, strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.global_step = int(ckpt.get("global_step", 0))
        self.tokens_seen = int(ckpt.get("tokens_seen", 0))
        self.best_loss = float(ckpt.get("best_loss", float("inf")))
        self.current_epoch = int(ckpt.get("current_epoch", 0))

        print(f"  Resumed at step {self.global_step}, tokens_seen={self.tokens_seen:,}")

    def _cleanup_checkpoints(self):
        cfg = self.config
        if cfg.keep_last_n <= 0:
            return

        # Only manage step checkpoints; keep "final"/"epoch"/"interrupt" etc.
        step_ckpts = sorted(Path(cfg.save_dir).glob("checkpoint_step_*.pt"))
        if len(step_ckpts) <= cfg.keep_last_n:
            return

        def _step_num(p: Path) -> Optional[int]:
            # checkpoint_step_12345.pt
            s = p.stem
            try:
                return int(s.split("_")[-1])
            except Exception:
                return None

        keep: set[Path] = set()

        # Keep milestone steps if requested
        if cfg.keep_every_steps and cfg.keep_every_steps > 0:
            for p in step_ckpts:
                n = _step_num(p)
                if n is not None and n % cfg.keep_every_steps == 0:
                    keep.add(p)

        # Keep newest N by mtime
        # Keep newest N by step number (deterministic; avoids coarse mtime flakiness)
        newest = sorted(step_ckpts, key=lambda p: (_step_num(p) if _step_num(p) is not None else -1), reverse=True)[: cfg.keep_last_n]
        keep.update(newest)

        # Delete others
        for p in step_ckpts:
            if p not in keep:
                try:
                    p.unlink()
                except Exception as e:
                    print(f"[WARN] Could not delete old checkpoint {p}: {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Train Onyx Model")

    p.add_argument("--data_glob", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="NousResearch/Hermes-2-Pro-Llama-3-8B")
    p.add_argument("--model_config", type=str, default=None)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--tokens_per_step", type=int, default=8192)
    p.add_argument("--no_pack_sequences", action="store_false", dest="pack_sequences")
    p.add_argument("--drop_remainder", action="store_true")
    p.add_argument("--shuffle_buffer_docs", type=int, default=2048)

    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--train_tokens_target", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)

    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--memory_lr_scale", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.1)

    p.add_argument("--use_adamw", action="store_true")

    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", type=str, default="reduce-overhead")

    p.add_argument("--amp", action="store_true", default=None)
    p.add_argument("--amp_dtype", type=str, choices=["float16", "bfloat16"], default="float16")

    p.add_argument("--bench_steps", type=int, default=0)
    p.add_argument("--disable_saves_during_bench", action="store_true")

    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--save_every_steps", type=int, default=0)
    p.add_argument("--keep_last_n", type=int, default=5)
    p.add_argument("--keep_every_steps", type=int, default=0)

    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--init_checkpoint", type=str, default=None)

    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # AMP default behavior
    amp_default = TrainingConfig.__dataclass_fields__["use_amp"].default
    use_amp = args.amp if args.amp is not None else amp_default

    cfg = TrainingConfig(
        data_glob=args.data_glob,
        tokenizer_name=args.tokenizer,
        model_config_path=args.model_config,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        tokens_per_step=args.tokens_per_step,
        pack_sequences=args.pack_sequences,
        drop_remainder=args.drop_remainder,
        shuffle_buffer_docs=args.shuffle_buffer_docs,
        num_epochs=args.num_epochs,
        train_tokens_target=args.train_tokens_target,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        memory_lr_scale=args.memory_lr_scale,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        use_m3_optimizer=(not args.use_adamw),
        use_torch_compile=args.compile,
        compile_mode=args.compile_mode,
        use_amp=use_amp,
        amp_dtype=args.amp_dtype,
        amp_flag=args.amp,
        bench_steps=args.bench_steps,
        disable_saves_during_bench=args.disable_saves_during_bench,
        save_dir=args.save_dir,
        save_every_steps=args.save_every_steps,
        keep_last_n=args.keep_last_n,
        keep_every_steps=args.keep_every_steps,
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
        log_every=args.log_every,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    if args.dry_run:
        cfg.max_steps = 3
        cfg.log_every = 1
        cfg.save_every_steps = 0
        cfg.use_torch_compile = False
        cfg.wandb_project = None
        print("\nDRY RUN MODE (3 steps)\n")

    Trainer(cfg).train()


if __name__ == "__main__":
    main()
