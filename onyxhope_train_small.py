#!/usr/bin/env python3
"""
Training script for OnyxHope Small (~25-30M params)
Optimized for M2 MacBook Pro with MPS backend

Author: Marvin Tutt, Caia Tech
"""

import os
import sys
import json
import glob
import time
import math
import shutil
import signal
import random
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Import model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onyxhope_model_small import (
    OnyxHopeSmall, OnyxHopeSmallConfig, M3Optimizer, create_m3_optimizer
)

# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# Training Config
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration optimized for M2 MacBook Pro"""

    # Data
    data_glob: str = "/Users/owner/Desktop/caiatech/models/onyx-300m/onyx-update-small/master-dataset.jsonl"
    tokenizer: str = "NousResearch/Hermes-2-Pro-Llama-3-8B"
    eval_glob: Optional[str] = None

    # Training (reduced for M2)
    tokens_per_step: int = 4_000       # Reduced for memory
    max_steps: Optional[int] = 1000
    train_tokens_target: Optional[int] = 20_000_000  # 20M tokens
    max_seq_len: int = 256             # Shorter sequences for memory
    fill_ratio: float = 0.9
    batch_size: int = 2                # Small batch for memory

    # Optimization
    learning_rate: float = 1e-3        # Higher LR for smaller model
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    # Optimizer type
    optimizer_type: str = "m3"         # "adamw" or "m3"
    m3_beta_slow: float = 0.99
    m3_slow_freq: int = 50
    m3_slow_weight: float = 0.1

    # Checkpointing
    save_dir: str = "./checkpoints_hope_small"
    save_every_steps: int = 200
    eval_every_steps: int = 100
    resume: Optional[str] = None

    # Logging
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None
    log_every: int = 10
    loss_ema_alpha: float = 0.1

    # System
    num_workers: int = 0
    seed: int = 1337
    dry_run: bool = False

    # Model options
    use_hope_attention: bool = True
    use_cms_ffn: bool = True
    cms_num_levels: int = 2
    memory_lr_init: float = 0.1
    memory_decay_init: float = 0.95


# ============================================================================
# Dataset
# ============================================================================

class PackedDocument:
    def __init__(self, input_ids: List[int], doc_spans: List[Tuple[int, int]], seq_len: int):
        self.input_ids = input_ids
        self.doc_spans = doc_spans
        self.seq_len = seq_len


class StreamingPackedDataset(IterableDataset):
    """Streaming dataset that packs documents into sequences"""

    def __init__(
        self,
        file_pattern: str,
        tokenizer,
        max_seq_len: int = 512,
        fill_ratio: float = 0.9,
        eod_token_id: int = 3,
        pad_token_id: int = 0,
        seed: int = 42
    ):
        self.file_pattern = file_pattern
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.fill_ratio = fill_ratio
        self.eod_token_id = eod_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

        self.files = sorted(glob.glob(file_pattern, recursive=True))
        if not self.files:
            raise ValueError(f"No files found matching {file_pattern}")

        self.stats = defaultdict(int)
        self._epoch = 0

    def _pack_documents(self, doc_iterator, target_len: int) -> Optional[PackedDocument]:
        packed_ids, doc_spans = [], []
        current_pos = 0
        target_fill = int(target_len * self.fill_ratio)

        for tokens in doc_iterator:
            try:
                if not tokens:
                    continue

                new_len = current_pos + len(tokens) + (1 if packed_ids else 0)
                if new_len > target_len:
                    if current_pos >= target_fill:
                        break
                    if len(tokens) > target_len - current_pos:
                        continue

                if packed_ids:
                    packed_ids.append(self.eod_token_id)
                    current_pos += 1

                start = current_pos
                packed_ids.extend(tokens)
                current_pos += len(tokens)
                doc_spans.append((start, current_pos))

                if current_pos >= target_fill:
                    break

            except Exception:
                continue

        if packed_ids and (current_pos + 1) <= target_len and packed_ids[-1] != self.eod_token_id:
            packed_ids.append(self.eod_token_id)
            current_pos += 1

        if not packed_ids:
            return None

        return PackedDocument(packed_ids, doc_spans, len(packed_ids))

    def _read_jsonl_file(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            warnings.warn(f"Error reading {filepath}: {e}", stacklevel=2)

    def _batched_token_stream(self, files, batch_size: int = 16):
        buf = []
        def _flush(buf):
            if not buf:
                return []
            enc = self.tokenizer(
                buf,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False
            )
            return enc["input_ids"]

        for fp in files:
            for doc in self._read_jsonl_file(fp):
                text = doc if isinstance(doc, str) else (doc.get("text", "") or doc.get("content", "")) if isinstance(doc, dict) else ""
                if not text.strip():
                    continue
                buf.append(text)
                if len(buf) >= batch_size:
                    for tokens in _flush(buf):
                        yield tokens
                    buf.clear()
        for tokens in _flush(buf):
            yield tokens

    def __iter__(self):
        files = list(self.files)
        random.Random(self.seed + self._epoch).shuffle(files)
        self._epoch += 1

        def token_stream():
            for filepath in files:
                yield from self._batched_token_stream([filepath])

        doc_iter = token_stream()

        while True:
            packed = self._pack_documents(doc_iter, self.max_seq_len)
            if packed is None:
                doc_iter = token_stream()
                continue

            labels = packed.input_ids[1:] + [-100]

            self.stats["total_sequences"] += 1
            self.stats["total_tokens"] += packed.seq_len

            yield {
                "input_ids": torch.tensor(packed.input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "seq_len": packed.seq_len,
                "doc_spans": packed.doc_spans
            }


def create_dataloader(config: TrainingConfig, tokenizer, is_eval: bool = False) -> DataLoader:
    file_pattern = config.eval_glob if is_eval else config.data_glob
    if not file_pattern:
        return None

    dataset = StreamingPackedDataset(
        file_pattern=file_pattern,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        fill_ratio=config.fill_ratio,
        eod_token_id=tokenizer.convert_tokens_to_ids("<eod>"),
        pad_token_id=tokenizer.pad_token_id,
        seed=config.seed
    )

    def collate_fn(batch):
        seqs = [x["input_ids"] for x in batch]
        labs = [x["labels"] for x in batch]
        seq_lens = torch.tensor([x["seq_len"] for x in batch], dtype=torch.long)

        S = max(int(t.size(0)) for t in seqs)
        pad_id = tokenizer.pad_token_id

        def pad_to(t, length, pad_value):
            if t.size(0) == length:
                return t
            out = torch.full((length,), pad_value, dtype=t.dtype)
            out[:t.size(0)] = t
            return out

        input_ids = torch.stack([pad_to(t, S, pad_id) for t in seqs])
        labels = torch.stack([pad_to(t, S, -100) for t in labs])

        return {"input_ids": input_ids, "labels": labels, "seq_lens": seq_lens}

    batch_size = config.batch_size if not is_eval else 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        drop_last=False
    )


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    def __init__(self, model: OnyxHopeSmall, config: TrainingConfig, tokenizer, device: torch.device):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        self.optimizer = self._create_optimizer()
        self.scheduler = None

        self.global_step = 0
        self.global_tokens = 0
        self.accum_token_cap = config.tokens_per_step

        self.memory_states = None

        self.loss_history = deque(maxlen=1000)
        self.step_times = deque(maxlen=100)

        self.loss_ema = None
        self.grad_norm_ema = None
        self.throughput_ema = None
        self.ema_alpha = config.loss_ema_alpha

        self.best_eval_loss = float('inf')
        self.checkpoint_dir = Path(config.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = WANDB_AVAILABLE and config.wandb_project
        if self.use_wandb:
            wandb.init(project=config.wandb_project, name=config.wandb_run, config=asdict(config))

    def _create_optimizer(self):
        if self.config.optimizer_type == "m3":
            optimizer = create_m3_optimizer(
                self.model,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                beta_slow=self.config.m3_beta_slow,
                slow_freq=self.config.m3_slow_freq,
                slow_weight=self.config.m3_slow_weight,
                eps=self.config.adam_eps,
            )
            print(f"Using M3 optimizer (slow_freq={self.config.m3_slow_freq})")
            return optimizer

        # AdamW fallback
        groups = self.model.get_param_groups()
        optimizer = AdamW(
            [
                {"params": groups["decay"], "weight_decay": self.config.weight_decay},
                {"params": groups["no_decay"], "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps
        )
        print("Using AdamW optimizer")
        return optimizer

    def _create_scheduler(self, num_steps: int):
        from torch.optim.lr_scheduler import LambdaLR
        warm = self.config.warmup_steps
        base = self.config.learning_rate
        eta_min = self.config.min_lr

        def lr_lambda(step):
            if step < warm:
                return float(step) / float(max(1, warm))
            progress = float(step - warm) / float(max(1, num_steps - warm))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (eta_min / base) + (1 - eta_min / base) * cos

        return LambdaLR(self.optimizer, lr_lambda)

    def _init_memory_states(self, batch_size: int):
        return self.model.init_all_memory_states(batch_size, self.device, torch.float32)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Initialize memory states if needed
        if self.memory_states is None:
            self.memory_states = self._init_memory_states(input_ids.shape[0])

        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            memory_states=self.memory_states,
            update_memories=True,
        )

        # Update memory states
        self.memory_states = outputs['memory_states']

        loss = outputs["loss"]
        effective_tokens = batch["seq_lens"].sum().item()

        loss.backward()

        return {"loss": float(loss.item()), "tokens": effective_tokens}

    def optimizer_step(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, num_steps: int = 30) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_tokens = [], 0

        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                memory_states=self.memory_states,
                update_memories=False,
            )

            total_loss.append(outputs["loss"].item())
            total_tokens += batch["seq_lens"].sum().item()

        return {
            "eval_loss": float(np.mean(total_loss)),
            "eval_ppl": float(np.exp(np.mean(total_loss))),
            "eval_tokens": int(total_tokens)
        }

    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        print("\n" + "=" * 70)
        print("Starting OnyxHope Small Training")
        print(f"   Device: {self.device}")
        print(f"   Parameters: {self.model.get_num_params():,}")
        print(f"   Optimizer: {self.config.optimizer_type.upper()}")
        print(f"   Tokens per step: {self.config.tokens_per_step:,}")
        print(f"   Target tokens: {self.config.train_tokens_target:,}")
        print(f"   Max steps: {self.config.max_steps}")
        print(f"   LR: {self.config.learning_rate:.2e} -> {self.config.min_lr:.2e}")
        print(f"   Sequence length: {self.config.max_seq_len}")
        print(f"   Batch size: {self.config.batch_size}")
        print("=" * 70 + "\n")

        total_steps = self.config.max_steps or (self.config.train_tokens_target // self.config.tokens_per_step)

        if self.scheduler is None:
            self.scheduler = self._create_scheduler(total_steps)

        pbar = tqdm(total=total_steps, initial=self.global_step, desc="Training", ncols=100)

        accumulated_loss = 0.0
        accumulated_tokens = 0
        accumulated_steps = 0
        last_log_time = time.time()
        step_start_time = time.time()

        def signal_handler(signum, frame):
            print("\nInterrupt received, saving...")
            self.save_checkpoint("interrupt")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            for batch_idx, batch in enumerate(train_dataloader):
                if self.global_tokens >= self.config.train_tokens_target:
                    print("\nReached target tokens!")
                    break
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    print("\nReached max steps!")
                    break

                # Handle batch size changes
                current_bs = batch["input_ids"].shape[0]
                if self.memory_states is not None:
                    mem_bs = len(self.memory_states[0].get('attention', {}).get('k', torch.empty(0)).shape)
                    if mem_bs > 0 and current_bs != mem_bs:
                        self.memory_states = self._init_memory_states(current_bs)

                try:
                    metrics = self.train_step(batch)
                    accumulated_loss += metrics["loss"]
                    accumulated_tokens += metrics["tokens"]
                    accumulated_steps += 1

                    if accumulated_tokens >= self.accum_token_cap:
                        step_time = time.time() - step_start_time
                        self.step_times.append(step_time)

                        grad_norm = self.optimizer_step()

                        self.global_step += 1
                        self.global_tokens += accumulated_tokens

                        avg_loss = accumulated_loss / max(1, accumulated_steps)
                        tokens_per_sec = accumulated_tokens / max(1e-6, time.time() - last_log_time)
                        current_lr = self.scheduler.get_last_lr()[0]

                        # Update EMAs
                        if self.loss_ema is None or self.throughput_ema is None:
                            self.loss_ema = avg_loss if self.loss_ema is None else self.loss_ema
                            self.grad_norm_ema = grad_norm if self.grad_norm_ema is None else self.grad_norm_ema
                            self.throughput_ema = tokens_per_sec
                        else:
                            self.loss_ema = self.ema_alpha * avg_loss + (1 - self.ema_alpha) * self.loss_ema
                            self.grad_norm_ema = self.ema_alpha * grad_norm + (1 - self.ema_alpha) * self.grad_norm_ema
                            self.throughput_ema = self.ema_alpha * tokens_per_sec + (1 - self.ema_alpha) * self.throughput_ema

                        self.loss_history.append(avg_loss)

                        if self.global_step % self.config.log_every == 0:
                            avg_step_time = np.mean(list(self.step_times)[-10:])
                            remaining = total_steps - self.global_step
                            eta_mins = remaining * avg_step_time / 60

                            print(f"\n[Step {self.global_step}/{total_steps}] "
                                  f"Loss: {avg_loss:.4f} (EMA: {self.loss_ema:.4f}) | "
                                  f"PPL: {np.exp(avg_loss):.1f} | "
                                  f"LR: {current_lr:.2e} | "
                                  f"Grad: {grad_norm:.3f} | "
                                  f"Tok/s: {tokens_per_sec:.0f} | "
                                  f"ETA: {eta_mins:.1f}m")

                            # Memory norm logging
                            if self.memory_states and len(self.memory_states) > 0:
                                attn = self.memory_states[0].get('attention', {})
                                if 'k' in attn:
                                    print(f"   Memory K norm: {attn['k'].norm().item():.3f}, "
                                          f"V norm: {attn.get('v', attn['k']).norm().item():.3f}")

                            pbar.set_postfix({
                                "loss": f"{avg_loss:.3f}",
                                "ppl": f"{np.exp(avg_loss):.0f}",
                                "tok/s": f"{tokens_per_sec:.0f}"
                            })
                            pbar.update(self.config.log_every)

                            if self.use_wandb:
                                wandb.log({
                                    "loss": avg_loss,
                                    "loss_ema": self.loss_ema,
                                    "ppl": np.exp(avg_loss),
                                    "lr": current_lr,
                                    "grad_norm": grad_norm,
                                    "tokens_per_sec": tokens_per_sec,
                                    "global_tokens": self.global_tokens,
                                }, step=self.global_step)

                        step_start_time = time.time()
                        accumulated_loss = 0.0
                        accumulated_tokens = 0
                        accumulated_steps = 0
                        last_log_time = time.time()

                        # Eval
                        if eval_dataloader and self.global_step % self.config.eval_every_steps == 0:
                            print("\nRunning evaluation...")
                            eval_metrics = self.evaluate(eval_dataloader)
                            print(f"   Eval Loss: {eval_metrics['eval_loss']:.4f}, "
                                  f"PPL: {eval_metrics['eval_ppl']:.1f}")
                            if eval_metrics["eval_loss"] < self.best_eval_loss:
                                self.best_eval_loss = eval_metrics["eval_loss"]
                                self.save_checkpoint("best")
                                print("   New best!")

                        # Save
                        if self.global_step % self.config.save_every_steps == 0:
                            self.save_checkpoint("step")

                except Exception as e:
                    print(f"\nError at step {self.global_step}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                if self.config.dry_run and self.global_step >= 10:
                    print("\nDry run complete!")
                    break

        finally:
            pbar.close()
            self.save_checkpoint("final")
            self._print_summary()

    def _print_summary(self):
        total_time = sum(self.step_times)
        avg_step = np.mean(list(self.step_times)) if self.step_times else 0
        final_loss = list(self.loss_history)[-1] if self.loss_history else 0
        initial_loss = list(self.loss_history)[0] if self.loss_history else 0

        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"   Steps: {self.global_step:,}")
        print(f"   Tokens: {self.global_tokens:,}")
        print(f"   Time: {total_time/60:.1f} minutes")
        print(f"   Avg step: {avg_step:.2f}s")
        print(f"   Initial loss: {initial_loss:.4f}")
        print(f"   Final loss: {final_loss:.4f}")
        print(f"   Improvement: {initial_loss - final_loss:.4f}")
        print(f"   Best eval: {self.best_eval_loss:.4f}")
        print(f"   Checkpoints: {self.checkpoint_dir}")
        print("=" * 70)

    def save_checkpoint(self, tag: str = "step"):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "global_step": self.global_step,
            "global_tokens": self.global_tokens,
            "best_eval_loss": self.best_eval_loss,
            "loss_ema": self.loss_ema,
            "grad_norm_ema": self.grad_norm_ema,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.memory_states:
            checkpoint["memory_states"] = self.memory_states

        path = self.checkpoint_dir / f"checkpoint_{tag}_{self.global_step}.pt"
        torch.save(checkpoint, path)
        shutil.copy(path, self.checkpoint_dir / "checkpoint_latest.pt")

        with open(self.checkpoint_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        print(f"Saved: {path}")

    def load_checkpoint(self, path: Union[str, Path]):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.global_tokens = checkpoint.get("global_tokens", 0)
        self.best_eval_loss = checkpoint.get("best_eval_loss", float('inf'))
        self.loss_ema = checkpoint.get("loss_ema")
        self.grad_norm_ema = checkpoint.get("grad_norm_ema")

        if "memory_states" in checkpoint:
            self.memory_states = checkpoint["memory_states"]

        print(f"Resumed from step {self.global_step}, tokens {self.global_tokens:,}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train OnyxHope Small")

    # Data
    parser.add_argument("--data_glob", type=str,
                       default="/Users/owner/Desktop/caiatech/models/onyx-300m/onyx-update-small/master-dataset.jsonl")
    parser.add_argument("--eval_glob", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Hermes-2-Pro-Llama-3-8B")

    # Training
    parser.add_argument("--tokens_per_step", type=int, default=4_000)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--train_tokens_target", type=int, default=20_000_000)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--optimizer_type", type=str, default="m3", choices=["adamw", "m3"])
    parser.add_argument("--m3_slow_freq", type=int, default=50)

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="./checkpoints_hope_small")
    parser.add_argument("--save_every_steps", type=int, default=200)
    parser.add_argument("--eval_every_steps", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=10)

    # System
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry_run", action="store_true")

    # Model
    parser.add_argument("--no_hope_attention", action="store_true")
    parser.add_argument("--no_cms_ffn", action="store_true")
    parser.add_argument("--cms_num_levels", type=int, default=2)

    args = parser.parse_args()

    config = TrainingConfig(
        data_glob=args.data_glob,
        eval_glob=args.eval_glob,
        tokenizer=args.tokenizer,
        tokens_per_step=args.tokens_per_step,
        max_steps=args.max_steps,
        train_tokens_target=args.train_tokens_target,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        optimizer_type=args.optimizer_type,
        m3_slow_freq=args.m3_slow_freq,
        save_dir=args.save_dir,
        save_every_steps=args.save_every_steps,
        eval_every_steps=args.eval_every_steps,
        resume=args.resume,
        wandb_project=args.wandb_project,
        log_every=args.log_every,
        seed=args.seed,
        dry_run=args.dry_run,
        use_hope_attention=not args.no_hope_attention,
        use_cms_ffn=not args.no_cms_ffn,
        cms_num_levels=args.cms_num_levels,
    )

    # Seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Tokenizer
    print(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, use_fast=True, trust_remote_code=True)

    special_tokens = {}
    if "<eod>" not in tokenizer.get_vocab():
        special_tokens["additional_special_tokens"] = tokenizer.additional_special_tokens + ["<eod>"]
    if tokenizer.pad_token is None:
        if "<pad>" not in tokenizer.get_vocab():
            special_tokens["pad_token"] = "<pad>"
        else:
            tokenizer.pad_token = "<pad>"
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)

    eod_id = tokenizer.convert_tokens_to_ids("<eod>")
    print(f"   Vocab: {len(tokenizer)}, PAD: {tokenizer.pad_token_id}, EOD: {eod_id}")

    # Model
    print("Creating OnyxHope Small...")
    model_config = OnyxHopeSmallConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eod_token_id=eod_id,
        max_seq_len=config.max_seq_len,
        use_hope_attention=config.use_hope_attention,
        use_cms_ffn=config.use_cms_ffn,
        cms_num_levels=config.cms_num_levels,
        memory_lr_init=config.memory_lr_init,
        memory_decay_init=config.memory_decay_init,
    )

    model = OnyxHopeSmall(model_config).to(device=device, dtype=torch.float32)
    print(f"   Parameters: {model.get_num_params():,}")
    print(f"   Config: d={model_config.d_model}, L={model_config.n_layers}, H={model_config.n_heads}")
    print(f"   Hope Attention: {config.use_hope_attention}")
    print(f"   CMS FFN: {config.use_cms_ffn} ({config.cms_num_levels} levels)")

    # Data
    print("Creating dataloader...")
    train_dataloader = create_dataloader(config, tokenizer, is_eval=False)

    eval_dataloader = None
    if config.eval_glob:
        eval_dataloader = create_dataloader(config, tokenizer, is_eval=True)

    # Trainer
    trainer = Trainer(model, config, tokenizer, device)

    if config.resume:
        if config.resume == "auto":
            ckpt = Path(config.save_dir) / "checkpoint_latest.pt"
            if ckpt.exists():
                trainer.load_checkpoint(ckpt)
        else:
            trainer.load_checkpoint(config.resume)

    # Train
    trainer.train(train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()
