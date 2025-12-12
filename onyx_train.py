#!/usr/bin/env python3
"""
Onyx 11M Training Script

Features:
- ~11M parameter model (excluding vocab)
- MPS support for Mac (fp32)
- CUDA support with optional bf16/fp16
- Memory state detachment between batches
- Gradient accumulation
- torch.compile support (CUDA only)
- Memory statistics logging to WandB
- Streaming packed dataset
- Signal handler for graceful interrupt/save

Usage:
  # Mac training
  python3 onyx_train.py \
    --data_glob ./data.jsonl \
    --batch_size 4 \
    --max_seq_len 256

  # GPU training
  python3 onyx_train.py \
    --data_glob /workspace/datapack.jsonl \
    --batch_size 16 \
    --max_seq_len 512

Author: Marvin Tutt, Caia Tech
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
from typing import Optional, Iterator, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

# Import model
from onyx_model import (
    Onyx,
    OnyxConfig,
    create_onyx,
    M3Optimizer,
    get_param_groups,
)

# Try to import optional dependencies
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

@dataclass
class TrainingConfig:
    """Training configuration"""

    # === Data ===
    data_glob: str = "./data.jsonl"
    tokenizer_name: str = "NousResearch/Hermes-2-Pro-Llama-3-8B"

    # === Model ===
    # Using OnyxConfig defaults (~11M params)

    # === Sequence ===
    max_seq_len: int = 256
    pack_sequences: bool = True

    # === Batch & Accumulation ===
    batch_size: int = 4
    tokens_per_step: int = 8192

    # === Training Duration ===
    num_epochs: int = 1
    train_tokens_target: Optional[int] = None  # None = use full dataset
    max_steps: Optional[int] = None

    # === Learning Rate ===
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    memory_lr_scale: float = 0.1
    warmup_steps: int = 50  # Fewer warmup for larger batch
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
    amp_dtype: str = "bfloat16"

    # === Compilation ===
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"

    # === Gradient ===
    gradient_clip: float = 1.0

    # === Checkpointing ===
    save_dir: str = "./checkpoints"
    save_every_epoch: bool = True
    save_every_steps: int = 0
    keep_last_n: int = 5
    resume: Optional[str] = None
    init_checkpoint: Optional[str] = None  # Load model weights only (for grown checkpoints)
    model_config_path: Optional[str] = None  # Load model architecture from JSON

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

class StreamingPackedDataset(IterableDataset):
    """Streaming dataset that packs documents into sequences."""

    def __init__(
        self,
        data_glob: str,
        tokenizer,
        max_seq_len: int = 2048,
        pack: bool = True,
        seed: int = 42,
    ):
        self.data_glob = data_glob
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pack = pack
        self.seed = seed

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
        data_glob = self.data_glob
        if '*' not in data_glob and '?' not in data_glob:
            if os.path.exists(data_glob):
                return [data_glob]
            return []
        return sorted(glob(data_glob))

    def _read_documents(self) -> Iterator[str]:
        for filepath in self.data_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        text = data.get('text') or data.get('content') or data.get('input', '')
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        continue

    def _tokenize_document(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens

    def _pack_sequences_with_boundaries(self) -> Iterator[tuple]:
        """Pack sequences and track document boundaries for varlen attention."""
        current_seq = []
        current_boundaries = [0]  # Start positions of each document in the sequence
        rng = random.Random(self.seed)

        docs = list(self._read_documents())
        rng.shuffle(docs)

        for doc in docs:
            tokens = self._tokenize_document(doc)
            if not tokens:
                continue

            tokens.append(self.eod_token_id)
            doc_start_in_seq = len(current_seq)
            current_seq.extend(tokens)

            # Track where this doc ends (relative to sequence start)
            while len(current_seq) >= self.max_seq_len:
                # Yield current full sequence
                seq_to_yield = current_seq[:self.max_seq_len]

                # Adjust boundaries for this sequence
                seq_boundaries = []
                for b in current_boundaries:
                    if b < self.max_seq_len:
                        seq_boundaries.append(b)
                # Add the end boundary
                seq_boundaries.append(self.max_seq_len)

                yield seq_to_yield, seq_boundaries

                # Remainder becomes new sequence
                current_seq = current_seq[self.max_seq_len:]

                # Adjust boundaries for remainder
                current_boundaries = [0]  # New sequence starts fresh
                # If current doc continues into remainder, it starts at 0

        # Handle final partial sequence
        if current_seq:
            if len(current_seq) < self.max_seq_len:
                # Pad and mark the padding boundary
                pad_start = len(current_seq)
                current_seq.extend([self.eod_token_id] * (self.max_seq_len - len(current_seq)))
                current_boundaries.append(pad_start)
            current_boundaries.append(self.max_seq_len)
            yield current_seq[:self.max_seq_len], current_boundaries

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for seq, boundaries in self._pack_sequences_with_boundaries():
            input_ids = torch.tensor(seq, dtype=torch.long)
            # Convert boundaries to cu_seqlens format (cumulative sequence lengths)
            cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)
            max_seqlen = max(boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)) if len(boundaries) > 1 else len(seq)
            yield {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
            }


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_lr(step: int, config: TrainingConfig, total_steps: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    decay_steps = total_steps - config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, decay_steps)
    return config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float, config: TrainingConfig):
    for i, param_group in enumerate(optimizer.param_groups):
        if param_group.get('lr_scale'):
            param_group['lr'] = lr * param_group['lr_scale']
        else:
            param_group['lr'] = lr


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """Training loop handler"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_complete = False

        self._interrupt_requested = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"\n[SIGNAL] Received {signum}, will save and exit after current step...")
        self._interrupt_requested = True

    def setup(self):
        config = self.config

        torch.manual_seed(config.seed)
        random.seed(config.seed)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
            print(f"Using CUDA: {torch.cuda.get_device_name()}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_type = "mps"
            print("Using MPS (Apple Silicon)")
            config.amp_dtype = "float32"
            config.use_amp = False
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"
            print("Using CPU")
            config.use_amp = False

        if config.amp_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif config.amp_dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        print(f"Training dtype: {self.dtype}")

        if TRANSFORMERS_AVAILABLE:
            print(f"Loading tokenizer: {config.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            raise RuntimeError("transformers not installed")

        print(f"Loading dataset from: {config.data_glob}")
        self.dataset = StreamingPackedDataset(
            data_glob=config.data_glob,
            tokenizer=self.tokenizer,
            max_seq_len=config.max_seq_len,
            pack=config.pack_sequences,
            seed=config.seed,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=(self.device_type == "cuda"),
        )
        self.data_iter = iter(self.dataloader)

        # Load model config from JSON, init_checkpoint, or use defaults
        if config.model_config_path:
            print(f"Loading model config from: {config.model_config_path}")
            import json
            with open(config.model_config_path, 'r') as f:
                json_cfg = json.load(f)
            arch = json_cfg.get('architecture', json_cfg)
            mem = json_cfg.get('memory', {})
            model_config = OnyxConfig(
                d_model=arch.get('d_model', 384),
                n_layers=arch.get('n_layers', 6),
                n_heads=arch.get('n_heads', 6),
                n_kv_heads=arch.get('n_kv_heads', 2),
                d_ff=arch.get('d_ff', 4096),
                vocab_size=arch.get('vocab_size', 128258),
                max_seq_len=config.max_seq_len,  # Use training seq len
                train_seq_len=config.max_seq_len,
                rope_base=arch.get('rope_theta', arch.get('rope_base', 500000.0)),
                norm_eps=arch.get('norm_eps', 1e-5),
                use_flash_attention=(self.device_type == "cuda"),
                gradient_checkpointing=(self.device_type == "cuda"),
                memory_reg_weight=config.memory_reg_weight,
            )
        elif config.init_checkpoint:
            print(f"Loading model config from: {config.init_checkpoint}")
            init_ckpt = torch.load(config.init_checkpoint, map_location='cpu', weights_only=False)
            if 'config' in init_ckpt:
                ckpt_cfg = init_ckpt['config']
                if isinstance(ckpt_cfg, dict):
                    import dataclasses
                    valid_fields = {f.name for f in dataclasses.fields(OnyxConfig)}
                    filtered_cfg = {k: v for k, v in ckpt_cfg.items() if k in valid_fields}
                    model_config = OnyxConfig(**filtered_cfg)
                else:
                    model_config = ckpt_cfg
                # Override runtime settings
                model_config.use_flash_attention = (self.device_type == "cuda")
                model_config.gradient_checkpointing = (self.device_type == "cuda")
                model_config.memory_reg_weight = config.memory_reg_weight
                model_config.max_seq_len = config.max_seq_len
                model_config.train_seq_len = config.max_seq_len
            else:
                raise ValueError("init_checkpoint has no config - use --resume for full checkpoints")
        else:
            print("Creating Onyx model with default config...")
            model_config = OnyxConfig(
                use_flash_attention=(self.device_type == "cuda"),
                gradient_checkpointing=(self.device_type == "cuda"),
                memory_reg_weight=config.memory_reg_weight,
                max_seq_len=config.max_seq_len,
                train_seq_len=config.max_seq_len,
            )

        print(f"Model config: d_model={model_config.d_model}, n_layers={model_config.n_layers}, "
              f"n_heads={model_config.n_heads}")

        self.model = Onyx(model_config)

        # Load weights from init_checkpoint if provided
        if config.init_checkpoint:
            print(f"Loading model weights from: {config.init_checkpoint}")
            if 'model_state_dict' in init_ckpt:
                self.model.load_state_dict(init_ckpt['model_state_dict'])
            elif 'model' in init_ckpt:
                self.model.load_state_dict(init_ckpt['model'])
            else:
                self.model.load_state_dict(init_ckpt)
            print("Model weights loaded successfully")

        self.model = self.model.to(self.device)

        total_params = self.model.get_num_params()
        print(f"Model parameters: {total_params:,} ({total_params/1e6:.0f}M)")

        if config.use_torch_compile and self.device_type == "cuda" and hasattr(torch, 'compile'):
            print(f"Compiling model with mode: {config.compile_mode}")
            self.model = torch.compile(self.model, mode=config.compile_mode)

        param_groups = get_param_groups(
            self.model,
            weight_decay=config.weight_decay,
            memory_lr_scale=config.memory_lr_scale,
        )

        if config.use_m3_optimizer:
            print("Using M3 optimizer")
            self.optimizer = M3Optimizer(
                param_groups,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                beta_slow=config.m3_beta_slow,
                slow_freq=config.m3_slow_freq,
                slow_weight=config.m3_slow_weight,
            )
        else:
            print("Using AdamW optimizer")
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=config.weight_decay,
            )

        tokens_per_batch = config.batch_size * config.max_seq_len
        self.accumulation_steps = max(1, config.tokens_per_step // tokens_per_batch)
        print(f"Gradient accumulation: {self.accumulation_steps} steps")
        print(f"Effective batch: {self.accumulation_steps * config.batch_size} sequences")
        print(f"Tokens per step: {self.accumulation_steps * tokens_per_batch:,}")

        if config.max_steps:
            self.total_steps = config.max_steps
        elif config.train_tokens_target:
            self.total_steps = config.train_tokens_target // (self.accumulation_steps * tokens_per_batch)
        else:
            self.total_steps = None  # Will be set in train() after counting tokens
        if self.total_steps:
            print(f"Total training steps: {self.total_steps:,}")

        self.scaler = None
        if config.use_amp and self.device_type == "cuda" and config.amp_dtype == "float16":
            self.scaler = torch.amp.GradScaler('cuda')

        self.global_step = 0
        self.tokens_seen = 0
        self.best_loss = float('inf')

        if config.resume:
            self.load_checkpoint(config.resume)

        os.makedirs(config.save_dir, exist_ok=True)

        if config.wandb_project and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config={
                    **vars(config),
                    "total_params": total_params,
                    "device": str(self.device),
                }
            )

        self.setup_complete = True
        print("\nSetup complete! Ready to train.\n")

    def get_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            try:
                batch = next(self.data_iter)
            except StopIteration:
                return None

        result = {}
        for k, v in batch.items():
            if k == "cu_seqlens":
                # cu_seqlens needs special handling for batching
                # Each sample has its own cu_seqlens, we need to combine them
                # with proper offsets for the flattened batch
                result[k] = v  # Keep on CPU for now, handled in train_step
            elif k == "max_seqlen":
                result[k] = v  # Keep as-is
            else:
                result[k] = v.to(self.device)
        return result

    def train_step(self, memory_states: Optional[List[Dict[str, Any]]] = None):
        config = self.config
        self.model.train()

        if memory_states is None:
            memory_states = self.model.init_memory_states(
                config.batch_size,
                self.device,
                self.dtype if self.device_type == "cuda" else torch.float32,
            )

        self.optimizer.zero_grad()

        total_loss = 0.0
        total_mem_reg_loss = 0.0
        total_tokens = 0

        for micro_step in range(self.accumulation_steps):
            batch = self.get_batch()
            if batch is None:
                break

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            batch_tokens = input_ids.numel()
            total_tokens += batch_tokens

            # Process cu_seqlens for varlen attention
            cu_seqlens = None
            max_seqlen = None
            if "cu_seqlens" in batch and self.device_type == "cuda":
                # Combine cu_seqlens across batch with proper offsets
                # batch["cu_seqlens"] is [batch_size, varying_length] tensor
                B, S = input_ids.shape
                combined_cu = [0]
                batch_cu = batch["cu_seqlens"]  # List of tensors or stacked tensor
                batch_max_seqlen = batch.get("max_seqlen", None)

                offset = 0
                for b in range(B):
                    if isinstance(batch_cu, torch.Tensor):
                        sample_cu = batch_cu[b]
                    else:
                        sample_cu = batch_cu[b]
                    # Add all boundaries except the first (which is 0) with offset
                    for i in range(1, len(sample_cu)):
                        combined_cu.append(offset + sample_cu[i].item())
                    offset += S

                cu_seqlens = torch.tensor(combined_cu, dtype=torch.int32, device=self.device)
                # Get max sequence length across all documents in batch
                if batch_max_seqlen is not None:
                    if isinstance(batch_max_seqlen, torch.Tensor):
                        max_seqlen = batch_max_seqlen.max().item()
                    else:
                        max_seqlen = max(batch_max_seqlen)
                else:
                    max_seqlen = S

            memory_states = self.model.detach_memory_states(memory_states)

            if config.use_amp and self.device_type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        memory_states=memory_states,
                        update_memories=True,
                        return_memory_reg_loss=True,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    memory_states=memory_states,
                    update_memories=True,
                    return_memory_reg_loss=True,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

            memory_states = outputs["memory_states"]

            loss = outputs["loss"] / self.accumulation_steps
            mem_reg_loss = outputs.get("memory_reg_loss", 0.0)
            if isinstance(mem_reg_loss, torch.Tensor):
                mem_reg_loss = mem_reg_loss / self.accumulation_steps
            total_loss += loss.item() * self.accumulation_steps
            total_mem_reg_loss += mem_reg_loss.item() * self.accumulation_steps if isinstance(mem_reg_loss, torch.Tensor) else 0

            combined_loss = loss
            if isinstance(mem_reg_loss, torch.Tensor) and mem_reg_loss.item() > 0:
                combined_loss = loss + mem_reg_loss

            if self.scaler is not None:
                self.scaler.scale(combined_loss).backward()
            else:
                combined_loss.backward()

        if total_tokens == 0:
            return None, memory_states

        if config.gradient_clip > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.gradient_clip
            )
        else:
            grad_norm = 0.0

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        lr = get_lr(self.global_step, config, self.total_steps)
        set_lr(self.optimizer, lr, config)

        self.global_step += 1
        self.tokens_seen += total_tokens

        avg_loss = total_loss / self.accumulation_steps
        avg_mem_reg = total_mem_reg_loss / self.accumulation_steps if self.accumulation_steps > 0 else 0

        return {
            "loss": avg_loss,
            "mem_reg_loss": avg_mem_reg,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": lr,
            "tokens": total_tokens,
        }, memory_states

    def get_memory_stats(self, memory_states: List[Dict[str, Any]]) -> Dict[str, float]:
        if not memory_states:
            return {}

        k_norms = []
        v_norms = []

        for layer_mem in memory_states:
            if 'attention' in layer_mem:
                attn_mem = layer_mem['attention']
                if 'k' in attn_mem and torch.is_tensor(attn_mem['k']):
                    k_norms.append(attn_mem['k'].norm().item())
                if 'v' in attn_mem and torch.is_tensor(attn_mem['v']):
                    v_norms.append(attn_mem['v'].norm().item())

        stats = {}
        if k_norms:
            stats['memory/k_norm_mean'] = sum(k_norms) / len(k_norms)
            stats['memory/k_norm_max'] = max(k_norms)
        if v_norms:
            stats['memory/v_norm_mean'] = sum(v_norms) / len(v_norms)
            stats['memory/v_norm_max'] = max(v_norms)

        return stats

    def train(self):
        if not self.setup_complete:
            self.setup()

        config = self.config

        # Count actual tokens in dataset if train_tokens_target is None
        if config.train_tokens_target is None:
            print("Counting tokens in dataset (this may take a moment)...")
            dataset_tokens = 0
            for batch in self.dataset:
                dataset_tokens += batch['input_ids'].numel()
            # Reset dataset iterator
            self.dataset = StreamingPackedDataset(
                data_glob=config.data_glob,
                tokenizer=self.tokenizer,
                max_seq_len=config.max_seq_len,
                pack=config.pack_sequences,
                seed=config.seed,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=config.batch_size,
                num_workers=0,
                pin_memory=True,
            )
            self.data_iter = iter(self.dataloader)
            print(f"Dataset contains {dataset_tokens:,} tokens")
        else:
            dataset_tokens = config.train_tokens_target

        steps_per_epoch = max(1, dataset_tokens // config.tokens_per_step)
        total_steps = steps_per_epoch * config.num_epochs
        self.total_steps = total_steps  # Update instance variable

        print("=" * 70)
        print("Starting training")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Steps per epoch: {steps_per_epoch:,}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Dataset tokens: {dataset_tokens:,}")
        print("=" * 70)

        memory_states = None
        start_time = time.time()
        log_loss = 0.0
        log_tokens = 0
        log_steps = 0
        current_epoch = 0

        while self.global_step < total_steps:
            if self._interrupt_requested:
                print("\n[INTERRUPT] Saving checkpoint before exit...")
                self.save_checkpoint("interrupt")
                print("Checkpoint saved. Exiting.")
                break

            new_epoch = self.global_step // steps_per_epoch
            if new_epoch > current_epoch:
                current_epoch = new_epoch
                print(f"\n{'='*70}")
                print(f"Epoch {current_epoch} complete")
                print(f"{'='*70}\n")
                if config.save_every_epoch:
                    self.save_checkpoint(f"epoch_{current_epoch}")

            metrics, memory_states = self.train_step(memory_states)
            if metrics is None:
                continue

            log_loss += metrics["loss"]
            log_tokens += metrics["tokens"]
            log_steps += 1

            if self.global_step % config.log_every == 0:
                elapsed = time.time() - start_time
                avg_loss = log_loss / log_steps if log_steps > 0 else 0
                tokens_per_sec = log_tokens / elapsed if elapsed > 0 else 0
                epoch_progress = (self.global_step % steps_per_epoch) / steps_per_epoch * 100

                mem_stats = self.get_memory_stats(memory_states)

                print(f"Epoch {current_epoch+1}/{config.num_epochs} [{epoch_progress:5.1f}%] | "
                      f"Step {self.global_step:6d} | "
                      f"Loss {avg_loss:.4f} | "
                      f"LR {metrics['lr']:.2e} | "
                      f"Tok/s {tokens_per_sec:.0f}", flush=True)

                if config.wandb_project and WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/mem_reg_loss": metrics["mem_reg_loss"],
                        "train/lr": metrics["lr"],
                        "train/grad_norm": metrics["grad_norm"],
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/tokens_seen": self.tokens_seen,
                        "train/step": self.global_step,
                        "train/epoch": current_epoch + epoch_progress/100,
                        **mem_stats,
                    })

                log_loss = 0.0
                log_tokens = 0
                log_steps = 0
                start_time = time.time()

            if config.save_every_steps > 0 and self.global_step % config.save_every_steps == 0:
                self.save_checkpoint(f"step_{self.global_step}")

        if not self._interrupt_requested:
            self.save_checkpoint(f"epoch_{config.num_epochs}_final")
            print("\nTraining complete!")
            print(f"  Epochs: {config.num_epochs}")
            print(f"  Total steps: {self.global_step:,}")
            print(f"  Total tokens: {self.tokens_seen:,}")

    def save_checkpoint(self, name: str):
        config = self.config
        path = os.path.join(config.save_dir, f"checkpoint_{name}.pt")

        model = self.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_loss": self.best_loss,
            "config": vars(config),
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        self._cleanup_checkpoints()

    def load_checkpoint(self, path: str):
        print(f"Resuming from: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        model = self.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod

        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.tokens_seen = checkpoint["tokens_seen"]
        self.best_loss = checkpoint.get("best_loss", float('inf'))

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"  Resumed at step {self.global_step}, {self.tokens_seen:,} tokens seen")

    def _cleanup_checkpoints(self):
        config = self.config
        if config.keep_last_n <= 0:
            return

        checkpoints = sorted(Path(config.save_dir).glob("checkpoint_step_*.pt"))
        if len(checkpoints) > config.keep_last_n:
            for ckpt in checkpoints[:-config.keep_last_n]:
                ckpt.unlink()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Onyx Model")

    parser.add_argument("--data_glob", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Hermes-2-Pro-Llama-3-8B")
    parser.add_argument("--model_config", type=str, default=None,
                       help="Path to model config JSON (e.g., models/110m/config.json)")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--tokens_per_step", type=int, default=8192)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--train_tokens_target", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--memory_lr_scale", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=50)

    parser.add_argument("--use_adamw", action="store_true")

    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")

    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, default=None,
                       help="Initialize model weights from checkpoint (for grown models)")

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = TrainingConfig(
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        tokens_per_step=args.tokens_per_step,
        num_epochs=args.num_epochs,
        train_tokens_target=args.train_tokens_target,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        memory_lr_scale=args.memory_lr_scale,
        warmup_steps=args.warmup_steps,
        use_m3_optimizer=not args.use_adamw,
        use_torch_compile=args.compile,
        compile_mode=args.compile_mode,
        save_dir=args.save_dir,
        save_every_steps=args.save_every_steps,
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
        model_config_path=args.model_config,
        log_every=args.log_every,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    if args.data_glob:
        config.data_glob = args.data_glob

    if args.dry_run:
        config.max_steps = 3
        config.log_every = 1
        config.save_every_steps = 0
        config.use_torch_compile = False
        config.wandb_project = None
        print("\n" + "=" * 70)
        print("DRY RUN MODE - Will run 3 steps and exit")
        print("=" * 70 + "\n")

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
