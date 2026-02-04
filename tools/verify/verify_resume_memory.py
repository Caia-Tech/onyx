#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = "/Users/owner/Desktop/caiatech/models/onyx"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from onyx.model import OnyxConfig
from onyx.train import CMSFrequencyManager, TrainingConfig, Trainer


def _force_device(device: str) -> None:
    if device == "cpu":
        torch.cuda.is_available = lambda: False  # type: ignore[assignment]
        if hasattr(torch.backends, "mps"):
            torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
    elif device == "mps":
        torch.cuda.is_available = lambda: False  # type: ignore[assignment]
        if hasattr(torch.backends, "mps"):
            torch.backends.mps.is_available = lambda: True  # type: ignore[assignment]
    elif device == "cuda":
        torch.cuda.is_available = lambda: True  # type: ignore[assignment]


def _write_synth_data(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = [
        {"text": "hello world"},
        {"text": "goodbye world"},
        {"text": "lorem ipsum"},
        {"text": "quick brown fox"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def _write_model_config(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg = {
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "head_dim": 8,
        "d_ff": 64,
        "max_seq_len": 16,
        "train_seq_len": 16,
        "vocab_size": 128,
        "use_flash_attention": False,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "use_hope_attention": True,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)


class _StubTokenizer:
    def __init__(self, vocab_size: int = 128):
        self.eos_token_id = 1
        self.pad_token_id = 0
        self._vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = []
        for ch in text:
            ids.append((ord(ch) % (self._vocab_size - 2)) + 2)
        return ids

    def __len__(self) -> int:
        return self._vocab_size


class _StubAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _StubTokenizer()


def _memory_norms(memory_states: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not memory_states:
        return out
    layer0 = memory_states[0]
    attn = layer0.get("attention", {}) if isinstance(layer0, dict) else {}
    for key in ("k", "v"):
        if key in attn and torch.is_tensor(attn[key]):
            out[f"layer0_{key}"] = float(attn[key].float().norm().item())
    return out


def _print_table(rows: List[Dict[str, Any]]) -> None:
    headers = ["run_mode", "step", "layer0_k", "layer0_v"]
    print("\t".join(headers))
    for r in rows:
        print("\t".join(str(r.get(h, "")) for h in headers))


def _pick_data_dir() -> str:
    preferred = os.path.join(REPO_ROOT, "tools", "verify", "_data")
    try:
        os.makedirs(preferred, exist_ok=True)
        return preferred
    except Exception:
        fallback = os.path.join(REPO_ROOT, "onyx", ".verify_data")
        os.makedirs(fallback, exist_ok=True)
        return fallback


def _run_train_and_resume(device: str) -> None:
    _force_device(device)

    import onyx.train as onyx_train
    onyx_train.AutoTokenizer = _StubAutoTokenizer

    base_dir = _pick_data_dir()
    data_path = os.path.join(base_dir, "synth.jsonl")
    config_path = os.path.join(base_dir, "model_config.json")
    ckpt_dir = os.path.join(base_dir, "ckpt")

    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    _write_synth_data(data_path)
    _write_model_config(config_path)

    cfg = TrainingConfig(
        data_glob=data_path,
        tokenizer_name="stub",
        model_config_path=config_path,
        batch_size=2,
        max_seq_len=16,
        tokens_per_step=32,
        pack_sequences=True,
        drop_remainder=False,
        shuffle_buffer_docs=0,
        num_epochs=1,
        max_steps=3,
        learning_rate=1e-4,
        min_lr=1e-4,
        warmup_steps=1,
        weight_decay=0.0,
        use_m3_optimizer=False,
        use_amp=False,
        gradient_checkpointing=False,
        save_dir=ckpt_dir,
        save_every_steps=0,
        keep_last_n=1,
        log_every=1,
        monitor_diversity=False,
        monitor_memory_states=False,
        mem_report_every=0,
        mps_empty_cache_every=0,
        gc_collect_every=0,
        dataset_state_mode="light",
        quiet=True,
        persist_memory_across_steps=True,
        reset_memory_every_steps=0,
        reset_memory_on_epoch=False,
    )

    trainer = Trainer(cfg)
    trainer.train()

    rows: List[Dict[str, Any]] = []
    if trainer.last_memory_states is not None:
        norms = _memory_norms(trainer.last_memory_states)
        rows.append({"run_mode": "train", "step": trainer.global_step, **norms})

    # checkpoint should exist
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_final_step_{trainer.global_step}.pt")
    assert os.path.exists(ckpt_path), f"missing checkpoint: {ckpt_path}"

    # resume for one step
    cfg_resume = TrainingConfig(**{**vars(cfg), "resume": ckpt_path, "max_steps": trainer.global_step + 1})
    trainer_resume = Trainer(cfg_resume)
    trainer_resume.train()

    assert trainer_resume.global_step == trainer.global_step + 1
    assert trainer_resume.last_memory_states is not None

    # dtype check (fp32 since AMP off)
    layer0 = trainer_resume.last_memory_states[0]
    attn = layer0.get("attention", {}) if isinstance(layer0, dict) else {}
    for key in ("k", "v"):
        if key in attn and torch.is_tensor(attn[key]):
            assert attn[key].dtype == torch.float32

    norms_resume = _memory_norms(trainer_resume.last_memory_states)
    rows.append({"run_mode": "resume", "step": trainer_resume.global_step, **norms_resume})

    # reset behavior: persist_memory_across_steps=False
    cfg_reset = TrainingConfig(**{**vars(cfg), "persist_memory_across_steps": False, "max_steps": 2})
    trainer_reset = Trainer(cfg_reset)
    trainer_reset.train()
    norms_reset = _memory_norms(trainer_reset.last_memory_states) if trainer_reset.last_memory_states else {}
    rows.append({"run_mode": "reset", "step": trainer_reset.global_step, **norms_reset})

    _print_table(rows)


def _cms_schedule() -> None:
    class DummyCMS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = torch.nn.Module()
            self.ffn.level_ffns = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False) for _ in range(3)])

    cfg = OnyxConfig(
        use_cms_ffn=True,
        cms_update_every_base_steps=2,
        cms_update_every_multiplier=2,
    )
    model = DummyCMS()
    mgr = CMSFrequencyManager(model, cfg)

    schedule = {0: [], 1: [], 2: []}
    for step in range(1, 33):
        for p in model.parameters():
            p.grad = torch.ones_like(p.data)
        mgr.mask_gradients(global_step=step)
        for level in range(3):
            grad = model.ffn.level_ffns[level].weight.grad
            if grad is not None:
                schedule[level].append(step)

    print("CMS update steps (1..32):")
    for level in range(3):
        print(f"  level {level}: {schedule[level]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    _run_train_and_resume(args.device)
    _cms_schedule()


if __name__ == "__main__":
    main()
