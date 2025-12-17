import glob
import os
from pathlib import Path

import torch

from onyx_model import Onyx, OnyxConfig
from onyx_train import Trainer, TrainingConfig


def test_checkpoint_cleanup_keeps_last_n(tmp_path: Path):
    cfg = TrainingConfig(
        data_glob="unused.jsonl",
        save_dir=str(tmp_path),
        keep_last_n=5,
    )
    t = Trainer(cfg)

    model = Onyx(
        OnyxConfig(
            d_model=32,
            n_layers=1,
            n_heads=1,
            n_kv_heads=1,
            head_dim=32,
            d_ff=64,
            vocab_size=50,
            max_seq_len=16,
            train_seq_len=16,
        )
    )
    t.model = model
    t.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t.scaler = None
    t.tokens_seen = 0
    t.best_loss = 1e9
    t.current_epoch = 0

    for step in range(1, 8):  # 7 checkpoints
        t.global_step = step * 1000
        t.save_checkpoint(f"step_{t.global_step}")

    ckpts = sorted(glob.glob(os.path.join(str(tmp_path), "checkpoint_step_*.pt")))
    assert len(ckpts) == 5
    names = [Path(x).name for x in ckpts]
    assert any("step_7000" in n for n in names)
    assert not any("step_1000" in n for n in names)


def test_resume_restores_global_step(tmp_path: Path):
    cfg = TrainingConfig(data_glob="unused.jsonl", save_dir=str(tmp_path))
    t = Trainer(cfg)

    model = Onyx(
        OnyxConfig(
            d_model=32,
            n_layers=1,
            n_heads=1,
            n_kv_heads=1,
            head_dim=32,
            d_ff=64,
            vocab_size=50,
            max_seq_len=16,
            train_seq_len=16,
        )
    )
    t.model = model
    t.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t.scaler = None
    t.tokens_seen = 123
    t.best_loss = 9.0
    t.current_epoch = 0
    t.global_step = 4242
    t.save_checkpoint("step_4242")

    path = os.path.join(str(tmp_path), "checkpoint_step_4242.pt")

    t2 = Trainer(cfg)
    t2.model = Onyx(model.config)
    t2.optimizer = torch.optim.AdamW(t2.model.parameters(), lr=1e-3)
    t2.scaler = None
    t2.load_checkpoint(path)

    assert t2.global_step == 4242
    assert t2.tokens_seen == 123
