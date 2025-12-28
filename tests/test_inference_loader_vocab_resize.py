from pathlib import Path

import torch

from onyx.inference import load_model
from onyx.model import Onyx, OnyxConfig


def test_load_model_pads_vocab_when_ckpt_smaller(tmp_path: Path):
    cfg = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=32,
        d_ff=64,
        vocab_size=101,
        max_seq_len=16,
        train_seq_len=16,
    )
    model = Onyx(cfg)
    sd = model.state_dict()

    # Simulate checkpoint with smaller vocab (100 instead of 101)
    sd_small = dict(sd)
    sd_small["embed.weight"] = sd_small["embed.weight"][:100].clone()
    if "lm_head.weight" in sd_small:
        sd_small["lm_head.weight"] = sd_small["lm_head.weight"][:100].clone()

    ckpt = {"model_state_dict": sd_small, "config": cfg.__dict__}
    p = tmp_path / "ckpt.pt"
    torch.save(ckpt, str(p))

    loaded, loaded_cfg = load_model(str(p), device=torch.device("cpu"), dtype=torch.float32)
    assert loaded_cfg.vocab_size == 101
    assert loaded.embed.weight.shape[0] == 101
