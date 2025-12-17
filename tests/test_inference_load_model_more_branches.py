import json
from pathlib import Path

import torch

import onyx_inference
from onyx_model import Onyx, OnyxConfig


def test_load_model_reads_checkpoint_referenced_model_config_path(tmp_path: Path):
    # Write config JSON next to checkpoint and reference it via ckpt["config"]["model_config_path"].
    cfg = {
        "architecture": {
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 1,
            "n_kv_heads": 1,
            "head_dim": 32,
            "d_ff": 64,
            "vocab_size": 19,
            "max_seq_len": 16,
            "train_seq_len": 16,
        }
    }
    cfg_path = tmp_path / "model.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    m = Onyx(OnyxConfig(**cfg["architecture"]))
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model_state_dict": m.state_dict(),
            "config": {"model_config_path": cfg_path.name},
        },
        ckpt_path,
    )

    loaded, loaded_cfg = onyx_inference.load_model(
        str(ckpt_path),
        tokenizer=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        model_config_path=None,
    )
    assert loaded_cfg.vocab_size == 19
    assert loaded.embed.weight.shape[0] == 19


def test_load_model_falls_back_to_config_dict_in_checkpoint(tmp_path: Path):
    cfg = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=32,
        d_ff=64,
        vocab_size=23,
        max_seq_len=16,
        train_seq_len=16,
    )
    m = Onyx(cfg)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": m.state_dict(), "config": cfg.__dict__}, ckpt_path)

    loaded, loaded_cfg = onyx_inference.load_model(
        str(ckpt_path),
        tokenizer=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        model_config_path=None,
    )
    assert loaded_cfg.vocab_size == 23
    assert loaded.embed.weight.shape[0] == 23
