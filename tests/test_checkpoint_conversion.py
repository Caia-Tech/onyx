import json
from pathlib import Path

import torch

import convert_checkpoint
from onyx_model import Onyx, OnyxConfig


def _write_config(path: Path, arch: dict) -> None:
    path.write_text(json.dumps({"architecture": arch}), encoding="utf-8")


def _fill_pattern(sd: dict, key: str) -> None:
    t = sd[key]
    sd[key] = torch.arange(t.numel(), dtype=t.dtype).reshape(t.shape)


def test_convert_checkpoint_resizes_and_copies_last_layer(tmp_path: Path):
    old_arch = {
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 1,
        "head_dim": 4,
        "d_ff": 16,
        "vocab_size": 9,
        "max_seq_len": 16,
        "train_seq_len": 16,
        "use_hope_attention": True,
        "self_referential_keys": True,
        "self_referential_values": True,
        "generate_own_values": True,
        "use_cms_ffn": False,
        "use_flash_attention": False,
        "gradient_checkpointing": False,
    }
    new_arch = {
        "d_model": 10,
        "n_layers": 2,
        "n_heads": 3,
        "n_kv_heads": 2,
        "head_dim": 4,
        "d_ff": 20,
        "vocab_size": 11,
        "max_seq_len": 16,
        "train_seq_len": 16,
        "use_hope_attention": True,
        "self_referential_keys": True,
        "self_referential_values": True,
        "generate_own_values": True,
        "use_cms_ffn": False,
        "use_flash_attention": False,
        "gradient_checkpointing": False,
    }

    old_cfg = OnyxConfig(**old_arch)
    old_model = Onyx(old_cfg)
    sd = old_model.state_dict()

    _fill_pattern(sd, "layers.0.attention.q_proj.weight")
    _fill_pattern(sd, "layers.0.attention.k_memory.memory.M_init")
    _fill_pattern(sd, "layers.0.attention.k_memory.value_gen.0.weight")
    _fill_pattern(sd, "layers.0.ffn.w1.weight")
    _fill_pattern(sd, "embed.weight")

    ckpt_path = tmp_path / "old.pt"
    torch.save({"model_state_dict": sd}, ckpt_path)

    old_cfg_path = tmp_path / "old.json"
    new_cfg_path = tmp_path / "new.json"
    _write_config(old_cfg_path, old_arch)
    _write_config(new_cfg_path, new_arch)

    out_path = tmp_path / "converted.pt"
    convert_checkpoint.convert_checkpoint(
        input_ckpt=str(ckpt_path),
        output_ckpt=str(out_path),
        input_config=str(old_cfg_path),
        target_config=str(new_cfg_path),
        init_strategy="copy_last",
    )

    out = torch.load(out_path, map_location="cpu", weights_only=False)
    new_sd = out["model_state_dict"]
    new_cfg = OnyxConfig(**new_arch)
    Onyx(new_cfg).load_state_dict(new_sd, strict=True)

    old_q = sd["layers.0.attention.q_proj.weight"]
    new_q = new_sd["layers.0.attention.q_proj.weight"]
    old_q_view = old_q.view(old_arch["n_heads"], old_arch["head_dim"], old_arch["d_model"])
    new_q_view = new_q.view(new_arch["n_heads"], new_arch["head_dim"], new_arch["d_model"])
    assert torch.allclose(new_q_view[:2, :4, :8], old_q_view)

    new_q1 = new_sd["layers.1.attention.q_proj.weight"]
    new_q1_view = new_q1.view(new_arch["n_heads"], new_arch["head_dim"], new_arch["d_model"])
    assert torch.allclose(new_q1_view[:2, :4, :8], old_q_view)

    old_m = sd["layers.0.attention.k_memory.memory.M_init"]
    new_m = new_sd["layers.0.attention.k_memory.memory.M_init"]
    old_m_view = old_m.view(old_arch["n_kv_heads"], old_arch["head_dim"], old_arch["d_model"])
    new_m_view = new_m.view(new_arch["n_kv_heads"], new_arch["head_dim"], new_arch["d_model"])
    assert torch.allclose(new_m_view[:1, :4, :8], old_m_view)

    old_vg = sd["layers.0.attention.k_memory.value_gen.0.weight"]
    new_vg = new_sd["layers.0.attention.k_memory.value_gen.0.weight"]
    old_vg_view = old_vg.view(old_arch["n_kv_heads"], old_arch["head_dim"], old_arch["n_kv_heads"], old_arch["head_dim"])
    new_vg_view = new_vg.view(new_arch["n_kv_heads"], new_arch["head_dim"], new_arch["n_kv_heads"], new_arch["head_dim"])
    assert torch.allclose(new_vg_view[:1, :4, :1, :4], old_vg_view)

    old_w1 = sd["layers.0.ffn.w1.weight"]
    new_w1 = new_sd["layers.0.ffn.w1.weight"]
    assert torch.allclose(new_w1[: old_w1.shape[0], : old_w1.shape[1]], old_w1)

    old_embed = sd["embed.weight"]
    new_embed = new_sd["embed.weight"]
    assert torch.allclose(new_embed[: old_embed.shape[0], : old_embed.shape[1]], old_embed)


def test_convert_checkpoint_shrinks_heads_and_ffn(tmp_path: Path):
    old_arch = {
        "d_model": 12,
        "n_layers": 1,
        "n_heads": 3,
        "n_kv_heads": 2,
        "head_dim": 4,
        "d_ff": 24,
        "vocab_size": 13,
        "max_seq_len": 8,
        "train_seq_len": 8,
        "use_hope_attention": False,
        "use_cms_ffn": False,
        "use_flash_attention": False,
        "gradient_checkpointing": False,
    }
    new_arch = {
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 1,
        "head_dim": 4,
        "d_ff": 12,
        "vocab_size": 13,
        "max_seq_len": 8,
        "train_seq_len": 8,
        "use_hope_attention": False,
        "use_cms_ffn": False,
        "use_flash_attention": False,
        "gradient_checkpointing": False,
    }

    old_cfg = OnyxConfig(**old_arch)
    old_model = Onyx(old_cfg)
    sd = old_model.state_dict()

    _fill_pattern(sd, "layers.0.attention.q_proj.weight")
    _fill_pattern(sd, "layers.0.ffn.w1.weight")

    ckpt_path = tmp_path / "old.pt"
    torch.save({"model_state_dict": sd}, ckpt_path)

    old_cfg_path = tmp_path / "old.json"
    new_cfg_path = tmp_path / "new.json"
    _write_config(old_cfg_path, old_arch)
    _write_config(new_cfg_path, new_arch)

    out_path = tmp_path / "converted.pt"
    convert_checkpoint.convert_checkpoint(
        input_ckpt=str(ckpt_path),
        output_ckpt=str(out_path),
        input_config=str(old_cfg_path),
        target_config=str(new_cfg_path),
        init_strategy="random",
    )

    out = torch.load(out_path, map_location="cpu", weights_only=False)
    new_sd = out["model_state_dict"]
    new_cfg = OnyxConfig(**new_arch)
    Onyx(new_cfg).load_state_dict(new_sd, strict=True)

    old_q = sd["layers.0.attention.q_proj.weight"]
    new_q = new_sd["layers.0.attention.q_proj.weight"]
    old_q_view = old_q.view(old_arch["n_heads"], old_arch["head_dim"], old_arch["d_model"])
    new_q_view = new_q.view(new_arch["n_heads"], new_arch["head_dim"], new_arch["d_model"])
    assert torch.allclose(new_q_view, old_q_view[:2, :4, :8])

    old_w1 = sd["layers.0.ffn.w1.weight"]
    new_w1 = new_sd["layers.0.ffn.w1.weight"]
    assert torch.allclose(new_w1, old_w1[: new_w1.shape[0], : new_w1.shape[1]])
