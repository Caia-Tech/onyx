import torch

from onyx.model import Onyx, OnyxConfig


def test_standard_attention_forward_runs():
    cfg = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=32,
        d_ff=64,
        vocab_size=50,
        max_seq_len=32,
        train_seq_len=32,
        use_flash_attention=False,
        gradient_checkpointing=False,
        use_hope_attention=False,
    )
    m = Onyx(cfg).cpu().eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    out = m(input_ids=x, memory_states=m.init_memory_states(2, x.device, torch.float32), inference_mode=True)
    assert out["logits"].shape == (2, 8, cfg.vocab_size)


def test_tie_embeddings_false_creates_distinct_weights():
    cfg = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=32,
        d_ff=64,
        vocab_size=50,
        max_seq_len=16,
        train_seq_len=16,
        tie_embeddings=False,
        use_flash_attention=False,
        gradient_checkpointing=False,
    )
    m = Onyx(cfg).cpu().eval()
    assert m.embed.weight is not m.lm_head.weight


def test_gradient_checkpointing_backward_smoke():
    cfg = OnyxConfig(
        d_model=32,
        n_layers=2,
        n_heads=1,
        n_kv_heads=1,
        head_dim=32,
        d_ff=64,
        vocab_size=50,
        max_seq_len=32,
        train_seq_len=32,
        use_flash_attention=False,
        gradient_checkpointing=True,
    )
    m = Onyx(cfg).cpu().train()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = torch.randint(0, cfg.vocab_size, (1, 8))
    mem = m.init_memory_states(1, x.device, torch.float32)
    out = m(input_ids=x, labels=y, memory_states=mem, update_memories=True)
    assert out["loss"] is not None
    out["loss"].backward()
