import torch


def test_forward_shapes_and_loss(tiny_model, tiny_config):
    B, S = 2, 16
    x = torch.randint(0, tiny_config.vocab_size, (B, S))
    y = torch.randint(0, tiny_config.vocab_size, (B, S))

    mem = tiny_model.init_memory_states(B, device=x.device, dtype=torch.float32)
    out = tiny_model(
        input_ids=x,
        labels=y,
        memory_states=mem,
        update_memories=True,
        return_memory_reg_loss=True,
    )

    assert out["logits"].shape == (B, S, tiny_config.vocab_size)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert "memory_states" in out
