import torch


def test_kv_cache_next_token_matches_full(tiny_model, tiny_config):
    tiny_model.eval()
    B, S = 1, 12
    x = torch.randint(0, tiny_config.vocab_size, (B, S))

    mem = tiny_model.init_memory_states(B, device=x.device, dtype=torch.float32)

    # Build cache for prefix, then feed last token via cache.
    out_prefix = tiny_model(
        input_ids=x[:, :-1],
        memory_states=mem,
        update_memories=False,
        inference_mode=True,
        position_offset=0,
    )
    kv_cache = out_prefix["kv_cache"]
    mem2 = out_prefix["memory_states"]

    out_cached = tiny_model(
        input_ids=x[:, -1:],
        memory_states=mem2,
        update_memories=False,
        inference_mode=True,
        kv_cache=kv_cache,
        position_offset=S - 1,
    )
    logits_cached = out_cached["logits"][:, -1, :]

    # Full recompute for comparison.
    out_full = tiny_model(
        input_ids=x,
        memory_states=mem,
        update_memories=False,
        inference_mode=True,
        position_offset=0,
    )
    logits_last_full = out_full["logits"][:, -1, :]

    assert torch.allclose(logits_last_full, logits_cached, atol=1e-4, rtol=1e-4)
