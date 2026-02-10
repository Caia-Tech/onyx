import torch

from onyx.experimental import OnyxMHC
from onyx.model import Onyx, OnyxConfig


def _tiny_rmt_cfg(**overrides) -> OnyxConfig:
    base = dict(
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        d_ff=64,
        vocab_size=128,
        max_seq_len=64,
        train_seq_len=64,
        use_flash_attention=False,
        gradient_checkpointing=False,
        dropout=0.0,
        attention_dropout=0.0,
        use_cms_ffn=False,
        use_hope_attention=False,
        use_rmt=True,
        rmt_dk=16,
        rmt_dv=16,
        rmt_heads=2,
        rmt_ff_multiplier=2.0,
        rmt_init="normal",
        rmt_readwrite_dtype="float32",
    )
    base.update(overrides)
    return OnyxConfig(**base)


def _assert_finite_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_rmt_forward_shapes():
    torch.manual_seed(0)
    cfg = _tiny_rmt_cfg()
    model = Onyx(cfg).cpu().train()
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(input_ids=x, labels=y)
    assert out["logits"].shape == (2, 32, cfg.vocab_size)
    assert out["loss"] is not None and out["loss"].ndim == 0
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["loss"]).all()


def test_rmt_backward_finite_grads():
    torch.manual_seed(0)
    cfg = _tiny_rmt_cfg()
    model = Onyx(cfg).cpu().train()
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(input_ids=x, labels=y)
    out["loss"].backward()
    _assert_finite_grads(model)
    assert model.embed.weight.grad is not None
    assert torch.isfinite(model.embed.weight.grad).all()

    named = dict(model.named_parameters())
    assert "layers.0.attention.r_q" in named
    assert "layers.0.attention.w_o" in named
    assert named["layers.0.attention.r_q"].grad is not None
    assert named["layers.0.attention.w_o"].grad is not None
    assert torch.isfinite(named["layers.0.attention.r_q"].grad).all()
    assert torch.isfinite(named["layers.0.attention.w_o"].grad).all()


def test_rmt_with_hope_attention():
    torch.manual_seed(0)
    cfg = _tiny_rmt_cfg(use_hope_attention=True, self_referential_keys=True, self_referential_values=True)
    model = Onyx(cfg).cpu().train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=x, labels=y, update_memories=True)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]).all()
    _assert_finite_grads(model)


def test_rmt_with_mhc_sinkhorn():
    torch.manual_seed(0)
    cfg = _tiny_rmt_cfg()
    model = OnyxMHC(cfg, mhc_n=2, mhc_mode="mhc", mhc_sinkhorn=True, mhc_sinkhorn_iters=3).cpu().train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=x, labels=y)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]).all()
    _assert_finite_grads(model)

    mix = model.mixers[0]._mix_matrix()
    assert mix.dtype == torch.float32
    assert torch.allclose(mix.sum(dim=-1), torch.ones(mix.size(0)), atol=1e-2, rtol=1e-2)
    assert torch.allclose(mix.sum(dim=-2), torch.ones(mix.size(1)), atol=1e-2, rtol=1e-2)


def test_rmt_with_memory_updates():
    torch.manual_seed(0)
    cfg = _tiny_rmt_cfg(use_hope_attention=True, self_referential_keys=True, self_referential_values=True)
    model = Onyx(cfg).cpu().train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    mem0 = model.init_memory_states(2, x.device, torch.float32)
    before = model._memory_update_count
    out = model(input_ids=x, labels=y, memory_states=mem0, update_memories=True)
    out["loss"].backward()
    after = model._memory_update_count
    assert after > before
    assert out["memory_states"] is not None
    _assert_finite_grads(model)


def test_rmt_full_stack():
    torch.manual_seed(0)
    cfg = _tiny_rmt_cfg(use_hope_attention=True, self_referential_keys=True, self_referential_values=True)
    model = OnyxMHC(cfg, mhc_n=2, mhc_mode="mhc", mhc_sinkhorn=True, mhc_sinkhorn_iters=3).cpu().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(2):
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        y = torch.randint(0, cfg.vocab_size, (2, 32))
        out = model(input_ids=x, labels=y, update_memories=True)
        loss = out["loss"]
        assert loss is not None and torch.isfinite(loss).all()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        _assert_finite_grads(model)
        opt.step()


def test_training_step_integration():
    torch.manual_seed(0)
    cfg = _tiny_rmt_cfg()
    model = Onyx(cfg).cpu().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))

    tracked = [p for p in model.parameters() if p.requires_grad][:8]
    before = [p.detach().clone() for p in tracked]

    losses = []
    for _ in range(2):
        out = model(input_ids=x, labels=y)
        loss = out["loss"]
        assert loss is not None and torch.isfinite(loss).all()
        losses.append(float(loss.detach().item()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        _assert_finite_grads(model)
        opt.step()

    total_delta = 0.0
    for p0, p1 in zip(before, tracked):
        total_delta += float((p1.detach() - p0).norm().item())
    assert total_delta > 0.0
    assert all(torch.isfinite(torch.tensor(losses)))


def test_seed_repro():
    torch.manual_seed(123)
    cfg = _tiny_rmt_cfg(dropout=0.0, attention_dropout=0.0, use_hope_attention=False)
    model = Onyx(cfg).cpu().eval()
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        out1 = model(input_ids=x, update_memories=False)["logits"]
        out2 = model(input_ids=x, update_memories=False)["logits"]
    torch.testing.assert_close(out1, out2, rtol=0.0, atol=0.0)


def test_rmt_kv_cache_next_token_matches_full():
    torch.manual_seed(0)
    cfg = _tiny_rmt_cfg(use_hope_attention=False)
    model = Onyx(cfg).cpu().eval()
    B, S = 1, 12
    x = torch.randint(0, cfg.vocab_size, (B, S))
    mem = model.init_memory_states(B, device=x.device, dtype=torch.float32)

    out_prefix = model(
        input_ids=x[:, :-1],
        memory_states=mem,
        update_memories=False,
        inference_mode=True,
        position_offset=0,
    )
    kv_cache = out_prefix["kv_cache"]
    mem2 = out_prefix["memory_states"]

    out_cached = model(
        input_ids=x[:, -1:],
        memory_states=mem2,
        update_memories=False,
        inference_mode=True,
        kv_cache=kv_cache,
        position_offset=S - 1,
    )
    logits_cached = out_cached["logits"][:, -1, :]

    out_full = model(
        input_ids=x,
        memory_states=mem,
        update_memories=False,
        inference_mode=True,
        position_offset=0,
    )
    logits_last_full = out_full["logits"][:, -1, :]
    assert torch.allclose(logits_last_full, logits_cached, atol=1e-4, rtol=1e-4)
