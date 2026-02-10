import time

import pytest
import torch

from onyx.experimental import OnyxMHC
from onyx.experimental.mhc import sinkhorn_project, MHCMixer, MHCLiteMixer
from onyx.model import Onyx, OnyxConfig, M3Optimizer, get_param_groups


def _tiny_config() -> OnyxConfig:
    return OnyxConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        d_ff=64,
        vocab_size=64,
        max_seq_len=32,
        train_seq_len=32,
        use_flash_attention=False,
        gradient_checkpointing=False,
        dropout=0.0,
        attention_dropout=0.0,
        use_cms_ffn=False,
    )


def _run_train_steps(model: torch.nn.Module, device: torch.device, steps: int = 3) -> None:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    vocab_size = model.config.vocab_size
    batch_size = 2
    seq_len = 16

    for _ in range(steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        assert loss is not None
        assert torch.isfinite(loss).all()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def test_sinkhorn_doubly_stochastic():
    torch.manual_seed(0)
    mat = torch.rand(4, 4)
    proj = sinkhorn_project(mat, iters=20, eps=1e-6)
    row_sums = proj.sum(dim=-1)
    col_sums = proj.sum(dim=-2)
    ones = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, ones, atol=1e-2, rtol=1e-2)
    assert torch.allclose(col_sums, ones, atol=1e-2, rtol=1e-2)


def test_mhc_shapes():
    torch.manual_seed(0)
    config = _tiny_config()
    model = OnyxMHC(config, mhc_n=2, mhc_mode="mhc", mhc_sinkhorn=True, mhc_sinkhorn_iters=5)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))
    out = model(input_ids, labels=labels)
    assert out["logits"].shape == (2, 8, config.vocab_size)
    assert out["loss"] is not None
    assert out["loss"].ndim == 0


def test_mhc_lite_shapes():
    torch.manual_seed(0)
    config = _tiny_config()
    model = OnyxMHC(config, mhc_n=2, mhc_mode="mhc_lite")
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))
    out = model(input_ids, labels=labels)
    assert out["logits"].shape == (2, 8, config.vocab_size)
    assert out["loss"] is not None
    assert out["loss"].ndim == 0


def test_backward_no_nan():
    torch.manual_seed(0)
    config = _tiny_config()
    model = OnyxMHC(config, mhc_n=2)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))
    out = model(input_ids, labels=labels)
    loss = out["loss"]
    assert loss is not None
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_identity_equivalence_small():
    torch.manual_seed(0)
    config = _tiny_config()
    base = Onyx(config).eval()
    mhc = OnyxMHC(config, mhc_n=1, mhc_mode="mhc", mhc_sinkhorn=True).eval()

    res = mhc.load_state_dict(base.state_dict(), strict=False)
    assert res.unexpected_keys == []
    assert all(k.startswith("mixers.") for k in res.missing_keys)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    with torch.no_grad():
        base_out = base(input_ids)["logits"]
        mhc_out = mhc(input_ids)["logits"]
    torch.testing.assert_close(base_out, mhc_out, rtol=1e-5, atol=1e-5)


def test_smoke_train_3_steps_cpu():
    torch.manual_seed(0)
    config = _tiny_config()
    model = OnyxMHC(config, mhc_n=2)
    _run_train_steps(model, torch.device("cpu"), steps=3)


def test_mhc_mixer_no_nan_and_nonnegative():
    torch.manual_seed(0)
    mixer = MHCMixer(n_streams=2, mode="mhc", use_sinkhorn=True, sinkhorn_iters=10)
    opt = torch.optim.AdamW(mixer.parameters(), lr=1e-3)

    for _ in range(5):
        streams = torch.randn(2, 4, 2, 8)
        out = mixer(streams)
        loss = out.float().pow(2).mean()
        assert torch.isfinite(loss).all()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in mixer.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

        p_used = mixer._mix_matrix()
        assert torch.isfinite(p_used).all()
        assert p_used.min().item() >= -1e-8
        row_sum = p_used.sum(dim=-1)
        col_sum = p_used.sum(dim=-2)
        ones_row = torch.ones_like(row_sum)
        ones_col = torch.ones_like(col_sum)
        assert torch.allclose(row_sum, ones_row, atol=1e-3, rtol=1e-3)
        assert torch.allclose(col_sum, ones_col, atol=1e-3, rtol=1e-3)

        opt.step()


def test_mhc_lite_mixer_doubly_stochastic():
    torch.manual_seed(0)
    mixer = MHCLiteMixer(n_streams=3)
    streams = torch.randn(2, 4, 3, 8)
    out = mixer(streams)
    assert torch.isfinite(out).all()
    p_used = mixer._mix_matrix()
    assert torch.isfinite(p_used).all()
    assert p_used.min().item() >= -1e-8
    row_sum = p_used.sum(dim=-1)
    col_sum = p_used.sum(dim=-2)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4, rtol=1e-4)
    assert torch.allclose(col_sum, torch.ones_like(col_sum), atol=1e-4, rtol=1e-4)


def test_onyxmhc_no_nan_smoke_cpu():
    torch.manual_seed(0)
    config = _tiny_config()
    model = OnyxMHC(config, mhc_n=2, mhc_mode="mhc", mhc_sinkhorn=True, mhc_sinkhorn_iters=5)
    param_groups = get_param_groups(model, weight_decay=0.0, memory_lr_scale=0.1)
    opt = M3Optimizer(param_groups, lr=1e-3, weight_decay=0.0)
    vocab_size = config.vocab_size

    for _ in range(10):
        input_ids = torch.randint(0, vocab_size, (2, 8))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        assert loss is not None
        assert torch.isfinite(loss).all()
        assert torch.isfinite(out["logits"]).all()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
        for mixer in model.mixers:
            p_used = mixer._mix_matrix()
            assert torch.isfinite(p_used).all()
            assert p_used.min().item() >= -1e-8
            row_sum = p_used.sum(dim=-1)
            col_sum = p_used.sum(dim=-2)
            assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3, rtol=1e-3)
            assert torch.allclose(col_sum, torch.ones_like(col_sum), atol=1e-3, rtol=1e-3)

        for pg in opt.param_groups:
            pg["lr"] = 1e-3 * pg.get("lr_scale", 1.0)
        opt.step()


def test_onyxmhc_m3_no_nan_and_vector_mixer_mode():
    torch.manual_seed(0)
    config = _tiny_config()
    model = OnyxMHC(config, mhc_n=2, mhc_mode="mhc", mhc_sinkhorn=True, mhc_sinkhorn_iters=5)
    param_groups = get_param_groups(model, weight_decay=0.0, memory_lr_scale=0.1)
    mixer_groups = [g for g in param_groups if g.get("m3_mode") == "vector"]
    assert mixer_groups
    opt = M3Optimizer(param_groups, lr=1e-3, weight_decay=0.0)
    vocab_size = config.vocab_size

    for _ in range(20):
        input_ids = torch.randint(0, vocab_size, (2, 8))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        assert loss is not None
        assert torch.isfinite(loss).all()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
            if "mixers" in name:
                assert torch.isfinite(p).all()

        for pg in opt.param_groups:
            pg["lr"] = 1e-3 * pg.get("lr_scale", 1.0)
        opt.step()

    for name, p in model.named_parameters():
        if "mixers" in name:
            state = opt.state.get(p, {})
            assert "slow_memory" not in state


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
def test_smoke_train_3_steps_mps_fp32():
    torch.manual_seed(0)
    device = torch.device("mps")
    config = _tiny_config()
    model = OnyxMHC(config, mhc_n=2).to(device)
    _run_train_steps(model, device, steps=3)


def _run_perf(model: torch.nn.Module, device: torch.device, steps: int = 20) -> float:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    vocab_size = model.config.vocab_size
    batch_size = 2
    seq_len = 16
    tokens = steps * batch_size * seq_len

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

    start = time.time()
    for _ in range(steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elapsed = time.time() - start

    tokps = tokens / max(elapsed, 1e-9)
    return tokps


def test_perf_memory_smoke():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    config = _tiny_config()
    base = Onyx(config).to(device)
    mhc = OnyxMHC(config, mhc_n=2, mhc_mode="mhc", mhc_sinkhorn_iters=5).to(device)

    base_tokps = _run_perf(base, device)
    mhc_tokps = _run_perf(mhc, device)

    mem_msg = ""
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(device)
        mem_msg = f" | peak_mem={mem / (1024 ** 2):.1f} MiB"
    elif device.type == "mps":
        try:
            cur = torch.mps.current_allocated_memory()
            drv = torch.mps.driver_allocated_memory()
            mem_msg = f" | mps_current={cur / (1024 ** 2):.1f} MiB mps_driver={drv / (1024 ** 2):.1f} MiB"
        except Exception:
            mem_msg = " | mps_mem=unavailable"

    print(f"[perf] base_tokps={base_tokps:.1f} mhc_tokps={mhc_tokps:.1f}{mem_msg}")
