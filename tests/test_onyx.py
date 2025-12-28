import math

import pytest
import torch
import torch.nn as nn

from onyx.model import OnyxConfig, Onyx, M3Optimizer, zeropower_via_newtonschulz5
from onyx.train import CMSFrequencyManager


@pytest.fixture
def tiny_config():
    return OnyxConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=100,
        max_seq_len=64,
        train_seq_len=64,
        memory_chunk_size=16,
        use_cms_ffn=True,
        cms_num_levels=2,
        cms_base_chunk=1,
        cms_chunk_multiplier=2,
        use_hope_attention=True,
        memory_lr_learnable=True,
        memory_decay_learnable=True,
        use_flash_attention=False,
        gradient_checkpointing=False,
    )


@pytest.fixture
def model(tiny_config):
    torch.manual_seed(0)
    return Onyx(tiny_config)


def test_newton_schulz_orthogonality():
    torch.manual_seed(42)
    G = torch.randn(100, 100, dtype=torch.float32)

    X = zeropower_via_newtonschulz5(G, steps=5)

    I_true = torch.eye(100, dtype=torch.float32)
    diff = torch.norm(X @ X.T - I_true)
    rel = diff / torch.norm(I_true)

    # Newton–Schulz is an iterative approximation; require that it meaningfully
    # improves orthogonality vs the same normalization baseline.
    Gn = G / (G.norm() + 1e-7)
    rel0 = torch.norm(Gn @ Gn.T - I_true) / torch.norm(I_true)

    assert torch.isfinite(X).all()
    assert rel < 0.5
    assert rel < rel0


def test_m3_optimizer_structure(model):
    opt = M3Optimizer(model.parameters(), lr=0.01)

    x = torch.randint(0, 100, (1, 10))
    out = model(x, labels=x)
    loss = out["loss"]
    assert loss is not None
    loss.backward()
    opt.step()

    found_matrix_logic = False
    for group in opt.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = opt.state[p]

            if p.ndim == 2:
                if "slow_memory" in state:
                    found_matrix_logic = True
                assert "exp_avg_sq" not in state

    assert found_matrix_logic


def test_cms_gradient_masking(model, tiny_config):
    cms_manager = CMSFrequencyManager(model, tiny_config)

    level_1_param = None
    for item in cms_manager.param_schedules:
        if item["level"] == 1:
            level_1_param = item["param"]
            break

    if level_1_param is None:
        pytest.skip("Could not find a CMS Level 1 parameter to test.")

    x = torch.randint(0, 100, (1, 10))
    model.zero_grad(set_to_none=True)
    loss = model(x, labels=x)["loss"]
    assert loss is not None
    loss.backward()

    cms_manager.mask_gradients(global_step=1)
    is_masked = (level_1_param.grad is None) or torch.all(level_1_param.grad == 0)
    assert is_masked

    model.zero_grad(set_to_none=True)
    loss2 = model(x, labels=x)["loss"]
    assert loss2 is not None
    loss2.backward()
    cms_manager.mask_gradients(global_step=2)

    is_active = (level_1_param.grad is not None) and torch.any(level_1_param.grad != 0)
    assert is_active


def test_dynamic_hyperparams_exist(model):
    mem_module = model.layers[0].attention.k_memory.memory
    assert hasattr(mem_module, "eta_proj")
    assert hasattr(mem_module, "alpha_proj")


def test_memory_feedback_loop():
    torch.manual_seed(0)
    cfg = OnyxConfig(
        d_model=64,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=100,
        max_seq_len=64,
        train_seq_len=64,
        memory_chunk_size=16,
        use_flash_attention=False,
        gradient_checkpointing=False,
        use_hope_attention=True,
        self_referential_keys=True,
        generate_own_values=True,
        normalize_keys=False,
        use_memory_gate=True,
        memory_max_norm=1e9,
        max_memory_lr=1.0,
        min_memory_decay=0.0,
        inference_memory_lr=1.0,
        inference_memory_decay=1.0,
    )
    m = Onyx(cfg).eval()

    k_mem = m.layers[0].attention.k_memory
    assert k_mem.value_gen is not None

    nn.init.eye_(k_mem.value_gen[0].weight)
    nn.init.eye_(k_mem.value_gen[2].weight)

    # Ensure the delta-memory uses near-identity decay and a high learning rate.
    delta_mem = k_mem.memory
    delta_mem.inference_eta.fill_(cfg.max_memory_lr)
    delta_mem.inference_alpha.fill_(1.0)
    if hasattr(delta_mem, "gate_proj"):
        delta_mem.gate_proj.weight.data.zero_()
        delta_mem.gate_proj.bias.data.fill_(20.0)

    B, S, D = 1, 16, cfg.d_model
    x = torch.ones(B, S, D)
    M0 = torch.eye(D).unsqueeze(0)

    _out, M1 = k_mem(x, M=M0, inference_mode=True, update_memory=True)
    delta = torch.norm(M1 - M0)
    assert delta > 1e-3


def test_overfitting_single_batch(model):
    torch.manual_seed(0)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 100, (2, 32))
    labels = x.clone()

    initial_loss = model(x, labels=labels)["loss"].item()

    for _ in range(15):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, labels=labels)["loss"]
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.8


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
