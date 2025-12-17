import torch

from onyx_model import OnyxConfig, ChunkedLinearDeltaMemory


def test_dynamic_memory_hyperparams_are_input_dependent_in_training_mode():
    torch.manual_seed(0)
    cfg = OnyxConfig(
        d_model=8,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=8,
        d_ff=16,
        vocab_size=32,
        max_seq_len=16,
        train_seq_len=16,
        use_flash_attention=False,
        gradient_checkpointing=False,
        memory_chunk_size=2,
        max_memory_lr=0.2,
        min_memory_decay=0.5,
        inference_memory_lr=0.01,
        inference_memory_decay=0.95,
    )
    mem = ChunkedLinearDeltaMemory(d_in=8, d_out=4, config=cfg)

    # Make projections meaningfully input-dependent.
    mem.eta_proj.weight.data.fill_(0.25)
    mem.eta_proj.bias.data.zero_()
    mem.alpha_proj.weight.data.fill_(-0.25)
    mem.alpha_proj.bias.data.zero_()

    chunk0 = torch.zeros(2, 2, 8)
    chunk1 = torch.ones(2, 2, 8)

    eta0, alpha0 = mem._compute_chunk_hyperparams(chunk0, inference_mode=False)
    eta1, alpha1 = mem._compute_chunk_hyperparams(chunk1, inference_mode=False)

    assert eta0.shape == (2, 1, 1)
    assert alpha0.shape == (2, 1, 1)
    assert torch.all(eta0 >= 0.0) and torch.all(eta0 <= cfg.max_memory_lr)
    assert torch.all(alpha0 >= cfg.min_memory_decay) and torch.all(alpha0 <= 1.0)
    assert not torch.allclose(eta0, eta1)
    assert not torch.allclose(alpha0, alpha1)


def test_dynamic_memory_hyperparams_are_not_used_in_inference_mode():
    torch.manual_seed(0)
    cfg = OnyxConfig(
        d_model=8,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=8,
        d_ff=16,
        vocab_size=32,
        max_seq_len=16,
        train_seq_len=16,
        use_flash_attention=False,
        gradient_checkpointing=False,
        memory_chunk_size=2,
        inference_memory_lr=0.01,
        inference_memory_decay=0.95,
    )
    mem = ChunkedLinearDeltaMemory(d_in=8, d_out=4, config=cfg)

    # With zero projections, inference_mode should reduce to fixed inference defaults.
    mem.eta_proj.weight.data.zero_()
    mem.eta_proj.bias.data.zero_()
    mem.alpha_proj.weight.data.zero_()
    mem.alpha_proj.bias.data.zero_()

    chunk0 = torch.zeros(1, 2, 8)
    chunk1 = torch.ones(1, 2, 8)

    eta0, alpha0 = mem._compute_chunk_hyperparams(chunk0, inference_mode=True)
    eta1, alpha1 = mem._compute_chunk_hyperparams(chunk1, inference_mode=True)

    assert torch.allclose(eta0, eta1)
    assert torch.allclose(alpha0, alpha1)
    assert torch.allclose(eta0, torch.tensor(cfg.inference_memory_lr).view(1, 1, 1))
    assert torch.allclose(alpha0, torch.tensor(cfg.inference_memory_decay).view(1, 1, 1))

    # With non-zero projections, inference_mode becomes input-dependent.
    mem.eta_proj.weight.data.fill_(0.5)
    mem.alpha_proj.weight.data.fill_(-0.5)
    eta0b, alpha0b = mem._compute_chunk_hyperparams(chunk0, inference_mode=True)
    eta1b, alpha1b = mem._compute_chunk_hyperparams(chunk1, inference_mode=True)
    assert not torch.allclose(eta0b, eta1b)
    assert not torch.allclose(alpha0b, alpha1b)


def test_dynamic_hyperparam_projections_receive_gradients_in_training_mode():
    torch.manual_seed(0)
    cfg = OnyxConfig(
        d_model=8,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=8,
        d_ff=16,
        vocab_size=32,
        max_seq_len=16,
        train_seq_len=16,
        use_flash_attention=False,
        gradient_checkpointing=False,
        memory_chunk_size=2,
    )
    mem = ChunkedLinearDeltaMemory(d_in=8, d_out=4, config=cfg).train()

    x = torch.randn(1, 4, 8)
    out, _M = mem(x, M=None, inference_mode=False, update_memory=True)
    loss = out.sum()
    loss.backward()

    assert mem.eta_proj.weight.grad is not None
    assert mem.alpha_proj.weight.grad is not None
    assert mem.eta_proj.weight.grad.abs().sum().item() > 0
    assert mem.alpha_proj.weight.grad.abs().sum().item() > 0

    mem.zero_grad(set_to_none=True)
    out_inf, _M_inf = mem(x, M=None, inference_mode=True, update_memory=True)
    loss_inf = out_inf.sum()
    loss_inf.backward()
    assert mem.eta_proj.weight.grad is not None
    assert mem.alpha_proj.weight.grad is not None
