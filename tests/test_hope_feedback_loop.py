import torch
import torch.nn as nn
import torch.nn.functional as F

from onyx.model import OnyxConfig, ChunkedLinearDeltaMemory, ChunkedSelfReferentialMemory


def test_chunked_linear_delta_memory_uses_target_generator_for_delta_error():
    torch.manual_seed(0)
    cfg = OnyxConfig(
        d_model=4,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=4,
        d_ff=8,
        vocab_size=16,
        max_seq_len=8,
        train_seq_len=8,
        use_flash_attention=False,
        gradient_checkpointing=False,
        memory_chunk_size=2,
        normalize_keys=False,  # makes v_mean == Mk when targets==outputs
        use_delta_rule=True,
        use_memory_gate=False,
        memory_max_norm=1e9,
        max_memory_lr=1.0,
        min_memory_decay=0.0,
    )
    mem = ChunkedLinearDeltaMemory(d_in=3, d_out=2, config=cfg)

    B, C = 2, 2
    M = torch.randn(B, 2, 3)
    chunk = torch.randn(B, C, 3)
    eta = torch.ones(B, 1, 1)
    alpha = torch.ones(B, 1, 1)

    outputs, M_no_target = mem._process_chunk(
        M.clone(),
        chunk,
        eta,
        alpha,
        update_memory=True,
        target_generator=None,
    )
    assert torch.allclose(M_no_target, M)

    def target_gen(out: torch.Tensor) -> torch.Tensor:
        return out + 1.0

    outputs2, M_target = mem._process_chunk(
        M.clone(),
        chunk,
        eta,
        alpha,
        update_memory=True,
        target_generator=target_gen,
    )
    assert torch.allclose(outputs2, outputs)
    assert not torch.allclose(M_target, M)

    k_mean = chunk.mean(dim=1)  # normalize_keys=False
    v_target = (outputs + 1.0).mean(dim=1)
    Mk = torch.bmm(M, k_mean.unsqueeze(-1)).squeeze(-1)
    error = v_target - Mk
    expected_update = torch.bmm(error.unsqueeze(-1), k_mean.unsqueeze(1))
    expected_M = M + expected_update
    assert torch.allclose(M_target, expected_M, atol=1e-5, rtol=1e-5)


def test_chunked_self_referential_memory_feedback_loop_updates_memory_state():
    torch.manual_seed(0)
    cfg = OnyxConfig(
        d_model=4,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=4,
        d_ff=8,
        vocab_size=16,
        max_seq_len=8,
        train_seq_len=8,
        use_flash_attention=False,
        gradient_checkpointing=False,
        memory_chunk_size=2,
        normalize_keys=False,
        use_delta_rule=True,
        generate_own_values=True,
        use_memory_gate=False,
        memory_max_norm=1e9,
        max_memory_lr=1.0,
        min_memory_decay=0.0,
        inference_memory_lr=1.0,
        inference_memory_decay=1.0,
    )

    mem = ChunkedSelfReferentialMemory(d_in=2, d_out=2, config=cfg).eval()
    assert mem.value_gen is not None

    # Make value_gen non-identity so v_hat != retrieved values.
    nn.init.eye_(mem.value_gen[0].weight)
    nn.init.eye_(mem.value_gen[2].weight)

    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [B=1,S=2,D=2]
    M0 = torch.eye(2).unsqueeze(0)  # [1,2,2]

    out, M1 = mem(x, M=M0, inference_mode=True, update_memory=True)

    retrieved = torch.bmm(x, M0.transpose(-1, -2))
    expected_out = retrieved + F.silu(retrieved)
    assert torch.allclose(out, expected_out, atol=1e-6, rtol=0.0)

    # With normalize_keys=False and alpha≈1, memory should only change if the
    # feedback targets (v_hat) are used in the delta update.
    assert (M1 - M0).abs().sum().item() > 1e-3

