import torch
from onyx.model import OnyxConfig, StandardAttention


def run_attn(attn, x, boundaries=None):
    packed_cu = None
    packed_num = None
    if boundaries is not None:
        packed_cu = boundaries.unsqueeze(0)
        packed_num = torch.tensor([boundaries.numel()], dtype=torch.int32)
    out, _, _ = attn(
        x,
        packed_cu_seqlens=packed_cu,
        packed_num_segs=packed_num,
        kv_cache=None,
        position_offset=0,
    )
    return out


def test_standard_attention_respects_boundaries():
    torch.manual_seed(0)
    cfg = OnyxConfig(
        d_model=8, n_heads=2, n_kv_heads=1, head_dim=4, max_seq_len=16,
        use_flash_attention=False
    )
    attn = StandardAttention(cfg)
    attn.eval()

    B, S, D = 1, 6, 8
    x = torch.randn(B, S, D)

    # Make the two docs very different so leakage matters
    x[:, :3, :] += 5.0   # doc1 shifted
    x[:, 3:, :] -= 5.0   # doc2 shifted

    boundaries = torch.tensor([0, 3, 6], dtype=torch.int32)

    out_leaky = run_attn(attn, x, boundaries=None)
    out_blocked = run_attn(attn, x, boundaries=boundaries)

    # They should differ because leakage changes attention context
    diff = (out_leaky - out_blocked).abs().max().item()
    assert diff > 1e-5
