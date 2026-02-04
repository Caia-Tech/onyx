import torch
from onyx.model import build_packed_causal_mask


def test_build_packed_causal_mask_blocks_cross_doc():
    # Two docs: [0..2] and [3..5] in a length-6 sequence
    S = 6
    boundaries = torch.tensor([0, 3, 6], dtype=torch.int32)

    mask = build_packed_causal_mask(boundaries, S, device=torch.device("cpu"))  # [S,S] True=masked

    # Token 4 (doc2) should NOT attend to token 1 (doc1)
    assert mask[4, 1].item() is True

    # Token 4 can attend to token 3 (same doc, earlier)
    assert mask[4, 3].item() is False

    # Causal constraint: token 2 cannot attend to token 4
    assert mask[2, 4].item() is True

    # Within doc1: token 2 can attend to token 0
    assert mask[2, 0].item() is False
