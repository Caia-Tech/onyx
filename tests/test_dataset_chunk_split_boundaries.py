import json
from pathlib import Path

import torch

from onyx.train import StreamingPackedDataset


class _Tok:
    eos_token_id = 2
    pad_token_id = 2

    def __len__(self):
        return 128

    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        if text == "DOC1":
            n = 20
        elif text == "DOC2":
            n = 4
        else:
            n = 1
        return list(range(10, 10 + n))


def test_boundaries_shift_across_chunk_splits(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "DOC1"}) + "\n")
        f.write(json.dumps({"text": "DOC2"}) + "\n")

    ds = StreamingPackedDataset(
        data_glob=str(p),
        tokenizer=_Tok(),
        max_seq_len=16,
        pack=True,
        seed=0,
        drop_remainder=False,
        emit_cu_seqlens=True,
        shuffle_buffer_docs=0,
    )
    it = iter(ds)
    s0 = next(it)
    s1 = next(it)

    # First chunk is DOC1 continuation only.
    cu0 = s0["cu_seqlens"]
    assert cu0[0].item() == 0
    assert cu0[-1].item() == 16

    # Second chunk should contain an internal boundary where DOC2 begins.
    cu1 = s1["cu_seqlens"]
    assert cu1[0].item() == 0
    assert cu1[-1].item() == 16
    assert 5 in set(int(x.item()) for x in cu1)  # DOC2 starts after DOC1 remainder (5 tokens)
    assert torch.all(cu1[1:] >= cu1[:-1])
