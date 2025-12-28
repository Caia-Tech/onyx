import torch

from onyx.train import StreamingPackedDataset


def _assert_labels_match_input_except_padding(sample):
    x = sample["input_ids"]
    y = sample["labels"]
    assert x.shape == y.shape
    for i in range(x.numel()):
        if y[i].item() != -100:
            assert y[i].item() == x[i].item()


def test_packing_outputs_fixed_len_and_masks(dummy_tokenizer, tiny_jsonl):
    ds = StreamingPackedDataset(
        data_glob=tiny_jsonl,
        tokenizer=dummy_tokenizer,
        max_seq_len=32,
        pack=True,
        seed=123,
        drop_remainder=False,
        emit_cu_seqlens=True,
    )
    sample = next(iter(ds))

    assert sample["input_ids"].shape == (32,)
    assert sample["labels"].shape == (32,)
    assert (sample["labels"] == -100).sum().item() >= 0
    _assert_labels_match_input_except_padding(sample)


def test_varlen_boundaries_are_monotonic_and_end_at_seq_len(dummy_tokenizer, tiny_jsonl):
    ds = StreamingPackedDataset(
        data_glob=tiny_jsonl,
        tokenizer=dummy_tokenizer,
        max_seq_len=32,
        pack=True,
        seed=123,
        drop_remainder=False,
        emit_cu_seqlens=True,
    )
    sample = next(iter(ds))
    cu = sample["cu_seqlens"]
    assert cu.dtype in (torch.int32, torch.int64)
    assert cu[0].item() == 0
    assert cu[-1].item() == 32
    assert torch.all(cu[1:] >= cu[:-1])
    _assert_labels_match_input_except_padding(sample)
