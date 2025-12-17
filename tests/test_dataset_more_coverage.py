import torch

from onyx_train import StreamingPackedDataset


def test_packed_cu_seqlens_contains_internal_boundaries(dummy_tokenizer, tiny_jsonl):
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

    # Should include at least one internal boundary for multi-doc packing.
    assert cu.numel() >= 3
    assert cu[0].item() == 0
    assert cu[-1].item() == 32
    assert torch.all(cu[1:] >= cu[:-1])
    assert torch.all((cu >= 0) & (cu <= 32))


def test_non_packed_streaming_dataset_emits_fixed_len(dummy_tokenizer, tiny_jsonl):
    ds = StreamingPackedDataset(
        data_glob=tiny_jsonl,
        tokenizer=dummy_tokenizer,
        max_seq_len=32,
        pack=False,
        seed=123,
        drop_remainder=False,
        emit_cu_seqlens=False,
    )
    sample = next(iter(ds))
    assert sample["input_ids"].shape == (32,)
    assert sample["labels"].shape == (32,)
