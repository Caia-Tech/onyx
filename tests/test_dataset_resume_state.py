import json

import pytest

from onyx.train import StreamingPackedDataset


def _normalize_sample(sample):
    out = {
        "input_ids": sample["input_ids"].tolist(),
        "labels": sample["labels"].tolist(),
    }
    if "cu_seqlens" in sample:
        out["cu_seqlens"] = sample["cu_seqlens"].tolist()
        out["max_seqlen"] = int(sample["max_seqlen"])
    return out


def _collect_samples(dataset, n):
    it = iter(dataset)
    return [_normalize_sample(next(it)) for _ in range(n)]


def _write_jsonl(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")


@pytest.mark.parametrize("pack", [True, False])
def test_dataset_resume_state_matches_continuation(dummy_tokenizer, tiny_jsonl, pack):
    ds = StreamingPackedDataset(
        data_glob=tiny_jsonl,
        tokenizer=dummy_tokenizer,
        max_seq_len=16,
        pack=pack,
        seed=123,
        drop_remainder=False,
        emit_cu_seqlens=True,
        shuffle_buffer_docs=1,
    )
    _ = _collect_samples(ds, 1)
    state = ds.state_dict()
    expected = _collect_samples(ds, 2)

    ds2 = StreamingPackedDataset(
        data_glob=tiny_jsonl,
        tokenizer=dummy_tokenizer,
        max_seq_len=16,
        pack=pack,
        seed=123,
        drop_remainder=False,
        emit_cu_seqlens=True,
        shuffle_buffer_docs=1,
    )
    ds2.load_state_dict(state)
    resumed = _collect_samples(ds2, 2)

    assert resumed == expected


def test_dataset_resume_with_shuffle_buffer_multi_file(tmp_path, dummy_tokenizer):
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    _write_jsonl(file_a, [f"alpha-doc-{i:02d}" for i in range(6)])
    _write_jsonl(file_b, [f"bravo-doc-{i:02d}" for i in range(6)])

    data_glob = str(tmp_path / "*.jsonl")
    ds = StreamingPackedDataset(
        data_glob=data_glob,
        tokenizer=dummy_tokenizer,
        max_seq_len=8,
        pack=False,
        seed=7,
        drop_remainder=False,
        emit_cu_seqlens=False,
        shuffle_buffer_docs=3,
    )
    _ = _collect_samples(ds, 2)
    state = ds.state_dict()
    expected = _collect_samples(ds, 3)

    ds2 = StreamingPackedDataset(
        data_glob=data_glob,
        tokenizer=dummy_tokenizer,
        max_seq_len=8,
        pack=False,
        seed=7,
        drop_remainder=False,
        emit_cu_seqlens=False,
        shuffle_buffer_docs=3,
    )
    ds2.load_state_dict(state)
    resumed = _collect_samples(ds2, 3)

    assert resumed == expected


def test_dataset_resume_state_mismatch_falls_back_to_start(dummy_tokenizer, tiny_jsonl):
    ds = StreamingPackedDataset(
        data_glob=tiny_jsonl,
        tokenizer=dummy_tokenizer,
        max_seq_len=16,
        pack=True,
        seed=123,
        drop_remainder=False,
        emit_cu_seqlens=True,
        shuffle_buffer_docs=1,
    )
    _ = _collect_samples(ds, 1)
    state = ds.state_dict()

    ds_mismatch = StreamingPackedDataset(
        data_glob=tiny_jsonl,
        tokenizer=dummy_tokenizer,
        max_seq_len=12,
        pack=True,
        seed=123,
        drop_remainder=False,
        emit_cu_seqlens=True,
        shuffle_buffer_docs=1,
    )
    ds_mismatch.load_state_dict(state)
    after = _collect_samples(ds_mismatch, 1)

    ds_fresh = StreamingPackedDataset(
        data_glob=tiny_jsonl,
        tokenizer=dummy_tokenizer,
        max_seq_len=12,
        pack=True,
        seed=123,
        drop_remainder=False,
        emit_cu_seqlens=True,
        shuffle_buffer_docs=1,
    )
    fresh = _collect_samples(ds_fresh, 1)

    assert after == fresh
