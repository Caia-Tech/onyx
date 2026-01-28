import json

from onyx.train import StreamingPackedDataset, _jsonl_record_to_doc


def test_jsonl_record_to_doc_accepts_role_content_chat_single_turn():
    doc = _jsonl_record_to_doc(
        {
            "chat": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
    )
    assert isinstance(doc, dict)
    assert doc["user"] == "Hello"
    assert doc["assistant"] == "Hi there"


def test_streaming_dataset_accepts_role_content_chat(tmp_path, dummy_tokenizer):
    p = tmp_path / "data.jsonl"
    row = {
        "chat": [
            {"role": "user", "content": "Tell me a long story about cats."},
            {"role": "assistant", "content": "A" * 100},
        ]
    }
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    ds = StreamingPackedDataset(
        data_glob=str(p),
        tokenizer=dummy_tokenizer,
        max_seq_len=32,
        pack=False,
        seed=123,
        drop_remainder=False,
        emit_cu_seqlens=False,
        shuffle_buffer_docs=1,
    )
    sample = next(iter(ds))
    labels = sample["labels"].tolist()
    assert any(v == -100 for v in labels)
    assert any(v != -100 for v in labels)
