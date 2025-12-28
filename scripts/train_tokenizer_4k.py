#!/usr/bin/env python3
import argparse, json, os
from glob import glob
from pathlib import Path
from typing import Iterator, Dict, Any

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from transformers import PreTrainedTokenizerFast


def iter_jsonl_texts(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj: Dict[str, Any] = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            text = obj.get("text") or obj.get("content") or obj.get("input") or ""
            if isinstance(text, str) and text.strip():
                yield text
                continue

            # chat-shaped {system,user,assistant}
            user = obj.get("user")
            assistant = obj.get("assistant")
            system = obj.get("system", "")
            if isinstance(user, str) and isinstance(assistant, str):
                system = system if isinstance(system, str) else ""
                system = system.strip()
                user = user.strip()
                assistant = assistant.strip()
                if system:
                    yield f"System: {system}\n\nUser: {user}\nAssistant: {assistant}"
                else:
                    yield f"User: {user}\nAssistant: {assistant}"


def iter_corpus(data_glob: str) -> Iterator[str]:
    if "*" in data_glob or "?" in data_glob:
        files = sorted(glob(data_glob))
    else:
        files = [data_glob]
    for fp in files:
        if os.path.isfile(fp):
            yield from iter_jsonl_texts(fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", required=True, help="path or glob to jsonl")
    ap.add_argument("--out_dir", required=True, help="where to save tokenizer files")
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--min_frequency", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Byte-level BPE: robust for code + unicode + weird symbols
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([
        normalizers.NFKC(),  # normalize width/compat forms
    ])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()

    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
    )

    tok.train_from_iterator(iter_corpus(args.data_glob), trainer=trainer)

    # Wrap in HF fast tokenizer + set specials
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    hf_tok.save_pretrained(out_dir)
    print(f"Saved tokenizer to: {out_dir}")
    print(f"Vocab size: {len(hf_tok)}")
    print(f"EOS id: {hf_tok.eos_token_id} | PAD id: {hf_tok.pad_token_id}")


if __name__ == "__main__":
    main()
