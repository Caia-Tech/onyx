#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from byte_tokenizer import bytes_to_unicode


def _build_vocab_bytes_only() -> Dict[str, int]:
    enc = bytes_to_unicode()
    # Ensure ids match raw byte values.
    return {enc[b]: b for b in range(256)}

SpecialsMode = Literal["none", "eos", "full"]


def _apply_specials(vocab: Dict[str, int], mode: SpecialsMode) -> Tuple[Dict[str, int], Dict[str, Optional[str]]]:
    """
    Returns (vocab, hf_special_token_strings).

    Minimal requirements for `onyx/train.py`:
    - `tokenizer.eos_token_id` must not be None (used as document boundary).
    """
    hf: Dict[str, Optional[str]] = {"unk": None, "pad": None, "bos": None, "eos": None}

    if mode == "none":
        return vocab, hf

    if mode == "eos":
        # Add only EOS and reuse it for padding to keep things simple.
        vocab["[EOS]"] = len(vocab)  # 256
        hf["eos"] = "[EOS]"
        hf["pad"] = "[EOS]"
        return vocab, hf

    if mode == "full":
        for t in ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]:
            vocab[t] = len(vocab)
        hf.update({"unk": "[UNK]", "pad": "[PAD]", "bos": "[BOS]", "eos": "[EOS]"})
        return vocab, hf

    raise ValueError(f"Unknown specials mode: {mode}")


def build_tokenizer(*, add_prefix_space: bool, specials: SpecialsMode, add_bos_eos: bool) -> PreTrainedTokenizerFast:
    vocab = _build_vocab_bytes_only()
    vocab, hf_specials = _apply_specials(vocab, specials)

    tok = Tokenizer(
        models.BPE(
            vocab=vocab,
            merges=[],
            unk_token=hf_specials["unk"],
        )
    )
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
    tok.decoder = decoders.ByteLevel()

    if add_bos_eos:
        if specials != "full":
            raise ValueError("--add_bos_eos requires --specials full")
        bos_id = vocab["[BOS]"]
        eos_id = vocab["[EOS]"]
        tok.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
        )

    hf_kwargs = {"tokenizer_object": tok}
    if hf_specials["unk"] is not None:
        hf_kwargs["unk_token"] = hf_specials["unk"]
    if hf_specials["pad"] is not None:
        hf_kwargs["pad_token"] = hf_specials["pad"]
    if hf_specials["bos"] is not None:
        hf_kwargs["bos_token"] = hf_specials["bos"]
    if hf_specials["eos"] is not None:
        hf_kwargs["eos_token"] = hf_specials["eos"]

    hf_tok = PreTrainedTokenizerFast(**hf_kwargs)
    hf_tok.clean_up_tokenization_spaces = False
    return hf_tok


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a 256-byte vocab tokenizer (optionally + special tokens).")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to write tokenizer files into.")
    ap.add_argument(
        "--specials",
        choices=["none", "eos", "full"],
        default="eos",
        help="Special tokens to add. Default: eos (adds [EOS] as id 256 and also uses it as pad).",
    )
    ap.add_argument(
        "--add_bos_eos",
        action="store_true",
        help="Add a post-processor that wraps sequences with [BOS] and [EOS] (requires --specials full).",
    )
    ap.add_argument(
        "--add_prefix_space",
        action="store_true",
        help="Match GPT-2 style byte-level behavior by prefixing a space when appropriate.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_tok = build_tokenizer(
        add_prefix_space=bool(args.add_prefix_space),
        specials=args.specials,
        add_bos_eos=bool(args.add_bos_eos),
    )
    hf_tok.save_pretrained(out_dir)

    print(f"Saved tokenizer to: {out_dir}")
    print(f"len(tokenizer) = {len(hf_tok)}")
    print(f"PAD={hf_tok.pad_token_id} UNK={hf_tok.unk_token_id} BOS={hf_tok.bos_token_id} EOS={hf_tok.eos_token_id}")


if __name__ == "__main__":
    main()
