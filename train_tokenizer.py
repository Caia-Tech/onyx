import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to a text file (or JSONL treated as raw text lines).")
    ap.add_argument("--out", required=True, help="Output directory for save_pretrained().")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--min_frequency", type=int, default=2)
    args = ap.parse_args()

    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL,
        initial_alphabet=ByteLevel.alphabet(),  # critical: prevents UNK-collapse
    )

    tok.train([args.input], trainer)

    tok.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[
            ("[BOS]", tok.token_to_id("[BOS]")),
            ("[EOS]", tok.token_to_id("[EOS]")),
        ],
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast.save_pretrained(args.out)

    print("Saved tokenizer to:", args.out)
    print("vocab_size:", fast.vocab_size)
    print("special token ids:", {
        "pad": fast.pad_token_id,
        "unk": fast.unk_token_id,
        "bos": fast.bos_token_id,
        "eos": fast.eos_token_id,
    })

if __name__ == "__main__":
    main()
