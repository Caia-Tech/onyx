# Onyx

An experimental language model with Hope Attention and persistent memory, developed in public by [Caia Tech](https://github.com/Caia-Tech).

## License

Do whatever you want

## Author

Marvin Tutt, Caia Tech

## FP16 autocast benchmark (MPS)

A) Baseline fp32 bench (resume)
```bash
python onyx_train.py \
  --resume /Users/owner/Desktop/caiatech/models/onyx/checkpoints/checkpoint_interrupt.pt \
  --bench_steps 300 \
  --disable_saves_during_bench
```

B) Autocast fp16 bench (resume)
```bash
python onyx_train.py \
  --resume /Users/owner/Desktop/caiatech/models/onyx/checkpoints/checkpoint_interrupt.pt \
  --amp --amp_dtype float16 \
  --bench_steps 300 \
  --disable_saves_during_bench
```

## Train a small local tokenizer (recommended for tiny datasets)

If you have limited data, a smaller vocab can intentionally produce *more tokens*
per document (more training signal per byte of text) and also reduces the
vocab-dependent embedding/LM-head params.

Train a ByteLevel-BPE tokenizer from your JSONL:
```bash
python onyx_train_tokenizer.py \
  --data_glob "/Users/owner/Desktop/caiatech/datasets/onyx/caia-corpus.jsonl" \
  --vocab_size 8192 \
  --out_dir "/Users/owner/Desktop/caiatech/models/onyx/tokenizers/caia_bpe_8k"
```

Then point training/inference at the local folder:
```bash
python onyx_train.py --tokenizer "/Users/owner/Desktop/caiatech/models/onyx/tokenizers/caia_bpe_8k" ...
python onyx_inference.py --tokenizer "/Users/owner/Desktop/caiatech/models/onyx/tokenizers/caia_bpe_8k" ...
```

## Convert corpus to chat format

If you want the interactive `onyx_inference.py` chat mode to behave better, it helps
to train on a corpus that looks like:
`System: ...`, `User: ...`, `Assistant: ...`.

This repo includes a converter that writes a new JSONL with `system/user/assistant` fields:
```bash
python onyx_make_chat_corpus.py \
  --data_glob "/Users/owner/Desktop/caiatech/datasets/onyx/caia-corpus.jsonl" \
  --out_file "/Users/owner/Desktop/caiatech/datasets/onyx/caia-corpus-chat.jsonl" \
  --mode question_split
```
