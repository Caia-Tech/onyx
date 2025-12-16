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
