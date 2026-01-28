# Vast.ai notes (NVIDIA H-series)

These scripts help run `python -m onyx.train` on Vast.ai CUDA machines (e.g., H100).

## Suggested container

Use a PyTorch CUDA base image so you don't have to install torch yourself, e.g.:

- `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` (or similar)

## Vast.ai instance checklist

- Mount a persistent volume to `/workspace` (dataset + checkpoints + logs).
- Prefer a single H100 until you add DDP to `onyx.train` (multi-GPU instances will be underutilized).
- Make sure your instance has enough disk for checkpoints (and your dataset if you copy it locally).

## Quick start

1) SSH into the instance and clone your repo into the container (or attach it as a volume).

2) Put your dataset and tokenizer on the mounted volume (commonly `/workspace`):

- Dataset example: `/workspace/data/datamix.jsonl`
- Tokenizer example: `/workspace/tokenizer` (or use the repo’s `tokenizer/`)

3) Install deps:

```bash
bash scripts/vast/setup.sh
```

4) Run training (autoresumes from latest checkpoint in `CKPT_DIR`):

```bash
source scripts/vast/presets/150m_24gb.sh   # or scripts/vast/presets/h100_80gb_1b.sh
export DATA_GLOB=/workspace/data/datamix.jsonl
export TOKENIZER_DIR=/workspace/tokenizer   # optional
export MODEL_CONFIG=$PWD/configs/onyx_1b.json
export RUN_NAME=onyx_1b_h100
export CKPT_DIR=/workspace/checkpoints/$RUN_NAME
export LOG_DIR=/workspace/logs
export INIT_CKPT=/workspace/init/checkpoint_grown.pt   # optional warm-start when CKPT_DIR is empty

# If your checkpoint was trained with mHC, enable it here too:
# export EXPERIMENTAL_MHC=1
# export MHC_N=2
# export MHC_MODE=mhc
# export MHC_SINKHORN=1
# export MHC_SINKHORN_ITERS=10

 bash scripts/vast/train_autoresume_cuda.sh
```

## Growing a checkpoint (warm-start across model sizes)

If your `INIT_CKPT` was trained with a smaller `d_model` than your target config (e.g. 4M `d_model=192` -> 1B `d_model=2048`), you must grow it first.

Example (run on the Vast box, from `/workspace/onyx`):

```bash
python tools/grow_model_ckpt.py \
  --in_ckpt /workspace/init/checkpoint_4m.pt \
  --out_ckpt /workspace/init/checkpoint_1b_grown.pt \
  --old_config configs/onyx_4m_8l.json \
  --new_config configs/onyx_1b.json \
  --sanity_check
```

Then set:

```bash
export INIT_CKPT=/workspace/init/checkpoint_1b_grown.pt
```

## Useful knobs

- `TEE=1` (default) mirrors logs to terminal as well as `LOG_DIR` (set `TEE=0` for file-only).
- `BATCH_SIZE` (default `4`)
- `MAX_SEQ_LEN` (default `2048`)
- `TOKENS_PER_STEP` (controls grad accumulation; default `131072`)
- `USE_COMPILE=1` (enables `torch.compile`)
- `GRADIENT_CHECKPOINTING=1` (reduces VRAM at cost of speed)

## Multi-GPU

`onyx.train` is currently single-process (no DDP), so these scripts will only use one process even on multi-GPU instances.
