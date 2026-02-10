#!/usr/bin/env bash
set -euo pipefail

# Starting point for Onyx ~300M (RMT enabled) on a single H100 80GB.

export RUN_NAME="${RUN_NAME:-onyx_300m_h100}"

# Set these explicitly for your instance:
#   export DATA_GLOB=/workspace/data/datamix.jsonl
#   export TOKENIZER_DIR=/workspace/tokenizer

export MODEL_CONFIG="${MODEL_CONFIG:-$PWD/configs/onyx.json}"
export CKPT_DIR="${CKPT_DIR:-/workspace/checkpoints/$RUN_NAME}"
export LOG_DIR="${LOG_DIR:-/workspace/logs}"

# accumulation_steps = tokens_per_step / (batch_size * max_seq_len)
export BATCH_SIZE="${BATCH_SIZE:-8}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
export TOKENS_PER_STEP="${TOKENS_PER_STEP:-262144}"

export LEARNING_RATE="${LEARNING_RATE:-3e-4}"
export MIN_LR="${MIN_LR:-1e-4}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.0005}"

export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-1000}"
export LOG_EVERY="${LOG_EVERY:-100}"

# Performance/stability
export USE_COMPILE="${USE_COMPILE:-1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
