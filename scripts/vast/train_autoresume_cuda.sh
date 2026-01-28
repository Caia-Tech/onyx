#!/usr/bin/env bash
set -euo pipefail

# Vast.ai-friendly autoresume wrapper for `python -m onyx.train`.
# Notes:
# - Onyx training is currently single-process (no DDP); this uses only the visible GPU(s).
# - Put datasets/checkpoints on a mounted volume (commonly /workspace on Vast.ai).

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WORKDIR="${WORKDIR:-/workspace}"
if [[ -d "$WORKDIR" ]]; then
  DEFAULT_OUT_ROOT="$WORKDIR"
else
  DEFAULT_OUT_ROOT="$BASE_DIR"
fi

DATA_GLOB="${DATA_GLOB:-}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$BASE_DIR/tokenizer}"
MODEL_CONFIG="${MODEL_CONFIG:-$BASE_DIR/configs/onyx_1b.json}"
RUN_NAME="${RUN_NAME:-onyx_1b_vast}"

CKPT_DIR="${CKPT_DIR:-$DEFAULT_OUT_ROOT/checkpoints/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$DEFAULT_OUT_ROOT/logs}"

INIT_CKPT="${INIT_CKPT:-}"
INIT_CKPT_DIR="${INIT_CKPT_DIR:-}"

# Experimental mHC toggles (must match your checkpoint if you trained with --experimental_mhc)
EXPERIMENTAL_MHC="${EXPERIMENTAL_MHC:-0}"
MHC_N="${MHC_N:-2}"
MHC_MODE="${MHC_MODE:-mhc}"
MHC_SINKHORN="${MHC_SINKHORN:-1}"
MHC_SINKHORN_ITERS="${MHC_SINKHORN_ITERS:-10}"

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-131072}"
SHUFFLE_BUFFER_DOCS="${SHUFFLE_BUFFER_DOCS:-2048}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MIN_LR="${MIN_LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0002}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-1000}"
LOG_EVERY="${LOG_EVERY:-100}"
TRAIN_TOKENS_TARGET="${TRAIN_TOKENS_TARGET:-}"
MAX_STEPS="${MAX_STEPS:-}"

TEE="${TEE:-1}"

USE_COMPILE="${USE_COMPILE:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

if [[ -z "$DATA_GLOB" ]]; then
  cat >&2 <<EOF
DATA_GLOB is required.
Example:
  export DATA_GLOB=/workspace/data/datamix.jsonl
  export TOKENIZER_DIR=/workspace/tokenizer   # or keep default: $BASE_DIR/tokenizer
EOF
  exit 2
fi
if [[ ! -e "$DATA_GLOB" && "$DATA_GLOB" != *"*"* && "$DATA_GLOB" != *"?"* && "$DATA_GLOB" != *"["* ]]; then
  echo "DATA_GLOB not found: $DATA_GLOB" >&2
  exit 2
fi
if [[ ! -d "$TOKENIZER_DIR" ]]; then
  echo "TOKENIZER_DIR not found: $TOKENIZER_DIR" >&2
  exit 2
fi
if [[ ! -f "$MODEL_CONFIG" ]]; then
  echo "MODEL_CONFIG not found: $MODEL_CONFIG" >&2
  exit 2
fi

LOCK_DIR="${CKPT_DIR}/.train_autoresume.lock"
LOCK_PID_FILE="${LOCK_DIR}/pid"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [[ -f "$LOCK_PID_FILE" ]]; then
    existing_pid=$(cat "$LOCK_PID_FILE" 2>/dev/null || true)
    if [[ -n "${existing_pid:-}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "Another train_autoresume_cuda.sh instance is already running (pid: $existing_pid)." >&2
      exit 1
    fi
  fi
  rm -f "$LOCK_PID_FILE" 2>/dev/null || true
  rmdir "$LOCK_DIR" 2>/dev/null || true
  mkdir "$LOCK_DIR"
fi
echo $$ > "$LOCK_PID_FILE"
cleanup_lock() {
  rm -f "$LOCK_PID_FILE" 2>/dev/null || true
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "/venv/main/bin/python" ]]; then
    PYTHON_BIN="/venv/main/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Error: neither 'python' nor 'python3' found on PATH." >&2
    exit 1
  fi
fi

BASE_CMD=(
  "$PYTHON_BIN" -m onyx.train
  --data_glob "$DATA_GLOB"
  --tokenizer "$TOKENIZER_DIR"
  --model_config "$MODEL_CONFIG"
  --batch_size "$BATCH_SIZE"
  --max_seq_len "$MAX_SEQ_LEN"
  --tokens_per_step "$TOKENS_PER_STEP"
  --shuffle_buffer_docs "$SHUFFLE_BUFFER_DOCS"
  --num_epochs 1
  --learning_rate "$LEARNING_RATE"
  --min_lr "$MIN_LR"
  --warmup_ratio "$WARMUP_RATIO"
  --amp
  --amp_dtype bfloat16
  --save_dir "$CKPT_DIR"
  --save_every_steps "$SAVE_EVERY_STEPS"
  --log_every "$LOG_EVERY"
  --log_file "${LOG_DIR}/${RUN_NAME}_train.log"
  --mem_report_every 50
  --gc_collect_every 200
  --dataset_state_mode light
)

if [[ "$TEE" == "1" ]]; then
  BASE_CMD+=(--tee --flush_logs)
fi

if [[ -n "$TRAIN_TOKENS_TARGET" ]]; then
  BASE_CMD+=(--train_tokens_target "$TRAIN_TOKENS_TARGET")
fi
if [[ -n "$MAX_STEPS" ]]; then
  BASE_CMD+=(--max_steps "$MAX_STEPS")
fi
if [[ "$USE_COMPILE" == "1" ]]; then
  BASE_CMD+=(--compile --compile_mode reduce-overhead)
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  BASE_CMD+=(--gradient_checkpointing)
fi

if [[ "$EXPERIMENTAL_MHC" == "1" ]]; then
  BASE_CMD+=(--experimental_mhc)
  BASE_CMD+=(--mhc_n "$MHC_N")
  BASE_CMD+=(--mhc_mode "$MHC_MODE")
  if [[ "$MHC_SINKHORN" == "1" ]]; then
    BASE_CMD+=(--mhc_sinkhorn)
  else
    BASE_CMD+=(--no-mhc_sinkhorn)
  fi
  BASE_CMD+=(--mhc_sinkhorn_iters "$MHC_SINKHORN_ITERS")
fi

RETRY_DELAY_SEC="${RETRY_DELAY_SEC:-5}"
MAX_SAME_CKPT_FAILURES="${MAX_SAME_CKPT_FAILURES:-2}"

last_ckpt=""
same_ckpt_failures=0
child_pid=""
stop_requested=0

on_stop() {
  stop_requested=1
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
}
trap on_stop INT TERM HUP

get_ckpt_list() {
  local dir="$1"
  local f base step
  for f in "$dir"/checkpoint_step_*.pt "$dir"/interrupt_step_*.pt; do
    [[ -e "$f" ]] || continue
    base="${f##*/}"
    step="${base##*_step_}"
    step="${step%.pt}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    printf "%s\t%s\n" "$step" "$f"
  done | sort -nr -k1,1 | cut -f2-
}

while true; do
  ckpts=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && ckpts+=("$line")
  done < <(get_ckpt_list "$CKPT_DIR" || true)

  ckpt=""
  init_ckpt=""
  if [[ ${#ckpts[@]} -gt 0 ]]; then
    ckpt="${ckpts[0]}"
    if [[ "$ckpt" == "$last_ckpt" && $same_ckpt_failures -ge $MAX_SAME_CKPT_FAILURES && ${#ckpts[@]} -gt 1 ]]; then
      ckpt="${ckpts[1]}"
      echo "Latest checkpoint failed $same_ckpt_failures times; trying previous: $ckpt"
    fi
  else
    if [[ -n "${INIT_CKPT:-}" && -f "$INIT_CKPT" ]]; then
      init_ckpt="$INIT_CKPT"
    elif [[ -n "${INIT_CKPT_DIR:-}" && -d "$INIT_CKPT_DIR" ]]; then
      init_ckpt="$(get_ckpt_list "$INIT_CKPT_DIR" | head -n 1 || true)"
    fi
  fi

  CMD=("${BASE_CMD[@]}")
  if [[ -n "$ckpt" ]]; then
    CMD+=(--resume "$ckpt")
  elif [[ -n "$init_ckpt" ]]; then
    echo "No checkpoints found; warm-starting from: $init_ckpt"
    CMD+=(--init_checkpoint "$init_ckpt" --no-init_strict)
  fi

  start_ts=$(date +%s)
  echo "Starting: ${CMD[*]}"
  cd "$BASE_DIR"
  "${CMD[@]}" &
  child_pid=$!
  wait "$child_pid"
  status=$?
  child_pid=""
  end_ts=$(date +%s)
  runtime=$((end_ts - start_ts))

  if [[ $stop_requested -eq 1 ]]; then
    echo "Stop requested; exiting."
    exit 130
  fi

  if [[ $status -eq 0 ]]; then
    echo "Training finished cleanly."
    break
  fi

  if [[ -n "$ckpt" && "$ckpt" == "$last_ckpt" ]]; then
    same_ckpt_failures=$((same_ckpt_failures + 1))
  else
    last_ckpt="$ckpt"
    same_ckpt_failures=1
  fi

  echo "Exited with $status after ${runtime}s; retrying in ${RETRY_DELAY_SEC}s..."
  sleep "$RETRY_DELAY_SEC"
done
