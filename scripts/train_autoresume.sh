#!/usr/bin/env bash
set -u
set -o pipefail

if [[ -z "${CAFFEINATED:-}" ]]; then
  export CAFFEINATED=1
  exec caffeinate -dimsu -- "$0" "$@"
fi

BASE_DIR="/Users/owner/Desktop/caiatech/models/onyx"
DATA_GLOB="${DATA_GLOB:-/Users/owner/Desktop/caiatech/datasets/datamix.shuffled.jsonl}"
SHUFFLE_BUFFER_DOCS="${SHUFFLE_BUFFER_DOCS:-2048}"
MODEL_CONFIG="${MODEL_CONFIG:-${BASE_DIR}/configs/onyx.json}"
RUN_NAME="${RUN_NAME:-onyx_64d_1l_mhc_shuf}"
CKPT_DIR="${CKPT_DIR:-${BASE_DIR}/checkpoints/onyx_64d_1l_mhc_shuf}"
VAST_CKPT_DIR="${VAST_CKPT_DIR:-}"
VAST_CKPT="${VAST_CKPT:-}"
INIT_CKPT="${INIT_CKPT:-}"
INIT_CKPT_DIR="${INIT_CKPT_DIR:-}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"
LOG_EVERY="${LOG_EVERY:-100}"
MONITOR_EVERY="${MONITOR_EVERY:-100}"          # 0 = follow log_every
ALERT_EFFECTIVE_VOCAB="${ALERT_EFFECTIVE_VOCAB:-100}"
MEM_REPORT_EVERY="${MEM_REPORT_EVERY:-100}"
PEAK_LR="${PEAK_LR:-1e-4}"
MIN_LR="${MIN_LR:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.002}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
EXPERIMENTAL_MHC="${EXPERIMENTAL_MHC:-1}"
MHC_N="${MHC_N:-4}"
MHC_MODE="${MHC_MODE:-mhc}"
MHC_SINKHORN="${MHC_SINKHORN:-1}"
MHC_SINKHORN_ITERS="${MHC_SINKHORN_ITERS:-10}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
LOCK_DIR="${CKPT_DIR}/.train_autoresume.lock"
LOCK_PID_FILE="${LOCK_DIR}/pid"

if [[ -z "${AMP_DTYPE:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    AMP_DTYPE="float16"
  else
    AMP_DTYPE="bfloat16"
  fi
fi

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

if [[ ! -e "$DATA_GLOB" ]]; then
  echo "DATA_GLOB not found: $DATA_GLOB" >&2
  exit 2
fi
if [[ ! -d "/Users/owner/Desktop/caiatech/datasets/tokenizers/onyx_tokenizer_32k" ]]; then
  echo "Tokenizer dir not found" >&2
  exit 2
fi

mkdir -p "$CKPT_DIR" "$LOG_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [[ -f "$LOCK_PID_FILE" ]]; then
    existing_pid=$(cat "$LOCK_PID_FILE" 2>/dev/null || true)
    if [[ -n "${existing_pid:-}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "Another train_autoresume.sh instance is already running (pid: $existing_pid)." >&2
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

BASE_CMD=(
  python -m onyx.train
  --data_glob "$DATA_GLOB"
  --tokenizer "/Users/owner/Desktop/caiatech/datasets/tokenizers/onyx_tokenizer_32k"
  --model_config "$MODEL_CONFIG"
  --batch_size 4
  --max_seq_len 2048
  --tokens_per_step 16384
  --shuffle_buffer_docs "$SHUFFLE_BUFFER_DOCS"
  --num_epochs 1
  --learning_rate "$PEAK_LR"
  --min_lr "$MIN_LR"
  --warmup_ratio "$WARMUP_RATIO"
  --label_smoothing "$LABEL_SMOOTHING"
  --no-amp
  --save_dir "$CKPT_DIR"
  --save_every_steps 500
  --train_tokens_target 7536720684
  --log_every "$LOG_EVERY"
  --monitor_every "$MONITOR_EVERY"
  --alert_effective_vocab "$ALERT_EFFECTIVE_VOCAB"
  --log_file "${LOG_DIR}/${RUN_NAME}_train.log"
  --mem_report_every "$MEM_REPORT_EVERY"
  --mps_empty_cache_every 1000
  --gc_collect_every 200
  --max_rss_gb 10
  --max_mps_alloc_gb 14
  --auto_interrupt_on_mem
  --dataset_state_mode light
  --no-init_strict
)

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

if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_TRAIN_ARGS)
  BASE_CMD+=("${EXTRA_ARR[@]}")
fi

RETRY_DELAY_SEC=5
MAX_SAME_CKPT_FAILURES=2

last_ckpt=""
same_ckpt_failures=0
child_pid=""
stop_requested=0

on_stop() {
  stop_requested=1
  if [[ -n "$child_pid" ]]; then
    kill -KILL "$child_pid" 2>/dev/null || true
  fi
  exit 130
}
trap on_stop INT TERM HUP TSTP

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
  # Prefer resuming from $CKPT_DIR; if empty, resume from the downloaded Vast.ai
  # checkpoint; if none, warm-start from the latest init checkpoint.
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
    if [[ -n "${VAST_CKPT:-}" && -f "$VAST_CKPT" ]]; then
      ckpt="$VAST_CKPT"
    elif [[ -n "${VAST_CKPT_DIR:-}" && -d "$VAST_CKPT_DIR" ]]; then
      ckpt="$(get_ckpt_list "$VAST_CKPT_DIR" | head -n 1 || true)"
    fi
    if [[ -z "${ckpt:-}" ]]; then
      if [[ -n "${INIT_CKPT:-}" && -f "$INIT_CKPT" ]]; then
        init_ckpt="$INIT_CKPT"
      elif [[ -n "${INIT_CKPT_DIR:-}" && -d "$INIT_CKPT_DIR" ]]; then
        init_ckpt="$(get_ckpt_list "$INIT_CKPT_DIR" | head -n 1 || true)"
      fi
    else
      echo "No local checkpoints found; resuming from: $ckpt"
    fi
  fi

  CMD=("${BASE_CMD[@]}")
  if [[ -n "$ckpt" ]]; then
    CMD+=(--resume "$ckpt")
  elif [[ -n "$init_ckpt" ]]; then
    echo "No checkpoints found; warm-starting from: $init_ckpt"
    CMD+=(--init_checkpoint "$init_ckpt")
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
