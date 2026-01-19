#!/usr/bin/env bash
set -u
set -o pipefail

if [[ -z "${CAFFEINATED:-}" ]]; then
  export CAFFEINATED=1
  exec caffeinate -dimsu -- "$0" "$@"
fi

BASE_DIR="/Users/owner/Desktop/caiatech/models/onyx"
CKPT_DIR="${BASE_DIR}/checkpoints_mhc"
LOG_DIR="${BASE_DIR}/logs"
LOCK_DIR="${CKPT_DIR}/.train_autoresume.lock"
LOCK_PID_FILE="${LOCK_DIR}/pid"

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
  --data_glob "/Users/owner/Desktop/caiatech/datasets/onyx-dataset.jsonl"
  --tokenizer "/Users/owner/Desktop/caiatech/datasets/tokenizer"
  --model_config "/Users/owner/Desktop/caiatech/models/onyx/configs/onyx_1m.json"
  --batch_size 4
  --max_seq_len 512
  --tokens_per_step 8192
  --num_epochs 1
  --warmup_ratio 0.005
  --save_dir "/Users/owner/Desktop/caiatech/models/onyx/checkpoints_mhc"
  --save_every_steps 5000
  --train_tokens_target 3146456434
  --log_every 1000
  --log_file "/Users/owner/Desktop/caiatech/models/onyx/logs/onyx_1m_mhc_train.log"
  --mem_report_every 1000
  --mps_empty_cache_every 1000
  --gc_collect_every 200
  --max_rss_gb 10
  --max_mps_alloc_gb 14
  --auto_interrupt_on_mem
  --dataset_state_mode light
  --experimental_mhc
  --mhc_n 2
  --mhc_mode mhc
  --mhc_sinkhorn_iters 10
)

RETRY_DELAY_SEC=5
MAX_SAME_CKPT_FAILURES=2

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
  local f base step
  for f in "$CKPT_DIR"/checkpoint_step_*.pt "$CKPT_DIR"/interrupt_step_*.pt; do
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
  done < <(get_ckpt_list || true)

  ckpt=""
  if [[ ${#ckpts[@]} -gt 0 ]]; then
    ckpt="${ckpts[0]}"
    if [[ "$ckpt" == "$last_ckpt" && $same_ckpt_failures -ge $MAX_SAME_CKPT_FAILURES && ${#ckpts[@]} -gt 1 ]]; then
      ckpt="${ckpts[1]}"
      echo "Latest checkpoint failed $same_ckpt_failures times; trying previous: $ckpt"
    fi
  fi

  CMD=("${BASE_CMD[@]}")
  if [[ -n "$ckpt" ]]; then
    CMD+=(--resume "$ckpt")
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
