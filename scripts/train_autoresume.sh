#!/usr/bin/env bash
set -u
set -o pipefail

CKPT_DIR="/Users/owner/Desktop/caiatech/models/onyx/checkpoints"

BASE_CMD=(
  caffeinate -dimsu python -m onyx.train
  --data_glob "/Users/owner/Desktop/caiatech/datasets/caia-chat.jsonl"
  --tokenizer "/Users/owner/Desktop/caiatech/datasets/tokenizers/test_32k"
  --model_config "/Users/owner/Desktop/caiatech/models/onyx/configs/onyx_42m.json"
  --batch_size 1
  --max_seq_len 4096
  --tokens_per_step 16384
  --num_epochs 1
  --warmup_ratio 0.04
  --save_dir "/Users/owner/Desktop/caiatech/models/onyx/checkpoints"
  --save_every_steps 100
  --train_tokens_target 849327647
  --log_every 50
  --log_file "/Users/owner/Desktop/caiatech/models/onyx/logs/onyx_42m_train.log"
  --mem_report_every 50
  --mps_empty_cache_every 50
  --gc_collect_every 200
  --max_rss_gb 10
  --max_mps_alloc_gb 14
  --auto_interrupt_on_mem
  --dataset_state_mode light
)

RETRY_DELAY_SEC=5
MAX_SAME_CKPT_FAILURES=2

last_ckpt=""
same_ckpt_failures=0

while true; do
  ckpts=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && ckpts+=("$line")
  done < <(ls -t "$CKPT_DIR"/checkpoint_step_*.pt 2>/dev/null || true)

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
  "${CMD[@]}"
  status=$?
  end_ts=$(date +%s)
  runtime=$((end_ts - start_ts))

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
