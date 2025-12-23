# Early Eval Harness

Purpose: fast, objective checks for early training health (loss trends), pipeline correctness, and copy/format milestones.

## Run

```bash
python early_eval.py \
  --checkpoint /path/to/checkpoint.pt \
  --tokenizer /path/to/tokenizer \
  --model_config /path/to/config.json \
  --val_data /path/to/val.jsonl \
  --device mps \
  --dtype float32
```

Reports are written to `early_eval_reports/` in JSON format. Each run prints a short summary and the report path.

## Assets

- `copy_tests.jsonl`: ~50 deterministic copy/format prompts (exact match + edit similarity + JSON parse rate).
- `gen_prompts.jsonl`: ~20 neutral prompts for repetition/diversity metrics.

All prompts use the Onyx chat format:

```
User: ...
Assistant:
```

## Metrics

- `loss.val_loss`: average cross-entropy on the frozen val set (nats).
- `loss.val_ppl`: `exp(val_loss)`.
- `copy_tests.*`: exact match rate + edit similarity + JSON parse success.
- `generation_metrics.*`: unique token ratio, repeat n-gram rates, max run length.
- `sanity.*`: tokenizer/model consistency checks and roundtrip stability.

## Alerts

- `ALERT_DIVERGENCE`: val loss NaN/Inf or jumps above threshold.
- `ALERT_LOOPING`: repeat_3gram_rate above threshold.
- `ALERT_COLLAPSE`: unique_token_ratio below threshold.
- `ALERT_TOKENIZER_MISMATCH`: vocab/special-token/roundtrip checks failed.

Thresholds are configurable via CLI flags.
