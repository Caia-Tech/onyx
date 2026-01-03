# Experimental mHC / Hyper-Connections

This folder contains an experimental Onyx model variant that expands the residual stream into multiple streams
and mixes them per layer with Hyper-Connections (HC) or doubly-stochastic mHC (Sinkhorn projection).

## How to run

Baseline training (existing behavior):
```
python -m onyx.train --data_glob ./data.jsonl --save_dir ./checkpoints
```

Experimental mHC training:
```
python -m onyx.train --data_glob ./data.jsonl --save_dir ./checkpoints --experimental_mhc \
  --mhc_n 2 --mhc_mode mhc --mhc_sinkhorn_iters 10
```

## Recommended starting params
- `--mhc_n 2`
- `--mhc_mode mhc`
- `--mhc_sinkhorn_iters 10`

## Notes
- Expect a throughput hit vs the baseline model due to extra streams and per-layer mixing.
