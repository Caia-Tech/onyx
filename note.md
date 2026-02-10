# Training Notes (RMT/mHC/mHC-lite)

Date: 2026-02-10

## Summary
- Vast H100 run with `RMT + mHC(n=4) + sinkhorn_iters=10` is stable at `batch_size=4`, but expensive.
- Local Mac (MPS) run with small model and `RMT + mHC-lite` is stable and unexpectedly faster than matched no-RMT/no-mHC baseline in this environment.
- Resume/autoresume behavior is working and restores dataset state, RNG, optimizer, and step.

## Key Findings

### 1) CUDA/H100 behavior
- `bs=16` on large RMT+mHC config OOMs.
- OOMs were also caused by multiple concurrent train processes; single-process discipline is required.
- With `RMT + mHC(n=4) + sinkhorn=10`, measured throughput around `~3.2k tok/s` (stable long-run profile at `bs=4`).
- With lighter non-RMT configuration, throughput was much higher (`~14k tok/s`) on H100.

Interpretation:
- At larger model/sequence settings (`seq_len=2048`), full mHC+Sinkhorn overhead dominates.
- Cost scales strongly with `mhc_n` and Sinkhorn iterations.

### 2) Label smoothing bug (important)
- `label_smoothing > 0` caused CUDA device-side assert in gather path due to ignored label (`-100`) handling.
- Temporary mitigation used during runs: `--label_smoothing 0.0`.
- In mHC model path, safe-label handling was introduced to avoid gather-on-ignored-index crashes.

### 3) Resume/autoresume sanity
- Resume from checkpoint correctly reported:
  - step/tokens restored
  - memory states restored
  - RNG restored
  - dataset state loaded (with `dataset_state_mode=light`)
- `light` mode may replay a small boundary region; expected.

### 4) Mac (MPS) small-model A/B
Matched comparison:
- Same dims: `d_model=192, n_layers=8, n_heads=6, n_kv_heads=2, d_ff=3072, vocab=32168, seq=1024`
- Same training knobs (`batch_size=4`, `tokens_per_step=16384`, fp32 MPS)

A: `RMT + mHC-lite (n=2)`
- Config: `configs/onyx_mac_rmt_mhc_lite_4m.json`
- 120-step run completed.
- Steady region (`step>=30`) tok/s:
  - median `3652.5`
  - mean `3627.1`

B: no RMT + no mHC
- Config: `configs/onyx_mac_no_rmt_matched_4m.json`
- Repeatedly slowed/stalled on MPS after early steps.
- Observed tok/s points: `2494`, `2540`, one degraded `945`.

Interpretation:
- On this Mac/MPS setup, the no-RMT/no-mHC path appears to hit less favorable runtime behavior.
- Extra theory FLOPs do not guarantee lower wall time; backend kernel/path effects dominate.

## mHC-lite Implementation (repo status)
- Added new mode `mhc_lite` as a real third mode (not just low Sinkhorn iters):
  - `onyx/experimental/mhc.py`: `MHCLiteMixer`
  - `onyx/experimental/onyx_mhc_model.py`: mode dispatch includes `mhc_lite`
  - `onyx/train.py`: CLI supports `--mhc_mode mhc_lite`
  - `onyx/inference.py`: CLI supports `--mhc_mode mhc_lite`
- Logging updated: sinkhorn flags are explicitly ignored in `mhc_lite`.
- Targeted tests for `mhc_lite` pass.

## Local Autoresume Defaults (separate Mac profile)
- `scripts/train_autoresume.sh` now defaults to a separate local profile:
  - model config: `configs/onyx_mac_rmt_mhc_lite_4m.json`
  - run/checkpoint naming separated from Vast runs
  - default `MHC_MODE=mhc_lite`, `MHC_N=2`
  - env overrides added for batch/seq/tokens/max steps/etc.
- Added matched no-RMT config for clean A/B:
  - `configs/onyx_mac_no_rmt_matched_4m.json`

## Practical Recommendations

### CUDA/H100
- For speed/stability:
  1. try `mhc_mode=mhc_lite`
  2. reduce `mhc_n` to 2
  3. avoid high Sinkhorn iteration configs unless needed
- Keep only one training process active at a time.

### Mac/MPS
- Use the separate small profile for development and fast iteration.
- Benchmark using >=100 steps; tiny runs (5-10 steps) are too noisy.

## Useful Run References
- Vast run dir:
  - `/workspace/checkpoints/onyx_rmt_mhc4_h100_ls0_bs4`
- Mac A/B logs:
  - `logs/ab_rmt_mhc_lite_train.log`
  - `logs/ab_no_rmt_no_mhc_train.log`
  - `logs/ab_no_rmt_no_mhc_short_train.log`

