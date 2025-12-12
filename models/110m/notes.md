# Onyx 110M

## Parameters
- **Total**: 108,123,078 (108M)
- **Core transformer**: 46,974,918 (47M)
- **Embeddings**: 61,148,160 (61M)

## Architecture
| Component | Value | Change from 80M |
|-----------|-------|-----------------|
| d_model | 448 | +64 |
| n_layers | 7 | +1 |
| n_heads | 7 | +1 |
| n_kv_heads | 2 | same |
| d_ff | 4096 | same |
| vocab_size | 127,872 | same |
| max_seq_len | 8,192 | same |

## Memory System
- memory_size: 2048
- memory_heads: 4
- Hope Attention with delta rule updates

## Growth Strategy
**Nested expansion** from 80M model:
- Existing weights center-padded into larger dimensions
- New layer (layer 6) initialized fresh
- New head dimensions initialized to zero

Growth: 81M → 108M (+27M params, +33% increase)

## Training
- Initialize from grown 80M checkpoint
- Recommended: same hyperparameters as 80M
- May need brief warmup period for new parameters

## Usage
```bash
# Grow from 80M checkpoint
python grow_model.py \
    --source checkpoints/80m.pt \
    --target_config models/110m/config.json \
    --output checkpoints/110m_init.pt

# Continue training
python onyx_train.py \
    --checkpoint checkpoints/110m_init.pt \
    --data_path your_data.jsonl
```

## Checkpoints
- `*_init.pt` - Grown from 80M (pre-training)
- (training checkpoints to follow)

## Notes
First growth step in progressive scaling. Modest increases to all dimensions.
