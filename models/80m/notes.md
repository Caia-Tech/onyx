# Onyx 80M

## Parameters
- **Total**: 81,221,052 (81M)
- **Core transformer**: 32,136,252 (32M)
- **Embeddings**: 49,084,800 (49M)

## Architecture
| Component | Value |
|-----------|-------|
| d_model | 384 |
| n_layers | 6 |
| n_heads | 6 |
| n_kv_heads | 2 (GQA) |
| d_ff | 4096 |
| vocab_size | 127,872 |
| max_seq_len | 8,192 |

## Memory System
- memory_size: 2048
- memory_heads: 4
- Hope Attention with delta rule updates

## Training
- Dataset: ~102M tokens
- Batch size: 16
- Sequence length: 2048
- Tokens per step: 32,768

## Checkpoints
- `checkpoint_8500.pt` - Latest (~70% through training)

## Notes
Base model for nested growth experiments. Uses Llama-3 tokenizer (Hermes-2-Pro-Llama-3-8B).
