# Onyx

An experimental language model with Hope Attention and persistent memory, developed in public by [Caia Tech](https://github.com/Caia-Tech).

## Architecture

Onyx is a transformer-based language model featuring:

- **Hope Attention** - Self-modifying K/V memory using delta rule updates
- **Persistent Memory** - Memory that carries across inference sessions
- **CMS FFN** - Chunked Multi-Scale Feed-Forward Networks
- **GQA** - Grouped Query Attention for memory efficiency
- **M3 Optimizer** - Multi-scale Momentum optimizer

### Current Model (v0.1)

| Parameter | Value |
|-----------|-------|
| d_model | 384 |
| n_layers | 6 |
| n_heads | 6 |
| Parameters | ~11M (excluding embeddings) |
| Context | 4096 tokens |
| Tokenizer | Llama-3 (128k vocab) |

## Development

This project is being developed in public. The goal is to explore nested/progressive model growth - starting small and incrementally scaling while preserving learned representations.

### Roadmap

- [x] Base 11M model architecture
- [x] Training pipeline
- [x] Inference with memory modes
- [ ] Progressive model scaling (11M -> 16M -> 24M -> ...)
- [ ] Inference-time learning
- [ ] Memory persistence across sessions

## Usage

### Training

```bash
python onyx_train.py \
  --batch_size 16 \
  --max_seq_len 2048 \
  --num_epochs 1 \
  --data_glob your_data.jsonl \
  --train_tokens_target 102080512
```

### Inference

```bash
# Interactive chat
python onyx_inference.py --checkpoint checkpoints/your_checkpoint.pt

# With learning mode (memory updates during inference)
python onyx_inference.py --checkpoint checkpoints/your_checkpoint.pt --learning

# Persistent memory (saves/loads between sessions)
python onyx_inference.py --checkpoint checkpoints/your_checkpoint.pt --memory persistent --memory_path memory.pt
```

### Memory Modes

- `stateless` - Fresh memory each call (standard LLM behavior)
- `session` - Memory persists within conversation
- `persistent` - Memory saved/loaded across sessions

## Files

- `onyx_model.py` - Model architecture
- `onyx_train.py` - Training script
- `onyx_inference.py` - Inference and chat interface
- `archive/legacy/` - Previous experimental versions

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers
- flash-attn (optional, for faster attention)

## License

Do whatever you want

## Author

Marvin Tutt, Caia Tech
