# Onyx

A language model with Hope Attention and persistent memory, developed in public by [Caia Tech](https://github.com/Caia-Tech).

## Logging and Memory Safety

Recommended run command (macOS-friendly):

```bash
python -m onyx.train ... --log_file /path/train.log --tee --mem_report_every 50 --mps_empty_cache_every 50
```

Note: On macOS, Terminal/iTerm scrollback can grow memory quickly during long runs. Use `--log_file` (and `--tee` if you want console output) or reduce scrollback limits.

## License

Do whatever you want

## Author

Marvin Tutt, Caia Tech
