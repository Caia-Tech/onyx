import json
from pathlib import Path

import torch

import onyx.train as onyx_train


class DummyAutoTokenizer:
    @staticmethod
    def from_pretrained(_name, trust_remote_code=True):
        _ = trust_remote_code

        class T:
            eos_token_id = 2
            bos_token_id = 1
            pad_token_id = 2

            def __len__(self):
                return 64

            def encode(self, text, add_special_tokens=False):
                _ = add_special_tokens
                n = max(1, min(len(text), 12))
                return list(range(10, 10 + n))

        return T()


def _write_tiny_model_config(path: Path, vocab_size: int = 64):
    cfg = {
        "architecture": {
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 1,
            "n_kv_heads": 1,
            "head_dim": 32,
            "d_ff": 64,
            "vocab_size": vocab_size,
            "rope_theta": 10000.0,
            "norm_eps": 1e-5,
        }
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")


def test_trainer_setup_and_train_two_steps_offline(monkeypatch, tmp_path: Path, tiny_jsonl):
    # Force CPU selection.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    monkeypatch.setattr(onyx_train, "TRANSFORMERS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(onyx_train, "AutoTokenizer", DummyAutoTokenizer, raising=False)

    cfg_path = tmp_path / "model.json"
    _write_tiny_model_config(cfg_path, vocab_size=64)

    save_dir = tmp_path / "ckpts"
    tc = onyx_train.TrainingConfig(
        data_glob=tiny_jsonl,
        tokenizer_name="dummy",
        model_config_path=str(cfg_path),
        max_seq_len=16,
        pack_sequences=True,
        shuffle_buffer_docs=0,
        batch_size=2,
        tokens_per_step=32,
        num_epochs=1,
        max_steps=2,
        use_amp=False,
        use_torch_compile=False,
        use_m3_optimizer=False,
        log_every=1,
        save_dir=str(save_dir),
        save_every_steps=0,
        keep_last_n=2,
        wandb_project=None,
    )

    t = onyx_train.Trainer(tc)
    t.train()

    # Final checkpoint should exist.
    final = list(Path(save_dir).glob("checkpoint_final_step_*.pt"))
    assert final


def test_train_main_parses_args_and_dry_run_without_network(monkeypatch, tmp_path: Path, tiny_jsonl):
    # Patch Trainer.train to avoid actually running; we just want main() coverage.
    seen = {}

    def _train(self):
        seen["cfg"] = self.config

    monkeypatch.setattr(onyx_train.Trainer, "train", _train, raising=True)

    argv = [
        "scripts/train.py",
        "--data_glob",
        tiny_jsonl,
        "--save_dir",
        str(tmp_path / "out"),
        "--dry_run",
    ]
    monkeypatch.setattr(onyx_train.sys, "argv", argv, raising=False)
    onyx_train.main()

    cfg = seen["cfg"]
    assert cfg.max_steps == 3
    assert cfg.save_every_steps == 0
    assert cfg.wandb_project is None


def test_train_bench_mode_saves_bench_checkpoint(monkeypatch, tmp_path: Path, tiny_jsonl):
    # Force CPU selection and stub tokenizer to stay offline.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    monkeypatch.setattr(onyx_train, "TRANSFORMERS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(onyx_train, "AutoTokenizer", DummyAutoTokenizer, raising=False)

    cfg_path = tmp_path / "model.json"
    _write_tiny_model_config(cfg_path, vocab_size=64)

    save_dir = tmp_path / "bench_ckpts"
    tc = onyx_train.TrainingConfig(
        data_glob=tiny_jsonl,
        tokenizer_name="dummy",
        model_config_path=str(cfg_path),
        max_seq_len=16,
        pack_sequences=True,
        shuffle_buffer_docs=0,
        batch_size=2,
        tokens_per_step=32,
        use_amp=False,
        use_torch_compile=False,
        use_m3_optimizer=False,
        log_every=1,
        save_dir=str(save_dir),
        save_every_steps=0,
        keep_last_n=2,
        bench_steps=1,
        disable_saves_during_bench=False,
        wandb_project=None,
    )

    t = onyx_train.Trainer(tc)
    t.train()

    bench = list(Path(save_dir).glob("checkpoint_bench_step_*.pt"))
    assert bench
