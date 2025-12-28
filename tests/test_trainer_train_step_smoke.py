import torch
from torch.utils.data import DataLoader

from onyx.model import Onyx, OnyxConfig
from onyx.train import Trainer, TrainingConfig, StreamingPackedDataset, collate_onyx


class SpyOnyx(Onyx):
    def __init__(self, config):
        super().__init__(config)
        self.last_forward = None

    def forward(self, *args, **kwargs):
        self.last_forward = dict(kwargs)
        return super().forward(*args, **kwargs)


def test_train_step_runs_and_builds_combined_cu_seqlens(dummy_tokenizer, tiny_jsonl):
    cfg = TrainingConfig(
        data_glob="unused.jsonl",
        max_seq_len=16,
        batch_size=2,
        tokens_per_step=32,
        pack_sequences=True,
        use_amp=False,
    )
    t = Trainer(cfg)

    # Force the varlen path (device_type == "cuda") while staying CPU-only.
    t.device = torch.device("cpu")
    t.device_type = "cuda"
    t.use_autocast = False
    t.autocast_dtype = torch.float32

    model = SpyOnyx(
        OnyxConfig(
            d_model=32,
            n_layers=1,
            n_heads=1,
            n_kv_heads=1,
            head_dim=32,
            d_ff=64,
            vocab_size=64,
            max_seq_len=cfg.max_seq_len,
            train_seq_len=cfg.max_seq_len,
            use_flash_attention=False,
            gradient_checkpointing=False,
        )
    ).to(t.device)

    t.model = model
    t.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t.scaler = None
    t.accumulation_steps = 1
    t.global_step = 0
    t.tokens_seen = 0
    t.best_loss = float("inf")

    ds = StreamingPackedDataset(
        data_glob=tiny_jsonl,
        tokenizer=dummy_tokenizer,
        max_seq_len=cfg.max_seq_len,
        pack=True,
        seed=123,
        drop_remainder=False,
        emit_cu_seqlens=True,
    )
    t.dataloader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0, pin_memory=False, collate_fn=collate_onyx)
    t.data_iter = iter(t.dataloader)

    metrics, _memory_states = t.train_step(memory_states=None)
    assert metrics is not None
    assert metrics["tokens"] == cfg.batch_size * cfg.max_seq_len
    assert "loss" in metrics

    cu = model.last_forward["cu_seqlens"]
    assert cu is not None
    assert cu.dtype in (torch.int32, torch.int64)
    assert cu[0].item() == 0
    assert cu[-1].item() == cfg.batch_size * cfg.max_seq_len
    assert torch.all(cu[1:] >= cu[:-1])
    assert model.last_forward["max_seqlen"] == cfg.max_seq_len
