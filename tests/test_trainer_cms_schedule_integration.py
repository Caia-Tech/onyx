import torch
from torch.utils.data import DataLoader

from onyx.model import Onyx, OnyxConfig
from onyx.train import Trainer, TrainingConfig, StreamingPackedDataset, collate_onyx


def test_trainer_train_step_respects_cms_update_schedule(dummy_tokenizer, tiny_jsonl):
    cfg = TrainingConfig(
        data_glob="unused.jsonl",
        max_seq_len=16,
        batch_size=2,
        tokens_per_step=32,
        pack_sequences=True,
        use_amp=False,
        gradient_clip=0.0,
    )
    t = Trainer(cfg)
    t.device = torch.device("cpu")
    t.device_type = "cpu"
    t.use_autocast = False
    t.autocast_dtype = torch.float32

    model_cfg = OnyxConfig(
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
        use_cms_ffn=True,
        cms_num_levels=2,
        cms_base_chunk=1,
        cms_chunk_multiplier=2,
    )
    t.model = Onyx(model_cfg).to(t.device)
    t.optimizer = torch.optim.SGD(t.model.parameters(), lr=0.1)
    t.scaler = None
    t.accumulation_steps = 1
    t.tokens_seen = 0
    t.best_loss = float("inf")

    ds = StreamingPackedDataset(
        data_glob=tiny_jsonl,
        tokenizer=dummy_tokenizer,
        max_seq_len=cfg.max_seq_len,
        pack=True,
        seed=123,
        drop_remainder=False,
        emit_cu_seqlens=False,
    )
    t.dataloader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0, pin_memory=False, collate_fn=collate_onyx)
    t.data_iter = iter(t.dataloader)

    # global_step=1 -> level 1 has update_every=2 -> should be skipped.
    t.global_step = 1

    before = {n: p.detach().clone() for n, p in t.model.named_parameters() if "ffn.level_ffns.1" in n}
    metrics, _mem = t.train_step(memory_states=None)
    assert metrics is not None

    after = {n: p.detach().clone() for n, p in t.model.named_parameters() if "ffn.level_ffns.1" in n}
    assert before.keys() == after.keys()
    assert all(torch.allclose(before[k], after[k], atol=0.0, rtol=0.0) for k in before.keys())

