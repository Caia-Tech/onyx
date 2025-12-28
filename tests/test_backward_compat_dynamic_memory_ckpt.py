import torch

from onyx.model import OnyxConfig, ChunkedLinearDeltaMemory


def test_old_state_dict_without_dynamic_hyperparams_loads_strict():
    cfg = OnyxConfig(
        d_model=8,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=8,
        d_ff=16,
        vocab_size=32,
        max_seq_len=16,
        train_seq_len=16,
        use_flash_attention=False,
        gradient_checkpointing=False,
        memory_chunk_size=2,
    )
    m1 = ChunkedLinearDeltaMemory(d_in=8, d_out=4, config=cfg)
    sd = m1.state_dict()
    sd = {k: v for k, v in sd.items() if "eta_proj" not in k and "alpha_proj" not in k}

    m2 = ChunkedLinearDeltaMemory(d_in=8, d_out=4, config=cfg)
    m2.load_state_dict(sd, strict=True)

