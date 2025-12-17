from onyx_model import Onyx, OnyxConfig, get_param_groups


def test_get_param_groups_includes_dynamic_memory_projection_params():
    cfg = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=32,
        d_ff=64,
        vocab_size=64,
        max_seq_len=16,
        train_seq_len=16,
        use_flash_attention=False,
        gradient_checkpointing=False,
        self_referential_keys=True,
        self_referential_values=True,
    )
    m = Onyx(cfg)
    groups = get_param_groups(m, weight_decay=0.1, memory_lr_scale=0.1)
    assert len(groups) == 3

    memory_group = groups[2]["params"]
    memory_param_ids = {id(p) for p in memory_group}

    for name, p in m.named_parameters():
        if any(k in name for k in ("eta_proj", "alpha_proj")):
            assert id(p) in memory_param_ids

