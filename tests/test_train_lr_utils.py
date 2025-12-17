import torch

from onyx_train import TrainingConfig, get_lr, set_lr


def test_get_lr_warmup_and_cosine_decay_bounds():
    cfg = TrainingConfig()
    cfg.learning_rate = 1e-3
    cfg.min_lr = 1e-4
    cfg.warmup_steps = 4
    total_steps = 20

    # Warmup ramps up to learning_rate at step warmup_steps-1
    assert abs(get_lr(0, cfg, total_steps) - (cfg.learning_rate / cfg.warmup_steps)) < 1e-12
    assert abs(get_lr(cfg.warmup_steps - 1, cfg, total_steps) - cfg.learning_rate) < 1e-12

    # End approaches min_lr.
    assert get_lr(total_steps, cfg, total_steps) >= cfg.min_lr
    assert abs(get_lr(total_steps, cfg, total_steps) - cfg.min_lr) < 1e-12


def test_set_lr_applies_lr_scale_per_group():
    p1 = torch.nn.Parameter(torch.zeros(()))
    p2 = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.SGD(
        [
            {"params": [p1], "lr_scale": 1.0},
            {"params": [p2], "lr_scale": 0.1},
        ],
        lr=1.0,
    )
    set_lr(opt, 2.0)
    assert opt.param_groups[0]["lr"] == 2.0
    assert abs(opt.param_groups[1]["lr"] - 0.2) < 1e-12
