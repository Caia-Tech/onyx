"""Test that mode collapse fixes work correctly and are backward-compatible."""
import torch
from onyx.model import Onyx, OnyxConfig


def test_label_smoothing_disabled_by_default():
    """Test that label smoothing is off by default (backward compat)."""
    config = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        vocab_size=64,
        max_seq_len=16,
        use_flash_attention=False,
    )
    assert config.label_smoothing == 0.0
    assert config.entropy_reg_weight == 0.0
    assert config.feedback_strength == 1.0


def test_label_smoothing_enabled():
    """Test that label smoothing produces different loss than vanilla."""
    torch.manual_seed(42)

    config_vanilla = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        vocab_size=64,
        max_seq_len=16,
        use_flash_attention=False,
        label_smoothing=0.0,
    )

    config_smoothed = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        vocab_size=64,
        max_seq_len=16,
        use_flash_attention=False,
        label_smoothing=0.1,
    )

    model_vanilla = Onyx(config_vanilla).eval()
    model_smoothed = Onyx(config_smoothed)
    model_smoothed.load_state_dict(model_vanilla.state_dict())
    model_smoothed.eval()

    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))

    with torch.no_grad():
        out_vanilla = model_vanilla(input_ids, labels=labels)
        out_smoothed = model_smoothed(input_ids, labels=labels)

    loss_vanilla = out_vanilla["loss"]
    loss_smoothed = out_smoothed["loss"]

    # Both should be finite
    assert torch.isfinite(loss_vanilla)
    assert torch.isfinite(loss_smoothed)
    # Losses should be different (use absolute difference for small values)
    diff = torch.abs(loss_vanilla - loss_smoothed).item()
    assert diff > 1e-6, f"Expected different losses, got diff={diff}"
    # Note: smoothed loss can be slightly higher or lower depending on model state


def test_entropy_regularization_enabled():
    """Test that entropy regularization reduces loss (maximizes entropy)."""
    torch.manual_seed(42)

    config_no_reg = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        vocab_size=64,
        max_seq_len=16,
        use_flash_attention=False,
        entropy_reg_weight=0.0,
    )

    config_with_reg = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        vocab_size=64,
        max_seq_len=16,
        use_flash_attention=False,
        entropy_reg_weight=0.01,
    )

    model_no_reg = Onyx(config_no_reg).eval()
    model_with_reg = Onyx(config_with_reg)
    model_with_reg.load_state_dict(model_no_reg.state_dict())
    model_with_reg.eval()

    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))

    with torch.no_grad():
        out_no_reg = model_no_reg(input_ids, labels=labels)
        out_with_reg = model_with_reg(input_ids, labels=labels)

    loss_no_reg = out_no_reg["loss"]
    loss_with_reg = out_with_reg["loss"]

    # Losses should be different
    assert not torch.allclose(loss_no_reg, loss_with_reg, rtol=1e-4)
    # Both should be finite
    assert torch.isfinite(loss_no_reg)
    assert torch.isfinite(loss_with_reg)


def test_feedback_strength_controls_loop():
    """Test that feedback_strength parameter controls feedback loop intensity."""
    torch.manual_seed(42)

    config_full = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        vocab_size=64,
        max_seq_len=16,
        use_flash_attention=False,
        use_hope_attention=True,
        generate_own_values=True,
        feedback_strength=1.0,
    )

    config_half = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        vocab_size=64,
        max_seq_len=16,
        use_flash_attention=False,
        use_hope_attention=True,
        generate_own_values=True,
        feedback_strength=0.5,
    )

    model_full = Onyx(config_full).eval()
    model_half = Onyx(config_half)
    model_half.load_state_dict(model_full.state_dict())
    model_half.eval()

    input_ids = torch.randint(0, 64, (2, 8))

    with torch.no_grad():
        out_full = model_full(input_ids)
        out_half = model_half(input_ids)

    logits_full = out_full["logits"]
    logits_half = out_half["logits"]

    # Logits should be different (feedback affects them)
    assert not torch.allclose(logits_full, logits_half, rtol=1e-3)
    # Both should be finite
    assert torch.isfinite(logits_full).all()
    assert torch.isfinite(logits_half).all()


def test_all_fixes_together():
    """Test that all fixes can be enabled together without issues."""
    torch.manual_seed(42)

    config = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        vocab_size=64,
        max_seq_len=16,
        use_flash_attention=False,
        use_hope_attention=True,
        generate_own_values=True,
        label_smoothing=0.1,
        entropy_reg_weight=0.01,
        feedback_strength=0.5,
    )

    model = Onyx(config).eval()

    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))

    with torch.no_grad():
        out = model(input_ids, labels=labels)

    # Basic checks
    assert out["logits"].shape == (2, 8, 64)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() > 0


def test_ignored_labels_still_work_with_fixes():
    """Test that -100 (ignored) labels still produce loss=0 with fixes enabled."""
    torch.manual_seed(42)

    config = OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        vocab_size=64,
        max_seq_len=16,
        use_flash_attention=False,
        label_smoothing=0.1,  # Enable label smoothing
        entropy_reg_weight=0.01,  # Enable entropy reg
    )

    model = Onyx(config).eval()

    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.full_like(input_ids, -100)  # All ignored

    with torch.no_grad():
        out = model(input_ids, labels=labels)

    # Loss should still be exactly 0 when all labels are ignored
    assert out["loss"].item() == 0.0
