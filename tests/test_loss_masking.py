import torch

from onyx.experimental import OnyxMHC
from onyx.model import Onyx, OnyxConfig


def _tiny_config() -> OnyxConfig:
    return OnyxConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        d_ff=64,
        vocab_size=64,
        max_seq_len=16,
        train_seq_len=16,
        use_flash_attention=False,
        gradient_checkpointing=False,
        dropout=0.0,
        attention_dropout=0.0,
        use_cms_ffn=False,
    )


def _assert_all_ignored_loss(model: torch.nn.Module) -> None:
    cfg = model.config
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    labels = torch.full_like(input_ids, -100)
    out = model(input_ids=input_ids, labels=labels)
    loss = out["loss"]
    assert loss is not None
    assert torch.isfinite(loss).all()
    assert loss.item() == 0.0


def test_loss_all_ignored_is_finite_onyx():
    torch.manual_seed(0)
    model = Onyx(_tiny_config()).eval()
    _assert_all_ignored_loss(model)


def test_loss_all_ignored_is_finite_mhc():
    torch.manual_seed(0)
    model = OnyxMHC(_tiny_config(), mhc_n=2, mhc_mode="mhc", mhc_sinkhorn=True, mhc_sinkhorn_iters=2).eval()
    _assert_all_ignored_loss(model)
