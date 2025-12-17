import torch
import torch.nn as nn

from onyx_model import OnyxConfig, CMSFFN
from onyx_train import CMSFrequencyManager


class TinyCMSModel(nn.Module):
    def __init__(self, cfg: OnyxConfig):
        super().__init__()
        self.ffn = CMSFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x).sum()


def test_cms_frequency_manager_masks_slow_level_grads_and_skips_update():
    torch.manual_seed(0)
    cfg = OnyxConfig(
        d_model=16,
        d_ff=48,
        dropout=0.0,
        use_cms_ffn=True,
        cms_num_levels=3,
        cms_base_chunk=1,
        cms_chunk_multiplier=2,
    )
    model = TinyCMSModel(cfg)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    mgr = CMSFrequencyManager(model, cfg)

    x = torch.randn(2, 8, cfg.d_model)
    loss = model(x)
    loss.backward()

    assert any(p.grad is not None for n, p in model.named_parameters() if "level_ffns.1" in n)
    assert any(p.grad is not None for n, p in model.named_parameters() if "level_ffns.0" in n)

    # Step=1 should update level 0 (every 1) and skip level 1 (every 2) and level 2 (every 4).
    mgr.mask_gradients(global_step=1)
    assert all(p.grad is None for n, p in model.named_parameters() if "level_ffns.1" in n)
    assert all(p.grad is None for n, p in model.named_parameters() if "level_ffns.2" in n)
    assert any(p.grad is not None for n, p in model.named_parameters() if "level_ffns.0" in n)

    before = {n: p.detach().clone() for n, p in model.named_parameters() if "level_ffns." in n}
    opt.step()

    changed_level0 = any(
        not torch.allclose(model.state_dict()[n], before[n], atol=0.0, rtol=0.0)
        for n in before.keys()
        if "level_ffns.0" in n
    )
    unchanged_level1 = all(
        torch.allclose(model.state_dict()[n], before[n], atol=0.0, rtol=0.0)
        for n in before.keys()
        if "level_ffns.1" in n
    )
    assert changed_level0
    assert unchanged_level1

