import torch
import torch.nn as nn
from onyx.train import CMSFrequencyManager
from onyx.model import OnyxConfig


class DummyCMS(nn.Module):
    def __init__(self):
        super().__init__()
        # mimic structure: ffn.level_ffns.<i>.<param>
        self.ffn = nn.Module()
        self.ffn.level_ffns = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)])


def test_cms_frequency_manager_masks_grads_on_schedule():
    model = DummyCMS()
    cfg = OnyxConfig(
        use_cms_ffn=True,
        cms_update_every_base_steps=2,
        cms_update_every_multiplier=2,
    )
    mgr = CMSFrequencyManager(model, cfg)

    # give all params grads
    for p in model.parameters():
        p.grad = torch.ones_like(p.data)

    # step=1: only levels with update_every dividing 1 should update
    # level0 update_every=2, level1=4, level2=8 => none should update at step=1
    mgr.mask_gradients(global_step=1)
    assert all(p.grad is None for p in model.parameters())

    # restore grads
    for p in model.parameters():
        p.grad = torch.ones_like(p.data)

    # step=2: level0 updates, level1 and level2 masked
    mgr.mask_gradients(global_step=2)

    # level0 params should still have grads
    assert model.ffn.level_ffns[0].weight.grad is not None
    assert model.ffn.level_ffns[1].weight.grad is None
    assert model.ffn.level_ffns[2].weight.grad is None
