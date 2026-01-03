"""Utilities for enforcing fp32 when AMP is disabled."""

from __future__ import annotations

import torch
import torch.nn as nn


def enforce_fp32_everywhere(module: nn.Module) -> None:
    """Ensure all floating parameters and buffers are fp32 in-place."""
    for _, param in module.named_parameters(recurse=True):
        if param.is_floating_point() and param.dtype != torch.float32:
            param.data = param.data.to(dtype=torch.float32)

    for mod in module.modules():
        for name, buf in mod.named_buffers(recurse=False):
            if buf is None:
                continue
            if buf.is_floating_point() and buf.dtype != torch.float32:
                setattr(mod, name, buf.to(dtype=torch.float32))
