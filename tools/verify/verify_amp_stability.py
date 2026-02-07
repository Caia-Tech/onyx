#!/usr/bin/env python3
import argparse
import sys
from typing import List

import torch

REPO_ROOT = "/Users/owner/Desktop/caiatech/models/onyx"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from onyx.experimental.onyx_mhc_model import OnyxMHC
from onyx.model import OnyxConfig


def _select_devices(requested: str) -> List[str]:
    if requested != "auto":
        return [requested]
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def _autocast_ctx(device: torch.device, use_amp: bool):
    if not use_amp:
        return torch.autocast(device_type=device.type, enabled=False)
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def _make_scaler(device: torch.device, use_amp: bool):
    if not use_amp:
        return None
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    if device.type == "mps":
        try:
            return torch.amp.GradScaler("mps", init_scale=2.0**10, growth_interval=2000)
        except Exception:
            return None
    return None


def _build_model(device: torch.device) -> OnyxMHC:
    cfg = OnyxConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        max_seq_len=32,
        train_seq_len=32,
        vocab_size=256,
        use_flash_attention=False,
        dropout=0.0,
        attention_dropout=0.0,
        use_hope_attention=True,
    )
    model = OnyxMHC(
        cfg,
        mhc_n=2,
        mhc_mode="mhc",
        mhc_sinkhorn=True,
        mhc_sinkhorn_iters=8,
    ).to(device)
    return model


def _forward_only(device: torch.device, use_amp: bool, steps: int) -> None:
    torch.manual_seed(0)
    model = _build_model(device)
    model.eval()
    for _ in range(steps):
        input_ids = torch.randint(0, 256, (2, 32), device=device)
        labels = input_ids.clone()
        with _autocast_ctx(device, use_amp):
            out = model(input_ids=input_ids, labels=labels)
        logits = out["logits"]
        loss = out["loss"]
        assert torch.isfinite(logits).all(), "non-finite logits"
        if loss is not None:
            assert torch.isfinite(loss).all(), "non-finite loss"


def _train_steps(device: torch.device, use_amp: bool, steps: int) -> int:
    torch.manual_seed(0)
    model = _build_model(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = _make_scaler(device, use_amp)
    skipped = 0

    for _ in range(steps):
        input_ids = torch.randint(0, 256, (2, 32), device=device)
        labels = input_ids.clone()
        opt.zero_grad(set_to_none=True)

        with _autocast_ctx(device, use_amp):
            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"]

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()

        finite = True
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                finite = False
                break
        if not finite:
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            continue

        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

    return skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--forward_steps", type=int, default=50)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    devices = _select_devices(args.device)
    for dev in devices:
        device = torch.device(dev)
        print(f"== device={dev} amp={args.amp} ==")
        _forward_only(device, args.amp, args.forward_steps)
        skipped = _train_steps(device, args.amp, args.train_steps)
        print(f"skipped_steps={skipped}")
        print("OK")


if __name__ == "__main__":
    main()
