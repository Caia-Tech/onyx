import torch

from onyx.experimental import OnyxMHC
from onyx.fp32_utils import enforce_fp32_everywhere
from onyx.model import Onyx, OnyxConfig


def _tiny_config() -> OnyxConfig:
    return OnyxConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        d_ff=64,
        vocab_size=64,
        max_seq_len=32,
        train_seq_len=32,
        memory_chunk_size=8,
        use_flash_attention=False,
        gradient_checkpointing=False,
        dropout=0.0,
        attention_dropout=0.0,
        use_cms_ffn=False,
    )


def _force_fp16_param(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.is_floating_point():
            param.data = param.data.to(dtype=torch.float16)
            return


def _force_fp16_buffer(model: torch.nn.Module) -> None:
    for mod in model.modules():
        for name, buf in mod.named_buffers(recurse=False):
            if buf is not None and buf.is_floating_point():
                setattr(mod, name, buf.to(dtype=torch.float16))
                return
    model.register_buffer("fp16_probe", torch.randn(3, dtype=torch.float16))


def _iter_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_tensors(value)


def _assert_model_fp32(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.is_floating_point():
            assert param.dtype == torch.float32
    for buf in model.buffers():
        if buf.is_floating_point():
            assert buf.dtype == torch.float32


def _assert_memory_states_fp32(memory_states) -> None:
    for tensor in _iter_tensors(memory_states):
        if tensor.is_floating_point():
            assert tensor.dtype == torch.float32


def test_fp32_enforcement_onyx():
    torch.manual_seed(0)
    config = _tiny_config()
    model = Onyx(config)

    _force_fp16_param(model)
    _force_fp16_buffer(model)

    enforce_fp32_everywhere(model)

    _assert_model_fp32(model)

    ms = model.init_memory_states(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    _assert_memory_states_fp32(ms)


def test_fp32_enforcement_mhc():
    torch.manual_seed(0)
    config = _tiny_config()
    model = OnyxMHC(config, mhc_n=2, mhc_mode="mhc", mhc_sinkhorn=True, mhc_sinkhorn_iters=2)

    _force_fp16_param(model)
    _force_fp16_buffer(model)

    enforce_fp32_everywhere(model)

    _assert_model_fp32(model)

    ms = model.init_memory_states(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    _assert_memory_states_fp32(ms)
