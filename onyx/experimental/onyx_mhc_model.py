"""
Experimental Onyx variant with Hyper-Connections (HC/mHC).
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from onyx.model import OnyxConfig, HopeBlock, RMSNorm
from onyx.experimental.mhc import MHCMixer, aggregate_streams, scatter_delta


class OnyxMHC(nn.Module):
    """
    Onyx model variant with an expanded residual stream and per-layer mixing.
    """

    def __init__(
        self,
        config: OnyxConfig,
        mhc_n: int = 2,
        mhc_mode: str = "mhc",
        mhc_sinkhorn: bool = True,
        mhc_sinkhorn_iters: int = 10,
    ) -> None:
        super().__init__()
        if mhc_n < 1:
            raise ValueError("mhc_n must be >= 1")
        if mhc_mode not in ("mhc", "hc"):
            raise ValueError(f"mhc_mode must be 'mhc' or 'hc', got {mhc_mode}")

        self.config = config
        self.mhc_n = int(mhc_n)
        self.mhc_mode = mhc_mode
        self.mhc_sinkhorn = bool(mhc_sinkhorn)
        self.mhc_sinkhorn_iters = int(mhc_sinkhorn_iters)

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([HopeBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self.mixers = nn.ModuleList(
            [
                MHCMixer(
                    self.mhc_n,
                    mode=self.mhc_mode,
                    use_sinkhorn=self.mhc_sinkhorn,
                    sinkhorn_iters=self.mhc_sinkhorn_iters,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self._init_weights()
        self.gradient_checkpointing = config.gradient_checkpointing
        self._memory_update_count = 0
        self.enable_finite_checks = bool(getattr(config, "mhc_debug_finite_checks", False))
        self.finite_check_every = int(getattr(config, "mhc_debug_every", 0))

    def _init_weights(self) -> None:
        std = 0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)

    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.tie_embeddings:
            n_params -= self.embed.weight.numel()
        return n_params

    def init_memory_states(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> List[Dict[str, Any]]:
        return [layer.init_memory_states(batch_size, device, dtype) for layer in self.layers]

    def detach_memory_states(self, memory_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for layer_mem in memory_states:
            layer_detached = {}
            for k, v in layer_mem.items():
                if isinstance(v, dict):
                    layer_detached[k] = {kk: vv.detach() if torch.is_tensor(vv) else vv for kk, vv in v.items()}
                elif torch.is_tensor(v):
                    layer_detached[k] = v.detach()
                else:
                    layer_detached[k] = v
            out.append(layer_detached)
        return out

    def _should_check(self, step: Optional[int]) -> bool:
        if not self.enable_finite_checks:
            return False
        if self.finite_check_every <= 0:
            return False
        if step is None:
            return True
        return step % self.finite_check_every == 0

    def _check_finite(self, name: str, tensor: Optional[Tensor], *, step: Optional[int] = None, layer: Optional[int] = None) -> None:
        if not self._should_check(step):
            return
        if tensor is None or not torch.is_tensor(tensor):
            return
        if torch.isfinite(tensor).all().item():
            return
        data = tensor.detach().float()
        nan_count = int(torch.isnan(data).sum().item())
        inf_count = int(torch.isinf(data).sum().item())
        finite = torch.isfinite(data)
        if finite.any():
            finite_vals = data[finite]
            mean = float(finite_vals.mean().item())
            vmax = float(finite_vals.max().item())
            vmin = float(finite_vals.min().item())
        else:
            mean = float("nan")
            vmax = float("nan")
            vmin = float("nan")
        prefix = f"[NonFinite] {name}"
        if layer is not None:
            prefix += f" layer={layer}"
        if step is not None:
            prefix += f" step={step}"
        msg = (
            f"{prefix} dtype={tensor.dtype} mean={mean:.4e} "
            f"max={vmax:.4e} min={vmin:.4e} nan={nan_count} inf={inf_count}"
        )
        print(msg)
        raise RuntimeError(f"Non-finite in {name} layer={layer} step={step}")

    def _check_finite_tree(self, name: str, obj: Any, *, step: Optional[int] = None, layer: Optional[int] = None) -> None:
        if not self._should_check(step):
            return
        if torch.is_tensor(obj):
            self._check_finite(name, obj, step=step, layer=layer)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                self._check_finite_tree(f"{name}.{key}", value, step=step, layer=layer)
            return
        if isinstance(obj, list):
            for idx, value in enumerate(obj):
                self._check_finite_tree(f"{name}[{idx}]", value, step=step, layer=layer)
            return
        if isinstance(obj, tuple):
            for idx, value in enumerate(obj):
                self._check_finite_tree(f"{name}[{idx}]", value, step=step, layer=layer)
            return

    def _block_forward(
        self,
        layer: HopeBlock,
        x: Tensor,
        memory_states: Optional[Dict[str, Any]],
        update_memories: bool,
        inference_mode: bool,
        cu_seqlens: Optional[Tensor],
        max_seqlen: Optional[int],
        kv_cache: Optional[Dict[str, Any]],
        position_offset: int,
        *,
        layer_idx: int,
        step: Optional[int],
    ) -> Tuple[Tensor, Dict[str, Any], Optional[Dict[str, Any]]]:
        attn_mem = memory_states.get("attention", {}) if memory_states else {}
        layer_kv_cache = kv_cache.get("kv", None) if kv_cache else None

        x_norm1 = layer.norm1(x)
        self._check_finite("norm1", x_norm1, step=step, layer=layer_idx)
        attn_out, new_attn_mem, new_kv = layer.attention(
            x_norm1,
            memory_states=attn_mem,
            update_memories=update_memories,
            inference_mode=inference_mode,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kv_cache=layer_kv_cache,
            position_offset=position_offset,
        )
        self._check_finite("attn_out", attn_out, step=step, layer=layer_idx)

        x = x + attn_out
        self._check_finite("resid_attn", x, step=step, layer=layer_idx)

        x_norm2 = layer.norm2(x)
        self._check_finite("norm2", x_norm2, step=step, layer=layer_idx)
        ffn_out = layer.ffn(x_norm2)
        self._check_finite("ffn_out", ffn_out, step=step, layer=layer_idx)

        x = x + ffn_out
        self._check_finite("resid_out", x, step=step, layer=layer_idx)

        new_states = {"attention": new_attn_mem} if new_attn_mem else {}
        new_cache = {"kv": new_kv} if new_kv else None
        return x, new_states, new_cache

    def get_memory_reg_loss(self, memory_states: List[Dict[str, Any]]) -> Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer, layer_mem in zip(self.layers, memory_states):
            total = total + layer.get_memory_reg_loss(layer_mem)
        return total * self.config.memory_reg_weight

    def save_memory_states(self, memory_states: List[Dict[str, Any]], path: str) -> None:
        cpu_states = []
        for layer_mem in memory_states:
            layer_cpu = {}
            for k, v in layer_mem.items():
                if isinstance(v, dict):
                    layer_cpu[k] = {kk: vv.cpu() if torch.is_tensor(vv) else vv for kk, vv in v.items()}
                elif torch.is_tensor(v):
                    layer_cpu[k] = v.cpu()
                else:
                    layer_cpu[k] = v
            cpu_states.append(layer_cpu)
        torch.save({"memory_states": cpu_states, "update_count": self._memory_update_count}, path)

    def load_memory_states(self, path: str, device: torch.device, dtype: Optional[torch.dtype] = None) -> List[Dict[str, Any]]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self._memory_update_count = ck.get("update_count", 0)
        loaded = []
        for layer_mem in ck["memory_states"]:
            layer_loaded = {}
            for k, v in layer_mem.items():
                if isinstance(v, dict):
                    layer_loaded[k] = {kk: vv.to(device=device, dtype=dtype) if torch.is_tensor(vv) else vv for kk, vv in v.items()}
                elif torch.is_tensor(v):
                    layer_loaded[k] = v.to(device=device, dtype=dtype)
                else:
                    layer_loaded[k] = v
            loaded.append(layer_loaded)
        return loaded

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        memory_states: Optional[List[Dict[str, Any]]] = None,
        update_memories: bool = True,
        inference_mode: bool = False,
        return_memory_reg_loss: bool = False,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        kv_cache: Optional[List[Optional[Dict[str, Any]]]] = None,
        position_offset: int = 0,
    ) -> Dict[str, Any]:
        B, S = input_ids.shape
        x = self.embed(input_ids)
        x = self.embed_dropout(x)
        scale = 1.0 / math.sqrt(self.mhc_n)
        x_streams = x.unsqueeze(2).expand(-1, -1, self.mhc_n, -1) * scale
        step = getattr(self, "global_step", None)
        self._check_finite("stream_split", x_streams, step=step)

        if memory_states is None:
            memory_states = [None] * len(self.layers)
        if kv_cache is None:
            if inference_mode:
                kv_cache = [{"kv": {}} for _ in range(len(self.layers))]
            else:
                kv_cache = [None] * len(self.layers)

        new_memory_states = []
        new_kv_cache = []

        for i, layer in enumerate(self.layers):
            x = aggregate_streams(x_streams)
            self._check_finite("aggregate", x, step=step, layer=i)

            if self.gradient_checkpointing and self.training:
                x_out, new_mem, new_kv = torch.utils.checkpoint.checkpoint(
                    lambda x_in, mem, upd, inf, cu, mx, kv, pos: self._block_forward(
                        layer,
                        x_in,
                        mem,
                        upd,
                        inf,
                        cu,
                        mx,
                        kv,
                        pos,
                        layer_idx=i,
                        step=step,
                    ),
                    x,
                    memory_states[i],
                    update_memories,
                    inference_mode,
                    cu_seqlens,
                    max_seqlen,
                    kv_cache[i],
                    position_offset,
                    use_reentrant=False,
                )
            else:
                x_out, new_mem, new_kv = self._block_forward(
                    layer,
                    x,
                    memory_states[i],
                    update_memories,
                    inference_mode,
                    cu_seqlens,
                    max_seqlen,
                    kv_cache[i],
                    position_offset,
                    layer_idx=i,
                    step=step,
                )

            delta = x_out - x
            x_streams = scatter_delta(x_streams, delta, stream_idx=0)
            x_streams = self.mixers[i](x_streams)
            self._check_finite("post_mix", x_streams, step=step, layer=i)

            new_memory_states.append(new_mem)
            new_kv_cache.append(new_kv)
            if self.finite_check_every > 0 and (i % self.finite_check_every == 0):
                self._check_finite_tree("memory_states", new_mem, step=step, layer=i)

        if update_memories:
            self._memory_update_count += S

        x = aggregate_streams(x_streams)
        self._check_finite("final_aggregate", x, step=step)
        x = self.norm(x)
        self._check_finite("final_norm", x, step=step)
        logits = self.lm_head(x)
        self._check_finite("logits", logits, step=step)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, self.config.vocab_size)
            flat_labels = shift_labels.view(-1)
            valid = flat_labels != -100
            if valid.any():
                if not torch.isfinite(flat_logits[valid]).all():
                    raise RuntimeError("Non-finite logits at valid label positions before cross_entropy")
                per_tok = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=-100,
                    reduction="none",
                )
                denom = valid.sum().clamp(min=1)
                loss = per_tok[valid].sum() / denom
            else:
                loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        out = {
            "logits": logits,
            "loss": loss,
            "memory_states": new_memory_states,
            "kv_cache": new_kv_cache,
        }
        if return_memory_reg_loss:
            out["memory_reg_loss"] = self.get_memory_reg_loss(new_memory_states)
        return out
