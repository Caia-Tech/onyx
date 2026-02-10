#!/usr/bin/env python3
"""
Onyx 11M - Hope Attention Transformer with Persistent Memory

Core model code (OnyxConfig, Onyx, M3Optimizer, param grouping).
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from flash_attn import flash_attn_func
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except Exception:
    FLASH_ATTN_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OnyxConfig:
    # === Core Architecture ===
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 4096

    # === Sequence & Vocab ===
    max_seq_len: int = 4096
    train_seq_len: int = 4096
    vocab_size: int = 128258

    # === RoPE ===
    rope_base: float = 500000.0

    # === Precision & Performance ===
    use_flash_attention: bool = False
    dtype: str = "float32"

    # === Regularization ===
    dropout: float = 0.0
    attention_dropout: float = 0.0
    norm_eps: float = 1e-5
    gradient_checkpointing: bool = False
    tie_embeddings: bool = True

    # === Hope Attention ===
    use_hope_attention: bool = True
    self_referential_keys: bool = True
    self_referential_values: bool = True
    generate_own_values: bool = True
    use_short_conv: bool = True
    conv_kernel_size: int = 4
    use_memory_gate: bool = True

    # === Delta Memory ===
    memory_type: str = "linear"
    use_delta_rule: bool = True
    normalize_keys: bool = True
    memory_lr_init: float = 0.08
    memory_lr_learnable: bool = True
    memory_decay_init: float = 0.85
    memory_decay_learnable: bool = True
    max_memory_lr: float = 0.2
    min_memory_decay: float = 0.5
    memory_max_norm: float = 30.0
    memory_chunk_size: int = 64

    # === Inference Memory ===
    inference_memory_lr: float = 0.01
    inference_memory_decay: float = 0.95
    inference_memory_max_updates: int = 100000

    # === Memory Regularization ===
    memory_reg_weight: float = 0.0001

    # === Loss & Regularization ===
    label_smoothing: float = 0.0  # 0.0 = disabled, 0.1 = recommended for mode collapse
    entropy_reg_weight: float = 0.0  # 0.0 = disabled, 0.01 = encourages diverse predictions
    feedback_strength: float = 1.0  # 1.0 = full feedback loop, 0.5 = dampened, 0.0 = no feedback

    # === CMS FFN ===
    use_cms_ffn: bool = True
    cms_num_levels: int = 4
    cms_base_chunk: int = 32
    cms_chunk_multiplier: int = 2
    cms_aggregation: str = "learned"
    cms_update_every_base_steps: int = 1
    cms_update_every_multiplier: int = 2

    # === M3 Optimizer defaults ===
    m3_beta_slow: float = 0.99
    m3_slow_freq: int = 50
    m3_slow_weight: float = 0.1

    # === MHC Debug ===
    mhc_debug_finite_checks: bool = False
    mhc_debug_every: int = 0

    # === Residual Matrix Transformer (RMT) ===
    use_rmt: bool = False
    rmt_dk: int = 16
    rmt_dv: int = 16
    rmt_heads: int = 0  # 0 => default to n_heads
    rmt_ff_multiplier: float = 2.0
    rmt_ff_dim: int = 0  # 0 => infer from rmt_heads * rmt_dv * rmt_ff_multiplier
    rmt_init: str = "normal"  # "normal" | "zero"
    rmt_readwrite_dtype: str = "float32"


# =============================================================================
# Basic Components
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Keep normalization in fp32 for stability under AMP (especially on MPS).
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            xf = x.float()
            norm = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            y = xf * norm
            y = y * self.weight.float()
        return y.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(self, x: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._update_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# =============================================================================
# Chunked Parallel Delta Memory
# =============================================================================

def normalize_for_delta(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


class ChunkedLinearDeltaMemory(nn.Module):
    def __init__(self, d_in: int, d_out: int, config: OnyxConfig):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.config = config
        self.chunk_size = config.memory_chunk_size

        self.M_init = nn.Parameter(torch.zeros(d_out, d_in))
        nn.init.xavier_uniform_(self.M_init, gain=0.1)

        def inv_sig(x):
            x = max(min(float(x), 0.999), 0.001)
            return math.log(x / (1 - x))

        # Dynamic hyperparams projections
        self.eta_proj = nn.Linear(d_in, 1, bias=True)
        self.alpha_proj = nn.Linear(d_in, 1, bias=True)
        nn.init.zeros_(self.eta_proj.weight)
        nn.init.constant_(self.eta_proj.bias, inv_sig(config.memory_lr_init))
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, inv_sig(config.memory_decay_init))

        # Legacy buffer compatibility hook
        self.register_load_state_dict_post_hook(self._ignore_missing_dynamic_hyperparams)
        
        self.register_buffer("inference_eta", torch.tensor(config.inference_memory_lr))
        self.register_buffer("inference_alpha", torch.tensor(config.inference_memory_decay))

        if config.use_memory_gate:
            self.gate_proj = nn.Linear(d_in, 1, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)

    def _compute_chunk_hyperparams(self, chunk: Tensor, inference_mode: bool) -> Tuple[Tensor, Tensor]:
        device_type = chunk.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            chunk_f = chunk.float()
            eps = 1e-6

            def _inv_sigmoid01(p: Tensor) -> Tensor:
                p = p.clamp(min=eps, max=1.0 - eps)
                return torch.log(p / (1.0 - p))

            if inference_mode:
                base_eta = self.inference_eta.to(device=chunk.device, dtype=torch.float32)
                base_eta01 = (base_eta / float(self.config.max_memory_lr)).clamp(min=eps, max=1.0 - eps)
                eta_base_logit = _inv_sigmoid01(base_eta01)

                min_d = float(self.config.min_memory_decay)
                base_alpha = self.inference_alpha.to(device=chunk.device, dtype=torch.float32)
                base_alpha01 = ((base_alpha - min_d) / (1.0 - min_d)).clamp(min=eps, max=1.0 - eps)
                alpha_base_logit = _inv_sigmoid01(base_alpha01)
            else:
                eta_base_logit = torch.tensor(0.0, device=chunk.device, dtype=torch.float32)
                alpha_base_logit = torch.tensor(0.0, device=chunk.device, dtype=torch.float32)

            eta_raw = eta_base_logit + self.eta_proj(chunk_f)
            alpha_raw = alpha_base_logit + self.alpha_proj(chunk_f)

            eta_tok = torch.sigmoid(eta_raw) * float(self.config.max_memory_lr)
            min_d = float(self.config.min_memory_decay)
            alpha_tok = min_d + torch.sigmoid(alpha_raw) * (1.0 - min_d)

            eta = eta_tok.mean(dim=1, keepdim=True)
            alpha = alpha_tok.mean(dim=1, keepdim=True)
            return eta, alpha

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        # Keep memory state in fp32 even under AMP for stability.
        return self.M_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device=device, dtype=torch.float32)

    def retrieve_batch(self, M: Tensor, queries: Tensor) -> Tensor:
        device_type = queries.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            q = queries.float()
            m = M.float()
            return torch.bmm(q, m.transpose(-1, -2))

    def _process_chunk(
        self,
        M: Tensor,
        chunk: Tensor,
        eta: Tensor,
        alpha: Tensor,
        update_memory: bool,
        target_generator: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        device_type = chunk.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            chunk_f = chunk.float()
            M_f = M.float()
            outputs_f = self.retrieve_batch(M_f, chunk_f)
            if not update_memory:
                return outputs_f.to(dtype=chunk.dtype), M_f

            keys = (
                normalize_for_delta(chunk_f, dim=-1, eps=self.config.norm_eps)
                if self.config.normalize_keys
                else chunk_f
            )

            k_mean = keys.mean(dim=1)

            # Feedback Loop Logic
            if target_generator is not None:
                targets = target_generator(outputs_f)
                v_target = targets.float().mean(dim=1)
            else:
                v_target = outputs_f.mean(dim=1)

            if self.config.use_delta_rule:
                Mk = torch.bmm(M_f, k_mean.unsqueeze(-1)).squeeze(-1)
                error = v_target - Mk
                update = torch.bmm(error.unsqueeze(-1), k_mean.unsqueeze(1))
            else:
                update = torch.bmm(v_target.unsqueeze(-1), k_mean.unsqueeze(1))

            C = int(chunk.size(1))
            chunk_scale = min(C / float(self.chunk_size), 1.0)
            M_new = alpha.float() * M_f + eta.float() * float(chunk_scale) * update

            if hasattr(self, "gate_proj"):
                gate = torch.sigmoid(self.gate_proj(k_mean)).unsqueeze(-1)
                M_new = gate * M_new + (1.0 - gate) * M_f

            M_norm = torch.norm(M_new, dim=(-2, -1), keepdim=True)
            scale = torch.clamp(float(self.config.memory_max_norm) / (M_norm + 1e-6), max=1.0)
            M_new = M_new * scale
            return outputs_f.to(dtype=chunk.dtype), M_new

    def forward(
        self,
        x: Tensor,
        M: Optional[Tensor] = None,
        inference_mode: bool = False,
        update_memory: bool = True,
        target_generator: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        B, S, _ = x.shape
        if M is None or M.size(0) != B:
            M = self.init_state(B, x.device, x.dtype)

        outs = []
        for start in range(0, S, self.chunk_size):
            end = min(start + self.chunk_size, S)
            chunk = x[:, start:end, :]
            eta, alpha = self._compute_chunk_hyperparams(chunk, inference_mode)
            chunk_out, M = self._process_chunk(M, chunk, eta, alpha, update_memory, target_generator)
            outs.append(chunk_out)
        return torch.cat(outs, dim=1), M

    @staticmethod
    def _ignore_missing_dynamic_hyperparams(module, incompatible_keys) -> None:
        missing = incompatible_keys.missing_keys
        # [FIX] Ignore missing projection keys when loading old checkpoints
        suffixes = (
            "eta_proj.weight", "eta_proj.bias",
            "alpha_proj.weight", "alpha_proj.bias",
            "gate_proj.weight", "gate_proj.bias"
        )
        for i in range(len(missing) - 1, -1, -1):
            if missing[i].endswith(suffixes):
                missing.pop(i)

    def get_memory_reg_loss(self, M: Tensor) -> Tensor:
        return M.pow(2).mean()


class ChunkedSelfReferentialMemory(nn.Module):
    def __init__(self, d_in: int, d_out: int, config: OnyxConfig):
        super().__init__()
        self.memory = ChunkedLinearDeltaMemory(d_in, d_out, config)
        self.config = config
        if config.generate_own_values:
            self.value_gen = nn.Sequential(
                nn.Linear(d_out, d_out, bias=False),
                nn.SiLU(),
                nn.Linear(d_out, d_out, bias=False),
            )
            nn.init.eye_(self.value_gen[0].weight)
            nn.init.zeros_(self.value_gen[2].weight)
        else:
            self.value_gen = None

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.memory.init_state(batch_size, device, dtype)

    def forward(
        self,
        x: Tensor,
        M: Optional[Tensor] = None,
        value_input: Optional[Tensor] = None,
        inference_mode: bool = False,
        update_memory: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        
        # Feedback Generator
        def target_gen_fn(retrieved_vals):
            if self.value_gen is not None:
                device_type = retrieved_vals.device.type
                with torch.autocast(device_type=device_type, enabled=False):
                    base = value_input if value_input is not None else retrieved_vals
                    base_f = base.float()
                    transformed = self.value_gen(base_f)
                    strength = float(self.config.feedback_strength)
                    return base_f + strength * transformed
            return retrieved_vals.float()

        outputs, M_new = self.memory(
            x, M, 
            inference_mode=inference_mode, 
            update_memory=update_memory,
            target_generator=target_gen_fn 
        )
        
        final_out = target_gen_fn(outputs).to(dtype=x.dtype)
        return final_out, M_new

    def get_memory_reg_loss(self, M: Tensor) -> Tensor:
        return self.memory.get_memory_reg_loss(M)


# =============================================================================
# CMS FFN
# =============================================================================

class CMSFFN(nn.Module):
    def __init__(self, config: OnyxConfig):
        super().__init__()
        self.config = config
        self.num_levels = config.cms_num_levels
        
        self.chunk_sizes = [
            config.cms_base_chunk * (config.cms_chunk_multiplier ** i)
            for i in range(self.num_levels)
        ]

        per_level_ff = max(1, config.d_ff // self.num_levels)
        self.level_ffns = nn.ModuleList([
            SwiGLUFFN(config.d_model, per_level_ff, config.dropout)
            for _ in range(self.num_levels)
        ])

        if config.cms_aggregation == "learned":
            self.level_weights = nn.Parameter(torch.ones(self.num_levels) / self.num_levels)
        else:
            self.register_buffer("level_weights", torch.ones(self.num_levels) / self.num_levels)

        self.output_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        weights = F.softmax(self.level_weights, dim=0)

        level_outputs = []
        for w, chunk_size, ffn in zip(weights, self.chunk_sizes, self.level_ffns):
            if chunk_size >= S:
                out = ffn(x)
            else:
                num_chunks = (S + chunk_size - 1) // chunk_size
                padded_len = num_chunks * chunk_size
                x_padded = F.pad(x, (0, 0, 0, padded_len - S)) if padded_len > S else x
                chunks = x_padded.view(B * num_chunks, chunk_size, D)
                processed = ffn(chunks).view(B, num_chunks, chunk_size, D)
                out = processed.view(B, padded_len, D)[:, :S, :]
            level_outputs.append(out * w)

        combined = sum(level_outputs)
        return self.output_proj(combined)


# =============================================================================
# Attention
# =============================================================================

def _causal_mask_with_cache(S_q: int, S_k: int, device: torch.device, position_offset: int) -> Tensor:
    q_pos = (torch.arange(S_q, device=device) + position_offset).unsqueeze(1)
    k_pos = torch.arange(S_k, device=device).unsqueeze(0)
    return k_pos > q_pos

def build_packed_causal_mask(boundaries: Tensor, S: int, device: torch.device) -> Tensor:
    """
    boundaries: 1D int tensor like [0, a, b, ..., S]
    returns mask bool [S, S], where True means "mask out"
    - standard causal within each segment
    - mask everything outside segment
    """
    b = boundaries.to(device=device)
    b = b[b >= 0]
    if b.numel() == 0 or int(b[0]) != 0:
        b = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), b])
    if int(b[-1]) != S:
        b = torch.cat([b, torch.tensor([S], device=device, dtype=torch.int32)])

    b = torch.clamp(b, 0, S)

    seg_id = torch.full((S,), -1, device=device, dtype=torch.int32)
    for i in range(b.numel() - 1):
        start = int(b[i].item())
        end = int(b[i + 1].item())
        if end <= start:
            continue
        start = max(0, start)
        end = min(S, end)
        seg_id[start:end] = i

    qpos = torch.arange(S, device=device).unsqueeze(1)
    kpos = torch.arange(S, device=device).unsqueeze(0)
    causal = kpos > qpos
    seg_mismatch = seg_id.unsqueeze(1) != seg_id.unsqueeze(0)
    return causal | seg_mismatch


def _build_batch_packed_mask(
    packed_cu_seqlens: Tensor,
    packed_num_segs: Optional[Tensor],
    S: int,
    device: torch.device,
) -> Tensor:
    masks: List[Tensor] = []
    B = packed_cu_seqlens.shape[0]
    for b in range(B):
        if packed_num_segs is not None:
            n = int(packed_num_segs[b].item())
            boundaries = packed_cu_seqlens[b, :n]
        else:
            boundaries = packed_cu_seqlens[b]
        masks.append(build_packed_causal_mask(boundaries, S, device))
    return torch.stack(masks, dim=0)

class HopeAttention(nn.Module):
    def __init__(self, config: OnyxConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)

        self.conv = None
        if config.use_short_conv:
            self.conv = nn.Conv1d(
                config.d_model, config.d_model,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.d_model
            )

        kv_dim = config.n_kv_heads * config.head_dim
        if config.self_referential_keys:
            self.k_memory = ChunkedSelfReferentialMemory(config.d_model, kv_dim, config)
        else:
            self.k_proj = nn.Linear(config.d_model, kv_dim, bias=False)

        if config.self_referential_values:
            self.v_memory = ChunkedSelfReferentialMemory(config.d_model, kv_dim, config)
        else:
            self.v_proj = nn.Linear(config.d_model, kv_dim, bias=False)

        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)

    def init_memory_states(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Tensor]:
        states = {}
        if hasattr(self, "k_memory"):
            states["k"] = self.k_memory.init_state(batch_size, device, dtype)
        if hasattr(self, "v_memory"):
            states["v"] = self.v_memory.init_state(batch_size, device, dtype)
        return states

    def get_memory_reg_loss(self, memory_states: Dict[str, Tensor]) -> Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if hasattr(self, "k_memory") and "k" in memory_states:
            loss = loss + self.k_memory.get_memory_reg_loss(memory_states["k"])
        if hasattr(self, "v_memory") and "v" in memory_states:
            loss = loss + self.v_memory.get_memory_reg_loss(memory_states["v"])
        return loss

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[Dict[str, Tensor]] = None,
        update_memories: bool = True,
        inference_mode: bool = False,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        packed_cu_seqlens: Optional[Tensor] = None,
        packed_num_segs: Optional[Tensor] = None,
        kv_cache: Optional[Dict[str, Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Dict[str, Tensor]]]:
        B, S, _ = x.shape

        new_conv_x = None
        if self.conv is not None:
            x_in = x
            k = int(self.conv.kernel_size[0])
            cache_len = max(0, k - 1)

            if cache_len > 0 and kv_cache is not None and isinstance(kv_cache, dict) and kv_cache.get("conv_x") is not None:
                prev = kv_cache["conv_x"]
                x_cat = torch.cat([prev, x_in], dim=1)
                x_conv_all = self.conv(x_cat.transpose(1, 2))[:, :, : x_cat.size(1)].transpose(1, 2)
                x_conv = x_conv_all[:, -S:, :]
                x = x_in + x_conv
                new_conv_x = x_cat[:, -cache_len:, :]
            else:
                x_conv = self.conv(x_in.transpose(1, 2))[:, :, :S].transpose(1, 2)
                x = x_in + x_conv
                if cache_len > 0:
                    new_conv_x = x_in[:, -cache_len:, :]

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)

        new_memory_states: Dict[str, Tensor] = {}

        if hasattr(self, "k_memory"):
            k_mem = memory_states.get("k") if memory_states else None
            k, new_k_mem = self.k_memory(x, k_mem, inference_mode=inference_mode, update_memory=update_memories)
            k = k.view(B, S, self.n_kv_heads, self.head_dim)
            new_memory_states["k"] = new_k_mem
        else:
            k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        if hasattr(self, "v_memory"):
            v_mem = memory_states.get("v") if memory_states else None
            v, new_v_mem = self.v_memory(x, v_mem, inference_mode=inference_mode, update_memory=update_memories)
            v = v.view(B, S, self.n_kv_heads, self.head_dim)
            new_memory_states["v"] = new_v_mem
        else:
            v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        total_seq_len = position_offset + S
        cos, sin = self.rotary(q, total_seq_len)
        cos = cos[position_offset:position_offset + S].unsqueeze(0)
        sin = sin[position_offset:position_offset + S].unsqueeze(0)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        q_rope = q.view(B * self.n_heads, S, self.head_dim)
        k_rope = k.view(B * self.n_kv_heads, S, self.head_dim)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        q = q_rope.view(B, self.n_heads, S, self.head_dim)
        k = k_rope.view(B, self.n_kv_heads, S, self.head_dim)

        new_kv_cache = None
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            new_kv_cache = {"k": k, "v": v}
            if new_conv_x is not None:
                new_kv_cache["conv_x"] = new_conv_x

        kv_seq_len = k.shape[2]

        if self.n_rep > 1:
            k_exp = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, kv_seq_len, self.head_dim)
            v_exp = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, kv_seq_len, self.head_dim)
        else:
            k_exp = k
            v_exp = v

        if self.config.use_flash_attention and FLASH_ATTN_AVAILABLE and kv_cache is None:
            if cu_seqlens is not None and max_seqlen is not None:
                q_fa = q.transpose(1, 2).reshape(B * S, self.n_heads, self.head_dim)
                k_fa = k_exp.transpose(1, 2).reshape(B * kv_seq_len, self.n_heads, self.head_dim)
                v_fa = v_exp.transpose(1, 2).reshape(B * kv_seq_len, self.n_heads, self.head_dim)
                attn_out = flash_attn_varlen_func(
                    q_fa, k_fa, v_fa,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                )
                attn_out = attn_out.reshape(B, S, -1)
            elif packed_cu_seqlens is None:
                attn_out = flash_attn_func(q.transpose(1, 2), k_exp.transpose(1, 2), v_exp.transpose(1, 2), causal=True)
                attn_out = attn_out.reshape(B, S, -1)
            else:
                attn_out = None
        else:
            attn_out = None

        if attn_out is None:
            scale = 1.0 / math.sqrt(self.head_dim)
            if x.device.type == "mps":
                device_type = x.device.type
                with torch.autocast(device_type=device_type, enabled=False):
                    qf = q.float()
                    kf = k_exp.float()
                    vf = v_exp.float()
                    attn = torch.matmul(qf, kf.transpose(-2, -1)) * float(scale)

                    if kv_cache is None:
                        if packed_cu_seqlens is not None:
                            mask = _build_batch_packed_mask(packed_cu_seqlens, packed_num_segs, S, x.device)
                        else:
                            mask = torch.triu(torch.ones(S, kv_seq_len, device=x.device, dtype=torch.bool), diagonal=1)
                    else:
                        mask = _causal_mask_with_cache(
                            S_q=S, S_k=kv_seq_len, device=x.device, position_offset=position_offset
                        )

                    if mask.dim() == 2:
                        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                    else:
                        attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
                    attn = F.softmax(attn, dim=-1)
                    attn = F.dropout(attn, p=self.config.attention_dropout, training=self.training)
                    attn_out = torch.matmul(attn, vf)

                attn_out = attn_out.to(dtype=q.dtype).transpose(1, 2).reshape(B, S, -1)
            else:
                attn = torch.matmul(q, k_exp.transpose(-2, -1)) * scale

                if kv_cache is None:
                    if packed_cu_seqlens is not None:
                        mask = _build_batch_packed_mask(packed_cu_seqlens, packed_num_segs, S, x.device)
                    else:
                        mask = torch.triu(torch.ones(S, kv_seq_len, device=x.device, dtype=torch.bool), diagonal=1)
                else:
                    mask = _causal_mask_with_cache(
                        S_q=S, S_k=kv_seq_len, device=x.device, position_offset=position_offset
                    )

                if mask.dim() == 2:
                    attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                else:
                    attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
                attn = F.softmax(attn, dim=-1)
                attn = F.dropout(attn, p=self.config.attention_dropout, training=self.training)
                attn_out = torch.matmul(attn, v_exp)
                attn_out = attn_out.transpose(1, 2).reshape(B, S, -1)

        out = self.o_proj(attn_out)
        return out, new_memory_states, new_kv_cache


class StandardAttention(nn.Module):
    def __init__(self, config: OnyxConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[Dict[str, Tensor]] = None,
        update_memories: bool = True,
        inference_mode: bool = False,
        cu_seqlens: Optional[Tensor] = None, # [FIX] Added missing args to signature
        max_seqlen: Optional[int] = None,    # [FIX] Added missing args to signature
        packed_cu_seqlens: Optional[Tensor] = None,
        packed_num_segs: Optional[Tensor] = None,
        kv_cache: Optional[Dict[str, Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Dict[str, Tensor]]]:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        total_seq_len = position_offset + S
        cos, sin = self.rotary(q, total_seq_len)
        cos = cos[position_offset:position_offset + S].unsqueeze(0)
        sin = sin[position_offset:position_offset + S].unsqueeze(0)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        q_rope = q.view(B * self.n_heads, S, self.head_dim)
        k_rope = k.view(B * self.n_kv_heads, S, self.head_dim)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        q = q_rope.view(B, self.n_heads, S, self.head_dim)
        k = k_rope.view(B, self.n_kv_heads, S, self.head_dim)

        new_kv_cache = None
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            new_kv_cache = {"k": k, "v": v}

        kv_seq_len = k.shape[2]

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, kv_seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, kv_seq_len, self.head_dim)

        if self.config.use_flash_attention and FLASH_ATTN_AVAILABLE and kv_cache is None:
            if packed_cu_seqlens is None:
                attn_out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True)
                attn_out = attn_out.reshape(B, S, -1)
            else:
                attn_out = None
        else:
            attn_out = None

        if attn_out is None:
            scale = 1.0 / math.sqrt(self.head_dim)
            if x.device.type == "mps":
                device_type = x.device.type
                with torch.autocast(device_type=device_type, enabled=False):
                    qf = q.float()
                    kf = k.float()
                    vf = v.float()
                    attn = torch.matmul(qf, kf.transpose(-2, -1)) * float(scale)

                    if kv_cache is None:
                        if packed_cu_seqlens is not None:
                            mask = _build_batch_packed_mask(packed_cu_seqlens, packed_num_segs, S, x.device)
                        else:
                            mask = torch.triu(torch.ones(S, kv_seq_len, device=x.device, dtype=torch.bool), diagonal=1)
                    else:
                        mask = _causal_mask_with_cache(
                            S_q=S, S_k=kv_seq_len, device=x.device, position_offset=position_offset
                        )

                    if mask.dim() == 2:
                        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                    else:
                        attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
                    attn = F.softmax(attn, dim=-1)
                    attn = F.dropout(attn, p=self.config.attention_dropout, training=self.training)
                    attn_out = torch.matmul(attn, vf)

                attn_out = attn_out.to(dtype=q.dtype).transpose(1, 2).reshape(B, S, -1)
            else:
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale

                if kv_cache is None:
                    if packed_cu_seqlens is not None:
                        mask = _build_batch_packed_mask(packed_cu_seqlens, packed_num_segs, S, x.device)
                    else:
                        mask = torch.triu(torch.ones(S, kv_seq_len, device=x.device, dtype=torch.bool), diagonal=1)
                else:
                    mask = _causal_mask_with_cache(
                        S_q=S, S_k=kv_seq_len, device=x.device, position_offset=position_offset
                    )

                if mask.dim() == 2:
                    attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                else:
                    attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
                attn = F.softmax(attn, dim=-1)
                attn = F.dropout(attn, p=self.config.attention_dropout, training=self.training)
                attn_out = torch.matmul(attn, v)
                attn_out = attn_out.transpose(1, 2).reshape(B, S, -1)

        return self.o_proj(attn_out), {}, new_kv_cache


def _rmt_dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unsupported rmt_readwrite_dtype={name}")
    return table[key]


def _rmt_init_param_(p: nn.Parameter, mode: str, std: float = 0.02) -> None:
    m = str(mode).lower()
    if m == "zero":
        nn.init.zeros_(p)
    elif m == "normal":
        nn.init.normal_(p, mean=0.0, std=std)
    else:
        raise ValueError(f"Unsupported rmt_init={mode}")


class RMTAttention(nn.Module):
    """
    Residual-matrix attention.
    X: [B, T, Dk, Dv]
    """

    def __init__(self, config: OnyxConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = int(config.rmt_heads or config.n_heads)
        self.n_kv_heads = int(config.n_kv_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dk = int(config.rmt_dk)
        self.dv = int(config.rmt_dv)
        self.readwrite_dtype = _rmt_dtype_from_name(config.rmt_readwrite_dtype)

        self.r_q = nn.Parameter(torch.empty(self.n_heads, self.dk))
        self.r_k = nn.Parameter(torch.empty(self.n_kv_heads, self.dk))
        self.r_v = nn.Parameter(torch.empty(self.n_kv_heads, self.dk))
        self.w_o = nn.Parameter(torch.empty(self.n_heads, self.dk))
        _rmt_init_param_(self.r_q, config.rmt_init)
        _rmt_init_param_(self.r_k, config.rmt_init)
        _rmt_init_param_(self.r_v, config.rmt_init)
        _rmt_init_param_(self.w_o, config.rmt_init)

        self.rotary = RotaryEmbedding(self.dv, config.max_seq_len, config.rope_base)

        kv_flat_dim = self.n_kv_heads * self.dv
        if config.use_hope_attention and config.self_referential_keys:
            self.k_memory = ChunkedSelfReferentialMemory(kv_flat_dim, kv_flat_dim, config)
        else:
            self.k_memory = None
        if config.use_hope_attention and config.self_referential_values:
            self.v_memory = ChunkedSelfReferentialMemory(kv_flat_dim, kv_flat_dim, config)
        else:
            self.v_memory = None

    def init_memory_states(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Tensor]:
        states: Dict[str, Tensor] = {}
        if self.k_memory is not None:
            states["k"] = self.k_memory.init_state(batch_size, device, dtype)
        if self.v_memory is not None:
            states["v"] = self.v_memory.init_state(batch_size, device, dtype)
        return states

    def get_memory_reg_loss(self, memory_states: Dict[str, Tensor]) -> Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if self.k_memory is not None and "k" in memory_states:
            loss = loss + self.k_memory.get_memory_reg_loss(memory_states["k"])
        if self.v_memory is not None and "v" in memory_states:
            loss = loss + self.v_memory.get_memory_reg_loss(memory_states["v"])
        return loss

    def _read_heads(self, X: Tensor, r: Tensor) -> Tensor:
        device_type = X.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            xf = X.to(dtype=self.readwrite_dtype)
            rf = r.to(device=X.device, dtype=self.readwrite_dtype)
            out = torch.einsum("hk,btkv->bthv", rf, xf)
        return out.to(dtype=X.dtype)

    def _write_heads(self, X: Tensor, w: Tensor, v: Tensor) -> Tensor:
        # X += sum_h w[h] outer v[h]
        device_type = X.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            xf = X.to(dtype=self.readwrite_dtype)
            wf = w.to(device=X.device, dtype=self.readwrite_dtype)
            vf = v.to(dtype=self.readwrite_dtype)
            delta = torch.einsum("hk,bthv->btkv", wf, vf)
            out = xf + delta
        return out.to(dtype=X.dtype)

    def forward(
        self,
        X: Tensor,
        memory_states: Optional[Dict[str, Tensor]] = None,
        update_memories: bool = True,
        inference_mode: bool = False,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        packed_cu_seqlens: Optional[Tensor] = None,
        packed_num_segs: Optional[Tensor] = None,
        kv_cache: Optional[Dict[str, Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Dict[str, Tensor]]]:
        del cu_seqlens, max_seqlen
        B, S, _, _ = X.shape

        q = self._read_heads(X, self.r_q)
        k = self._read_heads(X, self.r_k)
        v = self._read_heads(X, self.r_v)

        new_states: Dict[str, Tensor] = {}
        if self.k_memory is not None:
            k_mem = memory_states.get("k") if memory_states else None
            k_flat, new_k_mem = self.k_memory(
                k.reshape(B, S, -1),
                k_mem,
                inference_mode=inference_mode,
                update_memory=update_memories,
            )
            k = k_flat.view(B, S, self.n_kv_heads, self.dv)
            new_states["k"] = new_k_mem
        if self.v_memory is not None:
            v_mem = memory_states.get("v") if memory_states else None
            v_flat, new_v_mem = self.v_memory(
                v.reshape(B, S, -1),
                v_mem,
                inference_mode=inference_mode,
                update_memory=update_memories,
            )
            v = v_flat.view(B, S, self.n_kv_heads, self.dv)
            new_states["v"] = new_v_mem

        total_seq_len = position_offset + S
        cos, sin = self.rotary(q, total_seq_len)
        cos = cos[position_offset:position_offset + S].unsqueeze(0)
        sin = sin[position_offset:position_offset + S].unsqueeze(0)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        q_rope = q.view(B * self.n_heads, S, self.dv)
        k_rope = k.view(B * self.n_kv_heads, S, self.dv)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        q = q_rope.view(B, self.n_heads, S, self.dv)
        k = k_rope.view(B, self.n_kv_heads, S, self.dv)

        new_kv_cache: Optional[Dict[str, Tensor]] = None
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            new_kv_cache = {"k": k, "v": v}

        kv_seq_len = k.shape[2]

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, kv_seq_len, self.dv)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, kv_seq_len, self.dv)

        mask = None
        if kv_cache is None:
            if packed_cu_seqlens is not None:
                mask = _build_batch_packed_mask(packed_cu_seqlens, packed_num_segs, S, X.device)
            else:
                mask = torch.triu(torch.ones(S, kv_seq_len, device=X.device, dtype=torch.bool), diagonal=1)
        else:
            mask = _causal_mask_with_cache(S_q=S, S_k=kv_seq_len, device=X.device, position_offset=position_offset)

        scale = 1.0 / math.sqrt(self.dv)
        device_type = X.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            qf = q.float()
            kf = k.float()
            vf = v.float()
            attn = torch.matmul(qf, kf.transpose(-2, -1)) * float(scale)
            if mask.dim() == 2:
                attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            else:
                attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.config.attention_dropout, training=self.training)
            out = torch.matmul(attn, vf).to(dtype=X.dtype)

        out = out.transpose(1, 2).contiguous()  # [B, T, H, Dv]
        X_new = self._write_heads(X, self.w_o, out)
        return X_new, new_states, new_kv_cache


class RMTFFN(nn.Module):
    """
    Matrix FFN:
    - read R vectors from X -> [B, T, R, Dv]
    - FF core over [B, T, R*Dv]
    - write back via learned write keys
    """

    def __init__(self, config: OnyxConfig):
        super().__init__()
        self.config = config
        self.r = int(config.rmt_heads or config.n_heads)
        self.dk = int(config.rmt_dk)
        self.dv = int(config.rmt_dv)
        self.readwrite_dtype = _rmt_dtype_from_name(config.rmt_readwrite_dtype)

        self.r_ff = nn.Parameter(torch.empty(self.r, self.dk))
        self.w_ff = nn.Parameter(torch.empty(self.r, self.dk))
        _rmt_init_param_(self.r_ff, config.rmt_init)
        _rmt_init_param_(self.w_ff, config.rmt_init)

        in_dim = self.r * self.dv
        if int(config.rmt_ff_dim) > 0:
            hidden = int(config.rmt_ff_dim)
        else:
            hidden = max(1, int(round(in_dim * float(config.rmt_ff_multiplier))))
        self.core = SwiGLUFFN(in_dim, hidden, config.dropout)

    def forward(self, X: Tensor) -> Tensor:
        device_type = X.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            xf = X.to(dtype=self.readwrite_dtype)
            rf = self.r_ff.to(device=X.device, dtype=self.readwrite_dtype)
            reads_f = torch.einsum("rk,btkv->btrv", rf, xf)
        reads = reads_f.to(dtype=X.dtype)
        flat = reads.reshape(reads.shape[0], reads.shape[1], -1)
        out = self.core(flat).view(reads.shape[0], reads.shape[1], self.r, self.dv)
        with torch.autocast(device_type=device_type, enabled=False):
            wf = self.w_ff.to(device=X.device, dtype=self.readwrite_dtype)
            outf = out.to(dtype=self.readwrite_dtype)
            delta = torch.einsum("rk,btrv->btkv", wf, outf)
            y = xf + delta
        return y.to(dtype=X.dtype)


class RMTBlock(nn.Module):
    def __init__(self, config: OnyxConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.r = int(config.rmt_heads or config.n_heads)
        self.dk = int(config.rmt_dk)
        self.dv = int(config.rmt_dv)
        self.view_dim = self.r * self.dv

        self.norm1 = RMSNorm(self.view_dim, eps=config.norm_eps)
        self.norm2 = RMSNorm(self.view_dim, eps=config.norm_eps)
        self.norm_w1 = nn.Parameter(torch.empty(self.r, self.dk))
        self.norm_w2 = nn.Parameter(torch.empty(self.r, self.dk))
        _rmt_init_param_(self.norm_w1, config.rmt_init)
        _rmt_init_param_(self.norm_w2, config.rmt_init)

        self.attention = RMTAttention(config, layer_idx=layer_idx)
        self.ffn = RMTFFN(config)

    def _token_view(self, X: Tensor) -> Tensor:
        # Reuse FF read keys as a stable token view basis.
        return torch.einsum("rk,btkv->btrv", self.ffn.r_ff.to(dtype=X.dtype), X)

    def init_memory_states(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        if hasattr(self.attention, "init_memory_states"):
            return {"attention": self.attention.init_memory_states(batch_size, device, dtype)}
        return {}

    def get_memory_reg_loss(self, memory_states: Dict[str, Any]) -> Tensor:
        if hasattr(self.attention, "get_memory_reg_loss") and "attention" in memory_states:
            return self.attention.get_memory_reg_loss(memory_states["attention"])
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(
        self,
        X: Tensor,
        memory_states: Optional[Dict[str, Any]] = None,
        update_memories: bool = True,
        inference_mode: bool = False,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        packed_cu_seqlens: Optional[Tensor] = None,
        packed_num_segs: Optional[Tensor] = None,
        kv_cache: Optional[Dict[str, Any]] = None,
        position_offset: int = 0,
    ) -> Tuple[Tensor, Dict[str, Any], Optional[Dict[str, Any]]]:
        attn_mem = memory_states.get("attention", {}) if memory_states else {}

        v1 = self._token_view(X).reshape(X.shape[0], X.shape[1], self.view_dim)
        n1 = self.norm1(v1).view(X.shape[0], X.shape[1], self.r, self.dv)
        X = X + torch.einsum("rk,btrv->btkv", self.norm_w1.to(dtype=X.dtype), n1)

        X, new_attn_mem, new_kv = self.attention(
            X,
            memory_states=attn_mem,
            update_memories=update_memories,
            inference_mode=inference_mode,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            packed_cu_seqlens=packed_cu_seqlens,
            packed_num_segs=packed_num_segs,
            kv_cache=kv_cache.get("kv", None) if kv_cache else None,
            position_offset=position_offset,
        )

        v2 = self._token_view(X).reshape(X.shape[0], X.shape[1], self.view_dim)
        n2 = self.norm2(v2).view(X.shape[0], X.shape[1], self.r, self.dv)
        X = X + torch.einsum("rk,btrv->btkv", self.norm_w2.to(dtype=X.dtype), n2)
        X = self.ffn(X)

        new_states = {"attention": new_attn_mem} if new_attn_mem else {}
        new_cache = {"kv": new_kv} if new_kv else None
        return X, new_states, new_cache


class HopeBlock(nn.Module):
    def __init__(self, config: OnyxConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)

        if config.use_hope_attention:
            self.attention = HopeAttention(config, layer_idx)
        else:
            self.attention = StandardAttention(config, layer_idx)

        self.ffn = CMSFFN(config) if config.use_cms_ffn else SwiGLUFFN(config.d_model, config.d_ff, config.dropout)

    def init_memory_states(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        if hasattr(self.attention, "init_memory_states"):
            return {"attention": self.attention.init_memory_states(batch_size, device, dtype)}
        return {}

    def get_memory_reg_loss(self, memory_states: Dict[str, Any]) -> Tensor:
        if hasattr(self.attention, "get_memory_reg_loss") and "attention" in memory_states:
            return self.attention.get_memory_reg_loss(memory_states["attention"])
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[Dict[str, Any]] = None,
        update_memories: bool = True,
        inference_mode: bool = False,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        packed_cu_seqlens: Optional[Tensor] = None,
        packed_num_segs: Optional[Tensor] = None,
        kv_cache: Optional[Dict[str, Any]] = None,
        position_offset: int = 0,
    ) -> Tuple[Tensor, Dict[str, Any], Optional[Dict[str, Any]]]:
        attn_mem = memory_states.get("attention", {}) if memory_states else {}
        layer_kv_cache = kv_cache.get("kv", None) if kv_cache else None

        attn_out, new_attn_mem, new_kv = self.attention(
            self.norm1(x),
            memory_states=attn_mem,
            update_memories=update_memories,
            inference_mode=inference_mode,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            packed_cu_seqlens=packed_cu_seqlens,
            packed_num_segs=packed_num_segs,
            kv_cache=layer_kv_cache,
            position_offset=position_offset,
        )

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        new_states = {"attention": new_attn_mem} if new_attn_mem else {}
        new_cache = {"kv": new_kv} if new_kv else None
        return x, new_states, new_cache


class Onyx(nn.Module):
    def __init__(self, config: OnyxConfig):
        super().__init__()
        self.config = config
        self.use_rmt = bool(config.use_rmt)
        self.rmt_heads = int(config.rmt_heads or config.n_heads)
        self.rmt_dk = int(config.rmt_dk)
        self.rmt_dv = int(config.rmt_dv)

        if self.use_rmt:
            if self.rmt_heads != int(config.n_heads):
                raise ValueError("RMT currently requires rmt_heads == n_heads.")
            if self.rmt_dk < 1 or self.rmt_dv < 1:
                raise ValueError("RMT requires rmt_dk >= 1 and rmt_dv >= 1.")
            if self.rmt_dv % 2 != 0:
                raise ValueError("RMT requires rmt_dv to be even for rotary embedding.")

            self.embed = nn.Embedding(config.vocab_size, config.d_model)
            self.embed_dropout = nn.Dropout(config.dropout)
            self.rmt_in_proj = nn.Linear(config.d_model, self.rmt_dk * self.rmt_dv, bias=False)
            self.layers = nn.ModuleList([RMTBlock(config, layer_idx=i) for i in range(config.n_layers)])

            self.rmt_readout = nn.Parameter(torch.empty(self.rmt_heads, self.rmt_dk))
            _rmt_init_param_(self.rmt_readout, config.rmt_init)
            self.rmt_out_proj = nn.Linear(self.rmt_heads * self.rmt_dv, config.d_model, bias=False)
            self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            if config.tie_embeddings:
                self.lm_head.weight = self.embed.weight
        else:
            self.embed = nn.Embedding(config.vocab_size, config.d_model)
            self.embed_dropout = nn.Dropout(config.dropout)

            self.layers = nn.ModuleList([HopeBlock(config, layer_idx=i) for i in range(config.n_layers)])
            self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            if config.tie_embeddings:
                self.lm_head.weight = self.embed.weight

        self._init_weights()
        self.gradient_checkpointing = config.gradient_checkpointing
        self._memory_update_count = 0

    def _rmt_to_logits_input(self, X: Tensor) -> Tensor:
        view = torch.einsum("rk,btkv->btrv", self.rmt_readout.to(dtype=X.dtype), X)
        vec = view.reshape(X.shape[0], X.shape[1], -1)
        return self.rmt_out_proj(vec)

    def _init_weights(self):
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
        out = []
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

    def get_memory_reg_loss(self, memory_states: List[Dict[str, Any]]) -> Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer, layer_mem in zip(self.layers, memory_states):
            total = total + layer.get_memory_reg_loss(layer_mem)
        return total * self.config.memory_reg_weight

    def save_memory_states(self, memory_states: List[Dict[str, Any]], path: str):
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
        packed_cu_seqlens: Optional[Tensor] = None,
        packed_num_segs: Optional[Tensor] = None,
        kv_cache: Optional[List[Optional[Dict[str, Any]]]] = None,
        position_offset: int = 0,
    ) -> Dict[str, Any]:
        B, S = input_ids.shape
        x = self.embed(input_ids)
        x = self.embed_dropout(x)

        X = None
        if self.use_rmt:
            X = self.rmt_in_proj(x).view(B, S, self.rmt_dk, self.rmt_dv)

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
            if self.gradient_checkpointing and self.training:
                layer_in = X if self.use_rmt else x
                layer_out, new_mem, new_kv = torch.utils.checkpoint.checkpoint(
                    layer, layer_in, memory_states[i], update_memories, inference_mode,
                    cu_seqlens, max_seqlen, packed_cu_seqlens, packed_num_segs, kv_cache[i], position_offset,
                    use_reentrant=False,
                )
            else:
                layer_in = X if self.use_rmt else x
                layer_out, new_mem, new_kv = layer(
                    layer_in, memory_states[i], update_memories, inference_mode,
                    cu_seqlens, max_seqlen, packed_cu_seqlens, packed_num_segs, kv_cache[i], position_offset,
                )
            if self.use_rmt:
                X = layer_out
            else:
                x = layer_out
            new_memory_states.append(new_mem)
            new_kv_cache.append(new_kv)

        if update_memories:
            self._memory_update_count += S

        if self.use_rmt:
            assert X is not None
            x = self._rmt_to_logits_input(X)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Keep the loss computation in fp32 for numerical stability under AMP/MPS.
            device_type = logits.device.type
            with torch.autocast(device_type=device_type, enabled=False):
                shift_logits = logits[:, :-1, :].contiguous().float()
                shift_labels = labels[:, 1:].contiguous()
                flat_logits = shift_logits.view(-1, self.config.vocab_size)
                flat_labels = shift_labels.view(-1)
                valid = flat_labels != -100
                if valid.any():
                    if not torch.isfinite(flat_logits[valid]).all():
                        raise RuntimeError("Non-finite logits at valid label positions before cross_entropy")

                    # Apply label smoothing if configured
                    if self.config.label_smoothing > 0.0:
                        smoothing = float(self.config.label_smoothing)
                        # Compute log probabilities
                        log_probs = F.log_softmax(flat_logits, dim=-1)
                        # Standard cross-entropy component
                        nll = -log_probs.gather(dim=-1, index=flat_labels.unsqueeze(1)).squeeze(1)
                        # Smoothing component (uniform over vocab)
                        smooth_loss = -log_probs.mean(dim=-1)
                        # Combine: (1-smoothing)*nll + smoothing*smooth_loss
                        per_tok = (1.0 - smoothing) * nll + smoothing * smooth_loss
                    else:
                        # Standard cross-entropy (no smoothing)
                        per_tok = F.cross_entropy(
                            flat_logits,
                            flat_labels,
                            ignore_index=-100,
                            reduction="none",
                        )
                    denom = valid.sum().clamp(min=1)
                    loss = per_tok[valid].sum() / denom

                    # Add entropy regularization if configured (encourages diverse predictions)
                    if self.config.entropy_reg_weight > 0.0:
                        weight = float(self.config.entropy_reg_weight)
                        probs = F.softmax(flat_logits[valid], dim=-1)
                        log_probs = F.log_softmax(flat_logits[valid], dim=-1)
                        entropy = -(probs * log_probs).sum(dim=-1).mean()
                        loss = loss - weight * entropy
                else:
                    loss = torch.zeros((), device=logits.device, dtype=torch.float32)

        out = {
            "logits": logits,
            "loss": loss,
            "memory_states": new_memory_states,
            "kv_cache": new_kv_cache,
        }
        if return_memory_reg_loss:
            out["memory_reg_loss"] = self.get_memory_reg_loss(new_memory_states)
        return out


# =============================================================================
# M3 Optimizer + helpers
# =============================================================================

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power of matrix G.
    This projects G onto the orthogonal group (closest orthogonal matrix).
    """
    if G.ndim != 2:
        raise ValueError(f"zeropower_via_newtonschulz5 expects a 2D tensor, got shape={tuple(G.shape)}")
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    compute_dtype = torch.bfloat16 if G.device.type == "cuda" else torch.float32
    X = G.to(dtype=compute_dtype)
    X /= (X.norm() + eps)
    
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if transposed:
        X = X.T
        
    return X.to(dtype=G.dtype)

class M3Optimizer(torch.optim.Optimizer):
    """
    True Multi-scale Momentum Muon (M3) Implementation.
    """
    def __init__(
        self,
        params,
        lr: float = 0.02, 
        momentum: float = 0.95,
        beta_slow: float = 0.99,
        slow_freq: int = 50,
        slow_weight: float = 0.1,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        use_nesterov: bool = True
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            beta_slow=beta_slow,
            slow_freq=slow_freq,
            slow_weight=slow_weight,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            use_nesterov=use_nesterov
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            beta_slow = group["beta_slow"]
            slow_freq = group["slow_freq"]
            slow_weight = group["slow_weight"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            use_nesterov = group["use_nesterov"]
            mode = group.get("m3_mode", "auto")
            force_vector = mode == "vector"
            force_matrix = mode == "matrix"

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                use_matrix = p.ndim == 2
                if force_vector:
                    use_matrix = False
                elif force_matrix and p.ndim != 2:
                    use_matrix = False

                # --- 1. Vector/Embedding Logic (Standard SGD+Momentum) ---
                if not use_matrix:
                    g = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    
                    if use_nesterov:
                        update = g + momentum * buf
                    else:
                        update = buf
                        
                    if wd != 0:
                        p.data.mul_(1 - lr * wd)

                    p.data.add_(update, alpha=-lr)
                    continue

                # --- 2. Matrix Logic (The Muon Core) ---
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["fast_momentum"] = torch.zeros_like(p)
                    state["slow_memory"] = torch.zeros_like(p)

                g = p.grad
                state["step"] += 1
                
                # A. Update Fast Momentum
                buf = state["fast_momentum"]
                buf.mul_(momentum).add_(g)
                
                # Nesterov lookahead for the update direction
                update_g = (g + momentum * buf) if use_nesterov else buf

                # B. Update Slow Memory (Frequency Gated)
                if state["step"] % slow_freq == 0:
                    # Integrate fast momentum into slow memory
                    state["slow_memory"].mul_(beta_slow).add_(buf, alpha=1 - beta_slow)

                # C. Combine Fast + Slow
                if state["step"] > slow_freq:
                    combined_update = update_g + slow_weight * state["slow_memory"]
                else:
                    combined_update = update_g

                # D. Orthogonalization (Newton-Schulz)
                orthogonal_update = zeropower_via_newtonschulz5(combined_update, steps=ns_steps)

                # E. Apply Update
                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                
                # Apply the orthogonalized update
                p.data.add_(orthogonal_update, alpha=-lr)

        return loss


def create_onyx(**kwargs) -> Onyx:
    return Onyx(OnyxConfig(**kwargs))


def get_param_groups(model: Onyx, weight_decay: float = 0.1, memory_lr_scale: float = 0.1):
    decay_params, no_decay_params, memory_params = [], [], []
    mixer_params = []
    seen = set()

    def add_unique(lst, p):
        pid = id(p)
        if pid in seen:
            return
        seen.add(pid)
        lst.append(p)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "embed" in name or "lm_head" in name:
            # Keep embeddings in the no-decay bucket for stable defaults.
            add_unique(no_decay_params, param)
            continue

        if "mixers." in name:
            add_unique(mixer_params, param)
            continue

        if any(k in name for k in ("eta_proj", "alpha_proj", "gate_proj")):
            add_unique(memory_params, param)
            continue

        if "bias" in name or "norm" in name:
            add_unique(no_decay_params, param)
            continue

        add_unique(decay_params, param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": memory_params, "weight_decay": 0.0, "lr_scale": memory_lr_scale},
    ]
    if mixer_params:
        groups.append({"params": mixer_params, "weight_decay": 0.0, "m3_mode": "vector"})
    return groups
