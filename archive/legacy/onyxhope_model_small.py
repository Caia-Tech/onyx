#!/usr/bin/env python3
"""
OnyxHope Small: ~25-30M parameter model for M2 MacBook Pro training

Scaled down version of OnyxHope with Nested Learning integration:
1. Hope Attention (self-modifying K/V projections)
2. CMS FFN (multi-frequency MLP)
3. M3 Optimizer support

Author: Marvin Tutt, Caia Tech
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Config
# =============================================================================

@dataclass
class OnyxHopeSmallConfig:
    """Configuration for OnyxHope Small for M2 MacBook Pro

    Total params breakdown (~40M):
    - Embeddings: 128K vocab × 256 dim = ~33M (tied input/output)
    - Transformer: 6 layers × ~1.2M each = ~7M

    Memory usage: ~650MB with optimizer states
    """

    # === Core Architecture (scaled for M2) ===
    vocab_size: int = 128258       # Llama 3 vocab (dominates param count)
    d_model: int = 256             # Hidden dimension
    n_layers: int = 6              # Transformer layers
    n_heads: int = 4               # Attention heads (head_dim=64)
    n_kv_heads: int = 4            # KV heads (no GQA)
    d_ff: int = 1024               # FFN dimension (4x d_model)
    max_seq_len: int = 256         # Context length (reduced for M2 memory)

    eos_token_id: int = 2
    pad_token_id: int = 0
    eod_token_id: int = 3

    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    use_swiglu: bool = True
    use_rms_norm: bool = True
    norm_eps: float = 1e-5

    use_flash_attn: bool = False   # MPS doesn't support Flash Attention
    use_torch_compile: bool = False  # Disabled for MPS

    dropout: float = 0.0
    attention_dropout: float = 0.0
    gradient_checkpointing: bool = False
    tie_embeddings: bool = True

    # === Nested Learning Toggle ===
    use_nested_learning: bool = True

    # === Nested Learning: Delta Memory ===
    memory_type: str = "linear"
    use_delta_rule: bool = True
    normalize_keys: bool = True
    memory_lr_init: float = 0.05       # Reduced from 0.1 for stability
    memory_lr_learnable: bool = True
    memory_decay_init: float = 0.9     # Reduced from 0.95 for faster forgetting
    memory_decay_learnable: bool = True
    max_memory_lr: float = 0.2         # Reduced from 0.5 for stability
    min_memory_decay: float = 0.5
    memory_max_norm: float = 15.0      # NEW: Cap memory norm to prevent explosion

    # === Nested Learning: CMS FFN ===
    use_cms_ffn: bool = True
    cms_num_levels: int = 2        # Reduced from 3 for memory
    cms_base_chunk: int = 64
    cms_chunk_multiplier: int = 2
    cms_aggregation: str = "learned"

    # === Nested Learning: Self-Referential Attention ===
    use_hope_attention: bool = True
    self_referential_keys: bool = True
    self_referential_values: bool = True
    generate_own_values: bool = True
    use_short_conv: bool = True
    conv_kernel_size: int = 4

    # === Optimizer ===
    optimizer_type: str = "m3"
    m3_beta_slow: float = 0.99
    m3_slow_freq: int = 50
    m3_slow_weight: float = 0.1


# =============================================================================
# Basic Components
# =============================================================================

class RMSNorm(nn.Module):
    """RMSNorm normalization layer"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLUFFN(nn.Module):
    """Standard SwiGLU FFN"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# =============================================================================
# Delta Memory Components
# =============================================================================

def normalize_for_delta(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


class LinearDeltaMemory(nn.Module):
    """Linear associative memory with delta rule updates"""

    def __init__(self, d_in: int, d_out: int, config: OnyxHopeSmallConfig):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.config = config

        # Initial memory matrix
        self.M_init = nn.Parameter(torch.zeros(d_out, d_in))
        nn.init.xavier_uniform_(self.M_init, gain=0.1)

        # Inverse sigmoid for bounded parameters
        def inv_sig(x):
            x = max(min(x, 0.999), 0.001)
            return math.log(x / (1 - x))

        # Learning rate (eta)
        if config.memory_lr_learnable:
            self.eta_raw = nn.Parameter(torch.tensor(inv_sig(config.memory_lr_init)))
        else:
            self.register_buffer('eta_raw', torch.tensor(inv_sig(config.memory_lr_init)))

        # Decay rate (alpha)
        if config.memory_decay_learnable:
            self.alpha_raw = nn.Parameter(torch.tensor(inv_sig(config.memory_decay_init)))
        else:
            self.register_buffer('alpha_raw', torch.tensor(inv_sig(config.memory_decay_init)))

    @property
    def eta(self) -> Tensor:
        return torch.sigmoid(self.eta_raw) * self.config.max_memory_lr

    @property
    def alpha(self) -> Tensor:
        min_d = self.config.min_memory_decay
        return min_d + torch.sigmoid(self.alpha_raw) * (1 - min_d)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.M_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(dtype=dtype)

    def retrieve(self, M: Tensor, query: Tensor) -> Tensor:
        # M: (B, d_out, d_in), query: (B, d_in) -> (B, d_out)
        return torch.matmul(M, query.unsqueeze(-1)).squeeze(-1)

    def update(self, M: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if self.config.normalize_keys:
            key = normalize_for_delta(key, dim=-1, eps=self.config.norm_eps)

        k = key.unsqueeze(-1)      # (B, d_in, 1)
        v = value.unsqueeze(-1)    # (B, d_out, 1)

        if self.config.use_delta_rule:
            # Delta rule: M' = alpha*M + eta*(v - M*k)*k^T
            Mk = torch.matmul(M, k)          # (B, d_out, 1)
            error = v - Mk                    # (B, d_out, 1)
            M_new = self.alpha * M + self.eta * torch.matmul(error, k.transpose(-1, -2))
        else:
            # Hebbian: M' = alpha*M + eta*v*k^T
            M_new = self.alpha * M + self.eta * torch.matmul(v, k.transpose(-1, -2))

        # STABILITY FIX: Cap memory norm to prevent explosion
        M_norm = torch.norm(M_new, dim=(-2, -1), keepdim=True)
        max_norm = self.config.memory_max_norm
        scale = torch.clamp(max_norm / (M_norm + 1e-6), max=1.0)
        M_new = M_new * scale

        return M_new

    def forward(self, x: Tensor, M: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, S, _ = x.shape
        if M is None:
            M = self.init_state(B, x.device, x.dtype)

        outputs = []
        for t in range(S):
            x_t = x[:, t]
            out_t = self.retrieve(M, x_t)
            outputs.append(out_t)
            M = self.update(M, x_t, out_t)

        return torch.stack(outputs, dim=1), M


class SelfReferentialMemory(nn.Module):
    """Memory that can generate its own training signal"""

    def __init__(self, d_in: int, d_out: int, config: OnyxHopeSmallConfig):
        super().__init__()
        self.memory = LinearDeltaMemory(d_in, d_out, config)

        if config.generate_own_values:
            self.value_gen = nn.Sequential(
                nn.Linear(d_out, d_out, bias=False),
                nn.SiLU(),
                nn.Linear(d_out, d_out, bias=False),
            )
            # Initialize as identity + small perturbation
            nn.init.eye_(self.value_gen[0].weight)
            nn.init.zeros_(self.value_gen[2].weight)
        else:
            self.value_gen = None

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.memory.init_state(batch_size, device, dtype)

    def forward(self, x: Tensor, M: Optional[Tensor] = None, value_input: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, S, _ = x.shape
        if M is None:
            M = self.init_state(B, x.device, x.dtype)

        outputs = []
        for t in range(S):
            x_t = x[:, t]
            out_t = self.memory.retrieve(M, x_t)
            outputs.append(out_t)

            # Generate value for memory update
            if self.value_gen is not None:
                base_v = value_input[:, t] if value_input is not None else out_t
                v_hat = self.value_gen(base_v) + base_v  # Residual
            else:
                v_hat = out_t

            M = self.memory.update(M, x_t, v_hat)

        return torch.stack(outputs, dim=1), M


# =============================================================================
# CMS FFN
# =============================================================================

class CMSFFN(nn.Module):
    """Chunked Multi-Scale FFN"""

    def __init__(self, config: OnyxHopeSmallConfig):
        super().__init__()
        self.config = config

        d_ff_per_level = config.d_ff // config.cms_num_levels

        self.levels = nn.ModuleList([
            SwiGLUFFN(config.d_model, d_ff_per_level, config.dropout)
            for _ in range(config.cms_num_levels)
        ])

        self.chunk_sizes = [
            config.cms_base_chunk * (config.cms_chunk_multiplier ** i)
            for i in range(config.cms_num_levels)
        ]

        if config.cms_aggregation == "learned":
            self.level_weights = nn.Parameter(torch.ones(config.cms_num_levels))
        else:
            self.register_buffer('level_weights', torch.ones(config.cms_num_levels))

        if config.cms_aggregation == "concat":
            self.out_proj = nn.Linear(config.d_model * config.cms_num_levels, config.d_model, bias=False)
        else:
            self.out_proj = None

    def forward(self, x: Tensor) -> Tensor:
        level_outputs = [level(x) for level in self.levels]

        if self.config.cms_aggregation == "learned":
            weights = F.softmax(self.level_weights, dim=0)
            output = sum(w * out for w, out in zip(weights, level_outputs))
        elif self.config.cms_aggregation == "mean":
            output = sum(level_outputs) / len(level_outputs)
        elif self.config.cms_aggregation == "concat":
            output = self.out_proj(torch.cat(level_outputs, dim=-1))
        else:
            output = sum(level_outputs) / len(level_outputs)

        return output


# =============================================================================
# Hope Attention
# =============================================================================

class HopeAttention(nn.Module):
    """Self-modifying attention with memory-based K/V projections"""

    def __init__(self, config: OnyxHopeSmallConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = self.d_model // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        kv_dim = self.n_kv_heads * self.head_dim

        # Query: Static projection
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)

        # Key: Memory-based or static
        if config.self_referential_keys:
            self.k_memory = SelfReferentialMemory(self.d_model, kv_dim, config)
            self.k_proj = None
        else:
            self.k_proj = nn.Linear(self.d_model, kv_dim, bias=False)
            self.k_memory = None

        # Value: Memory-based or static
        if config.self_referential_values:
            self.v_memory = SelfReferentialMemory(self.d_model, kv_dim, config)
            self.v_proj = None
        else:
            self.v_proj = nn.Linear(self.d_model, kv_dim, bias=False)
            self.v_memory = None

        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Short conv for local context
        if config.use_short_conv:
            self.short_conv = nn.Conv1d(
                self.d_model, self.d_model,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=self.d_model,
            )
        else:
            self.short_conv = None

        # QK normalization
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # RoPE frequencies
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self.dropout = nn.Dropout(config.attention_dropout)

    def _apply_rope(self, x: Tensor, offset: int = 0) -> Tensor:
        B, S, H, D = x.shape
        t = torch.arange(offset, offset + S, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)

        cos = torch.cos(freqs).to(x.dtype).view(1, S, 1, -1)
        sin = torch.sin(freqs).to(x.dtype).view(1, S, 1, -1)

        x_even, x_odd = x[..., ::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out

    def init_memory_states(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Tensor]:
        states = {}
        if self.k_memory is not None:
            states['k'] = self.k_memory.init_state(batch_size, device, dtype)
        if self.v_memory is not None:
            states['v'] = self.v_memory.init_state(batch_size, device, dtype)
        return states

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[Dict[str, Tensor]] = None,
        update_memories: bool = True,
        attn_mask: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tuple[Tensor, Tensor]]]:
        B, S, _ = x.shape

        if memory_states is None:
            memory_states = self.init_memory_states(B, x.device, x.dtype)

        # Short conv for local context
        if self.short_conv is not None:
            x_conv = self.short_conv(x.transpose(1, 2))[:, :, :S].transpose(1, 2)
            x = x + x_conv

        # Query (always static)
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)

        # Key (memory-based or static)
        new_states = {}
        if self.k_memory is not None:
            k, new_k = self.k_memory(x, memory_states.get('k'), value_input=x)
            k = k.view(B, S, self.n_kv_heads, self.head_dim)
            if update_memories:
                new_states['k'] = new_k
        else:
            k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        # Value (memory-based or static)
        if self.v_memory is not None:
            v, new_v = self.v_memory(x, memory_states.get('v'), value_input=x)
            v = v.view(B, S, self.n_kv_heads, self.head_dim)
            if update_memories:
                new_states['v'] = new_v
        else:
            v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        # RoPE
        offset = past_kv[0].shape[1] if past_kv is not None else 0
        q = self._apply_rope(q, offset)
        k = self._apply_rope(k, offset)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # KV cache handling
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)
        new_kv = (k, v) if use_cache else None

        # GQA repeat for attention
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose for attention: (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention mask handling
        mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                mask = torch.zeros_like(attn_mask, dtype=torch.float32)
                mask.masked_fill_(~attn_mask, float('-inf'))
                mask = mask.unsqueeze(1)
            else:
                mask = attn_mask.unsqueeze(1) if attn_mask.dim() == 3 else attn_mask

        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=(mask is None)
        )

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        output = self.o_proj(attn_out)

        if not update_memories:
            new_states = memory_states

        return output, new_states, new_kv


# =============================================================================
# Hope Block
# =============================================================================

class HopeBlock(nn.Module):
    """Full Hope transformer block"""

    def __init__(self, config: OnyxHopeSmallConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)

        if config.use_hope_attention:
            self.attention = HopeAttention(config, layer_idx)
        else:
            self.attention = StandardAttention(config, layer_idx)

        if config.use_cms_ffn:
            self.ffn = CMSFFN(config)
        else:
            self.ffn = SwiGLUFFN(config.d_model, config.d_ff, config.dropout)

    def init_memory_states(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        if hasattr(self.attention, 'init_memory_states'):
            return {'attention': self.attention.init_memory_states(batch_size, device, dtype)}
        return {}

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[Dict[str, Any]] = None,
        update_memories: bool = True,
        attn_mask: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any], Optional[Tuple[Tensor, Tensor]]]:
        B = x.shape[0]

        if memory_states is None:
            memory_states = self.init_memory_states(B, x.device, x.dtype)

        # Attention with pre-norm
        residual = x
        x = self.norm1(x)

        if hasattr(self.attention, 'init_memory_states'):
            attn_out, new_attn_states, new_kv = self.attention(
                x, memory_states.get('attention'), update_memories, attn_mask, past_kv, use_cache
            )
        else:
            attn_out, new_kv = self.attention(x, attn_mask=attn_mask, past_kv=past_kv, use_cache=use_cache)
            new_attn_states = {}

        x = residual + attn_out

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        new_states = {'attention': new_attn_states}
        return x, new_states, new_kv


class StandardAttention(nn.Module):
    """Standard attention fallback (when Hope is disabled)"""

    def __init__(self, config: OnyxHopeSmallConfig, layer_idx: int = 0):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def _apply_rope(self, x: Tensor, offset: int = 0) -> Tensor:
        B, S, H, D = x.shape
        t = torch.arange(offset, offset + S, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = torch.cos(freqs).to(x.dtype).view(1, S, 1, -1)
        sin = torch.sin(freqs).to(x.dtype).view(1, S, 1, -1)
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        offset = past_kv[0].shape[1] if past_kv else 0
        q = self._apply_rope(q, offset)
        k = self._apply_rope(k, offset)

        if past_kv:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)
        new_kv = (k, v) if use_cache else None

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        return self.o_proj(attn_out), new_kv


# =============================================================================
# Full Model
# =============================================================================

class OnyxHopeSmall(nn.Module):
    """
    OnyxHope Small (~25-30M params) for M2 MacBook Pro training
    """

    def __init__(self, config: OnyxHopeSmallConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            HopeBlock(config, i) for i in range(config.n_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.d_model, config.norm_eps)

        # LM head
        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_all_memory_states(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """Initialize memory states for all layers"""
        return [
            layer.init_memory_states(batch_size, device, dtype)
            for layer in self.layers
        ]

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        memory_states: Optional[List[Dict[str, Any]]] = None,
        update_memories: bool = True,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        B, S = input_ids.shape
        device, dtype = input_ids.device, self.embed_tokens.weight.dtype

        # Initialize memory states if needed
        if memory_states is None and self.config.use_nested_learning:
            memory_states = self.init_all_memory_states(B, device, dtype)

        # Embeddings
        hidden = self.embed_tokens(input_ids)

        # Process layers
        new_memory_states = []
        new_past_kv = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_mem = memory_states[i] if memory_states else None
            past_kv = past_key_values[i] if past_key_values else None

            hidden, new_mem, new_kv = layer(
                hidden,
                memory_states=layer_mem,
                update_memories=update_memories,
                attn_mask=attn_mask,
                past_kv=past_kv,
                use_cache=use_cache,
            )

            new_memory_states.append(new_mem)
            if use_cache:
                new_past_kv.append(new_kv)

        # Final norm
        hidden = self.norm(hidden)

        # LM head
        if self.lm_head is None:
            logits = F.linear(hidden, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden)

        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        if return_dict:
            return {
                'logits': logits,
                'loss': loss,
                'memory_states': new_memory_states,
                'past_key_values': new_past_kv,
                'hidden_states': hidden,
            }

        return logits

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Get parameter groups for optimizer"""
        decay, no_decay = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # No decay for norms, biases, embeddings, and memory hyperparams
            if any(k in name.lower() for k in ['norm', 'bias', 'embed', 'eta', 'alpha', 'level_weight']):
                no_decay.append(param)
            else:
                decay.append(param)

        return {'decay': decay, 'no_decay': no_decay}


# =============================================================================
# M3 Optimizer
# =============================================================================

class M3Optimizer(torch.optim.Optimizer):
    """Multi-scale Momentum optimizer for Nested Learning"""

    def __init__(
        self,
        params,
        lr: float = 6e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        beta_slow: float = 0.99,
        slow_freq: int = 50,
        slow_weight: float = 0.1,
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        defaults = dict(
            lr=lr, betas=betas, beta_slow=beta_slow,
            slow_freq=slow_freq, slow_weight=slow_weight,
            eps=eps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta_slow = group['beta_slow']
            slow_freq = group['slow_freq']
            slow_weight = group['slow_weight']
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m1'] = torch.zeros_like(p)      # Fast momentum
                    state['m2'] = torch.zeros_like(p)      # Slow momentum
                    state['v'] = torch.zeros_like(p)       # Second moment
                    state['grad_buf'] = torch.zeros_like(p)  # Gradient buffer

                m1, m2, v = state['m1'], state['m2'], state['v']
                grad_buf = state['grad_buf']
                state['step'] += 1

                # Fast momentum update
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                grad_buf.add_(grad)

                # Slow momentum update (every slow_freq steps)
                if self._step_count % slow_freq == 0:
                    m2.mul_(beta_slow).add_(grad_buf, alpha=1 - beta_slow)
                    grad_buf.zero_()

                # Bias correction
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']

                m1_hat = m1 / bc1
                v_hat = v / bc2

                # Combine fast and slow momentum
                combined = m1_hat
                if self._step_count > slow_freq:
                    combined = combined + slow_weight * m2

                denom = v_hat.sqrt().add_(eps)

                # Weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # Update
                p.addcdiv_(combined, denom, value=-lr)

        return loss


def create_m3_optimizer(model: OnyxHopeSmall, lr: float = 6e-4, weight_decay: float = 0.1, **kwargs) -> M3Optimizer:
    """Create M3 optimizer with proper parameter groups"""
    groups = model.get_param_groups()

    param_groups = [
        {'params': groups['decay'], 'weight_decay': weight_decay},
        {'params': groups['no_decay'], 'weight_decay': 0.0},
    ]

    return M3Optimizer(param_groups, lr=lr, **kwargs)


# =============================================================================
# Factory
# =============================================================================

def create_onyx_hope_small(
    device: str = "mps",
    dtype: Optional[torch.dtype] = None,
    **config_overrides
) -> OnyxHopeSmall:
    """Create OnyxHope Small model optimized for M2 MacBook Pro"""

    # Default to float32 for MPS stability
    if dtype is None:
        dtype = torch.float32

    config = OnyxHopeSmallConfig(**config_overrides)
    model = OnyxHopeSmall(config)
    model = model.to(device=device, dtype=dtype)

    print(f"Created OnyxHope Small")
    print(f"   Parameters: {model.get_num_params():,}")
    print(f"   Config: d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
    print(f"   FFN: d_ff={config.d_ff}, CMS levels={config.cms_num_levels}")
    print(f"   Sequence length: {config.max_seq_len}")
    print(f"   Hope Attention: {config.use_hope_attention}")
    print(f"   Self-referential K/V: {config.self_referential_keys}/{config.self_referential_values}")
    print(f"   Device: {device}, Dtype: {dtype}")

    return model


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing OnyxHope Small (~25-30M params)")
    print("=" * 60)

    # Use MPS if available, else CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon MPS")
    else:
        device = "cpu"
        print("Using CPU")

    dtype = torch.float32

    # Create model
    model = create_onyx_hope_small(device=device, dtype=dtype)

    # Test forward
    print("\nTesting forward pass...")
    B, S = 2, 128
    input_ids = torch.randint(0, 1000, (B, S), device=device)
    labels = torch.randint(0, 1000, (B, S), device=device)

    outputs = model(input_ids, labels=labels)
    print(f"   Logits: {outputs['logits'].shape}")
    print(f"   Loss: {outputs['loss'].item():.4f}")
    print(f"   Memory states: {len(outputs['memory_states'])} layers")

    # Test gradient flow
    print("\nTesting gradient flow...")
    outputs['loss'].backward()

    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    print(f"   Total gradient norm: {total_grad_norm:.4f}")

    # Test M3 optimizer
    print("\nTesting M3 optimizer...")
    optimizer = create_m3_optimizer(model, lr=1e-4)
    optimizer.step()
    optimizer.zero_grad()
    print(f"   Optimizer step count: {optimizer._step_count}")

    # Memory usage estimate
    param_bytes = model.get_num_params() * 4  # float32
    print(f"\nMemory estimates:")
    print(f"   Model params: {param_bytes / 1e6:.1f} MB")
    print(f"   With gradients: ~{param_bytes * 2 / 1e6:.1f} MB")
    print(f"   With optimizer state: ~{param_bytes * 4 / 1e6:.1f} MB")

    print("\n" + "=" * 60)
    print("All tests passed!")
