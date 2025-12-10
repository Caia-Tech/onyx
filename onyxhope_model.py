#!/usr/bin/env python3
"""
OnyxHope: Onyx model with Nested Learning integration

This combines the base Onyx architecture with:
1. Hope Attention (self-modifying K/V projections)
2. CMS FFN (multi-frequency MLP)
3. M3 Optimizer support

Usage:
    from model_onyx_hope import OnyxHope, OnyxHopeConfig, create_m3_optimizer
    
    config = OnyxHopeConfig(use_nested_learning=True)
    model = OnyxHope(config)
    optimizer = create_m3_optimizer(model)
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import base model components (adjust path as needed)
try:
    from model_onyx125m import (
        OnyxConfig, RMSNorm, RoPE,
        OptimizedAttention, OptimizedFFN, TransformerBlock,
        FLASH_AVAILABLE
    )
    BASE_MODEL_AVAILABLE = True
except ImportError:
    BASE_MODEL_AVAILABLE = False
    warnings.warn("Base Onyx model not found. Using standalone components.")

# Import nested learning components
try:
    from nested_learning import (
        NestedLearningConfig, LinearDeltaMemory, SelfReferentialMemory,
        CMSFFN, HopeAttention, HopeBlock, M3Optimizer,
        normalize_for_delta, create_m3_optimizer
    )
    NESTED_LEARNING_AVAILABLE = True
except ImportError:
    NESTED_LEARNING_AVAILABLE = False
    warnings.warn("Nested learning module not found.")


# =============================================================================
# Integrated Config
# =============================================================================

@dataclass
class OnyxHopeConfig:
    """Configuration for OnyxHope model combining base Onyx with Nested Learning"""
    
    # === Base Onyx Config ===
    vocab_size: int = 128258
    d_model: int = 960
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    d_ff: int = 3840
    max_seq_len: int = 2048
    
    eos_token_id: int = 2
    pad_token_id: int = 0
    eod_token_id: int = 3
    
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    use_swiglu: bool = True
    use_rms_norm: bool = True
    norm_eps: float = 1e-5
    
    use_flash_attn: bool = True
    use_torch_compile: bool = False
    
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
    memory_lr_init: float = 0.1
    memory_lr_learnable: bool = True
    memory_decay_init: float = 0.95
    memory_decay_learnable: bool = True
    max_memory_lr: float = 0.5
    min_memory_decay: float = 0.5
    
    # === Nested Learning: CMS FFN ===
    use_cms_ffn: bool = True
    cms_num_levels: int = 3
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
    optimizer_type: str = "m3"  # "adamw" or "m3"
    m3_beta_slow: float = 0.99
    m3_slow_freq: int = 50
    m3_slow_weight: float = 0.1
    
    def to_nested_config(self):
        """Convert to NestedLearningConfig"""
        if not NESTED_LEARNING_AVAILABLE:
            raise RuntimeError("nested_learning module not available")
        
        return NestedLearningConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            d_ff=self.d_ff,
            memory_type=self.memory_type,
            use_delta_rule=self.use_delta_rule,
            normalize_keys=self.normalize_keys,
            memory_lr_init=self.memory_lr_init,
            memory_lr_learnable=self.memory_lr_learnable,
            memory_decay_init=self.memory_decay_init,
            memory_decay_learnable=self.memory_decay_learnable,
            max_memory_lr=self.max_memory_lr,
            min_memory_decay=self.min_memory_decay,
            cms_num_levels=self.cms_num_levels,
            cms_base_chunk=self.cms_base_chunk,
            cms_chunk_multiplier=self.cms_chunk_multiplier,
            cms_aggregation=self.cms_aggregation,
            self_referential_keys=self.self_referential_keys,
            self_referential_values=self.self_referential_values,
            generate_own_values=self.generate_own_values,
            use_short_conv=self.use_short_conv,
            conv_kernel_size=self.conv_kernel_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            eps=self.norm_eps,
        )


# =============================================================================
# Standalone Components (if base model not available)
# =============================================================================

class RMSNormStandalone(nn.Module):
    """RMSNorm if not imported from base"""
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
# Inline Nested Learning Components (if module not available)
# =============================================================================

def _normalize_for_delta(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


class LinearDeltaMemoryInline(nn.Module):
    """Inline version of LinearDeltaMemory"""
    
    def __init__(self, d_in: int, d_out: int, config: OnyxHopeConfig):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.config = config
        
        self.M_init = nn.Parameter(torch.zeros(d_out, d_in))
        nn.init.xavier_uniform_(self.M_init, gain=0.1)
        
        def inv_sig(x):
            x = max(min(x, 0.999), 0.001)
            return math.log(x / (1 - x))
        
        if config.memory_lr_learnable:
            self.eta_raw = nn.Parameter(torch.tensor(inv_sig(config.memory_lr_init)))
        else:
            self.register_buffer('eta_raw', torch.tensor(inv_sig(config.memory_lr_init)))
        
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
        return torch.matmul(M, query.unsqueeze(-1)).squeeze(-1)
    
    def update(self, M: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if self.config.normalize_keys:
            key = _normalize_for_delta(key, dim=-1, eps=self.config.norm_eps)
        
        k = key.unsqueeze(-1)
        v = value.unsqueeze(-1)
        
        if self.config.use_delta_rule:
            Mk = torch.matmul(M, k)
            error = v - Mk
            M_new = self.alpha * M + self.eta * torch.matmul(error, k.transpose(-1, -2))
        else:
            M_new = self.alpha * M + self.eta * torch.matmul(v, k.transpose(-1, -2))
        
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


class SelfReferentialMemoryInline(nn.Module):
    """Inline version of SelfReferentialMemory"""
    
    def __init__(self, d_in: int, d_out: int, config: OnyxHopeConfig):
        super().__init__()
        self.memory = LinearDeltaMemoryInline(d_in, d_out, config)
        
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
    
    def forward(self, x: Tensor, M: Optional[Tensor] = None, value_input: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, S, _ = x.shape
        if M is None:
            M = self.init_state(B, x.device, x.dtype)
        
        outputs = []
        for t in range(S):
            x_t = x[:, t]
            out_t = self.memory.retrieve(M, x_t)
            outputs.append(out_t)
            
            if self.value_gen is not None:
                base_v = value_input[:, t] if value_input is not None else out_t
                v_hat = self.value_gen(base_v) + base_v
            else:
                v_hat = out_t
            
            M = self.memory.update(M, x_t, v_hat)
        
        return torch.stack(outputs, dim=1), M


class CMSFFNInline(nn.Module):
    """Inline version of CMSFFN"""
    
    def __init__(self, config: OnyxHopeConfig):
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
        
        return output


# =============================================================================
# Hope Attention (Inline)
# =============================================================================

class HopeAttentionInline(nn.Module):
    """Self-modifying attention with memory-based K/V projections"""
    
    def __init__(self, config: OnyxHopeConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = self.d_model // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        kv_dim = self.n_kv_heads * self.head_dim
        
        # Query: Static
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        
        # Key/Value: Memory-based or static
        if config.self_referential_keys:
            self.k_memory = SelfReferentialMemoryInline(self.d_model, kv_dim, config)
            self.k_proj = None
        else:
            self.k_proj = nn.Linear(self.d_model, kv_dim, bias=False)
            self.k_memory = None
        
        if config.self_referential_values:
            self.v_memory = SelfReferentialMemoryInline(self.d_model, kv_dim, config)
            self.v_proj = None
        else:
            self.v_proj = nn.Linear(self.d_model, kv_dim, bias=False)
            self.v_memory = None
        
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # Short conv
        if config.use_short_conv:
            self.short_conv = nn.Conv1d(
                self.d_model, self.d_model,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=self.d_model,
            )
        else:
            self.short_conv = None
        
        # QK norm
        self.q_norm = RMSNormStandalone(self.head_dim)
        self.k_norm = RMSNormStandalone(self.head_dim)
        
        # RoPE
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
        
        # Short conv
        if self.short_conv is not None:
            x_conv = self.short_conv(x.transpose(1, 2))[:, :, :S].transpose(1, 2)
            x = x + x_conv
        
        # Query
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        
        # Key
        new_states = {}
        if self.k_memory is not None:
            k, new_k = self.k_memory(x, memory_states.get('k'), value_input=x)
            k = k.view(B, S, self.n_kv_heads, self.head_dim)
            if update_memories:
                new_states['k'] = new_k
        else:
            k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        
        # Value
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
        
        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # KV cache
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)
        new_kv = (k, v) if use_cache else None
        
        # GQA repeat
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
        
        # Attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # SDPA (Flash Attention handled automatically if available)
        mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                mask = torch.zeros_like(attn_mask, dtype=torch.float32)
                mask.masked_fill_(~attn_mask, float('-inf'))
                mask = mask.unsqueeze(1)
            else:
                mask = attn_mask.unsqueeze(1) if attn_mask.dim() == 3 else attn_mask
        
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=(mask is None)
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        output = self.o_proj(attn_out)
        
        if not update_memories:
            new_states = memory_states
        
        return output, new_states, new_kv


# =============================================================================
# Hope Block (Inline)
# =============================================================================

class HopeBlockInline(nn.Module):
    """Full Hope transformer block"""
    
    def __init__(self, config: OnyxHopeConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.norm1 = RMSNormStandalone(config.d_model, eps=config.norm_eps)
        self.norm2 = RMSNormStandalone(config.d_model, eps=config.norm_eps)
        
        if config.use_hope_attention:
            self.attention = HopeAttentionInline(config, layer_idx)
        else:
            # Fall back to standard attention if available
            self.attention = None  # Would need to import from base
        
        if config.use_cms_ffn:
            self.ffn = CMSFFNInline(config)
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
        
        # Attention
        residual = x
        x = self.norm1(x)
        
        if hasattr(self.attention, 'init_memory_states'):
            attn_out, new_attn_states, new_kv = self.attention(
                x, memory_states.get('attention'), update_memories, attn_mask, past_kv, use_cache
            )
        else:
            # Standard attention path
            attn_out, new_kv = self.attention(x, use_cache=use_cache, past_kv=past_kv, attn_mask=attn_mask)
            new_attn_states = {}
        
        x = residual + attn_out
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        new_states = {'attention': new_attn_states}
        return x, new_states, new_kv


# =============================================================================
# Full OnyxHope Model
# =============================================================================

class OnyxHope(nn.Module):
    """
    Onyx model with Nested Learning integration.
    
    When use_nested_learning=True:
    - Attention uses self-modifying K/V projections (Hope Attention)
    - FFN uses multi-frequency CMS
    - Optimizer should be M3
    
    When use_nested_learning=False:
    - Standard Transformer (same as base Onyx)
    """
    
    def __init__(self, config: OnyxHopeConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer layers
        if config.use_nested_learning:
            self.layers = nn.ModuleList([
                HopeBlockInline(config, i) for i in range(config.n_layers)
            ])
        else:
            # Standard transformer blocks
            self.layers = nn.ModuleList([
                self._create_standard_block(config, i) for i in range(config.n_layers)
            ])
        
        # Final norm
        self.norm = RMSNormStandalone(config.d_model, config.norm_eps)
        
        # LM head
        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize
        self.apply(self._init_weights)
    
    def _create_standard_block(self, config: OnyxHopeConfig, layer_idx: int) -> nn.Module:
        """Create standard transformer block (fallback when NL disabled)"""
        # Simple standard block
        class StandardBlock(nn.Module):
            def __init__(self, cfg, idx):
                super().__init__()
                self.norm1 = RMSNormStandalone(cfg.d_model, cfg.norm_eps)
                self.norm2 = RMSNormStandalone(cfg.d_model, cfg.norm_eps)
                
                # Standard attention
                self.n_heads = cfg.n_heads
                self.head_dim = cfg.d_model // cfg.n_heads
                self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
                self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
                self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
                self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                self.n_rep = cfg.n_heads // cfg.n_kv_heads
                
                # FFN
                self.ffn = SwiGLUFFN(cfg.d_model, cfg.d_ff, cfg.dropout)
                
                # RoPE
                inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
                self.register_buffer('inv_freq', inv_freq, persistent=False)
            
            def _apply_rope(self, x, offset=0):
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
            
            def forward(self, x, memory_states=None, update_memories=True, attn_mask=None, past_kv=None, use_cache=False):
                B, S, D = x.shape
                
                residual = x
                x = self.norm1(x)
                
                q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
                k = self.k_proj(x).view(B, S, -1, self.head_dim)
                v = self.v_proj(x).view(B, S, -1, self.head_dim)
                
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
                x = residual + self.o_proj(attn_out)
                
                residual = x
                x = self.norm2(x)
                x = residual + self.ffn(x)
                
                return x, {}, new_kv
        
        return StandardBlock(config, layer_idx)
    
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
            if hasattr(layer, 'init_memory_states') else {}
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
        """
        Forward pass.
        
        Args:
            input_ids: (B, S)
            labels: (B, S) for loss computation
            memory_states: List of memory state dicts per layer
            update_memories: Whether to update memories during forward
            use_cache: Return KV cache
            past_key_values: KV cache from previous forward
        
        Returns:
            Dict with 'logits', 'loss', 'memory_states', 'past_key_values'
        """
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
            if any(k in name.lower() for k in ['norm', 'bias', 'embed', 'eta', 'alpha', 'level_weight']):
                no_decay.append(param)
            else:
                decay.append(param)
        
        return {'decay': decay, 'no_decay': no_decay}


# =============================================================================
# M3 Optimizer (Inline)
# =============================================================================

class M3OptimizerInline(torch.optim.Optimizer):
    """Multi-scale Momentum optimizer"""
    
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
                
                if len(state) == 0:
                    state['step'] = 0
                    state['m1'] = torch.zeros_like(p)
                    state['m2'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['grad_buf'] = torch.zeros_like(p)
                
                m1, m2, v = state['m1'], state['m2'], state['v']
                grad_buf = state['grad_buf']
                state['step'] += 1
                
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                grad_buf.add_(grad)
                
                if self._step_count % slow_freq == 0:
                    m2.mul_(beta_slow).add_(grad_buf, alpha=1 - beta_slow)
                    grad_buf.zero_()
                
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                
                m1_hat = m1 / bc1
                v_hat = v / bc2
                
                combined = m1_hat
                if self._step_count > slow_freq:
                    combined = combined + slow_weight * m2
                
                denom = v_hat.sqrt().add_(eps)
                
                if wd > 0:
                    p.mul_(1 - lr * wd)
                
                p.addcdiv_(combined, denom, value=-lr)
        
        return loss


def create_m3_optimizer_for_model(model: OnyxHope, lr: float = 6e-4, weight_decay: float = 0.1, **kwargs) -> M3OptimizerInline:
    """Create M3 optimizer with proper parameter groups"""
    groups = model.get_param_groups()
    
    param_groups = [
        {'params': groups['decay'], 'weight_decay': weight_decay},
        {'params': groups['no_decay'], 'weight_decay': 0.0},
    ]
    
    return M3OptimizerInline(param_groups, lr=lr, **kwargs)


# =============================================================================
# Factory
# =============================================================================

def create_onyx_hope(
    use_nested_learning: bool = True,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    **config_overrides
) -> OnyxHope:
    """Create OnyxHope model"""
    
    if dtype is None:
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32
    
    config = OnyxHopeConfig(use_nested_learning=use_nested_learning, **config_overrides)
    model = OnyxHope(config)
    model = model.to(device=device, dtype=dtype)
    
    print(f"✅ Created OnyxHope Model")
    print(f"   Parameters: {model.get_num_params():,}")
    print(f"   Nested Learning: {config.use_nested_learning}")
    if config.use_nested_learning:
        print(f"   Hope Attention: {config.use_hope_attention}")
        print(f"   CMS FFN: {config.use_cms_ffn} ({config.cms_num_levels} levels)")
        print(f"   Self-referential K/V: {config.self_referential_keys}/{config.self_referential_values}")
    print(f"   Device: {device}, Dtype: {dtype}")
    
    return model


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing OnyxHope Model")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Use float32 for CPU testing
    
    # Create model
    model = create_onyx_hope(
        use_nested_learning=True,
        device=device,
        dtype=dtype,
        n_layers=4,  # Smaller for testing
    )
    
    # Test forward
    print("\nTesting forward pass...")
    B, S = 2, 64
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
    optimizer = create_m3_optimizer_for_model(model, lr=1e-4)
    optimizer.step()
    optimizer.zero_grad()
    print(f"   Optimizer step count: {optimizer._step_count}")
    
    # Test with standard model (NL disabled)
    print("\nTesting standard model (NL disabled)...")
    model_std = create_onyx_hope(
        use_nested_learning=False,
        device=device,
        dtype=dtype,
        n_layers=4,
    )
    
    outputs_std = model_std(input_ids, labels=labels)
    print(f"   Logits: {outputs_std['logits'].shape}")
    print(f"   Loss: {outputs_std['loss'].item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
