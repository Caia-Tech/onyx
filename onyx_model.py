"""
Onyx 11M - Hope Attention Transformer with Persistent Memory

A small model designed for rapid iteration and growth.

Architecture:
- ~11M parameters (excluding vocab embeddings)
- Hope Attention with self-modifying K/V memory (delta rule)
- Chunked parallel memory processing
- Persistent memory across inference sessions
- CMS FFN (Chunked Multi-Scale Feed-Forward)
- M3 Optimizer support
- GQA (Grouped Query Attention) for memory efficiency
- RoPE with extended context support

Memory Modes:
- stateless: Fresh memory each call (standard LLM behavior)
- session: Memory persists within conversation
- persistent: Memory saved/loaded across sessions

Author: Marvin Tutt, Caia Tech
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from flash_attn import flash_attn_func
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OnyxConfig:
    """Configuration for Onyx 11M model"""

    # === Core Architecture (~11M params) ===
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2           # 3:1 GQA
    head_dim: int = 64
    d_ff: int = 4096              # ~2.67x d_model

    # === Sequence & Vocab ===
    max_seq_len: int = 4096
    train_seq_len: int = 4096
    vocab_size: int = 128258      # Updated to match strict Llama-3 spec

    # === RoPE ===
    rope_base: float = 500000.0

    # === Precision & Performance ===
    use_flash_attention: bool = False
    use_torch_compile: bool = False
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

    # === Delta Memory (Training) ===
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

    # === CMS FFN ===
    use_cms_ffn: bool = True
    cms_num_levels: int = 4
    cms_base_chunk: int = 32
    cms_chunk_multiplier: int = 2
    cms_aggregation: str = "learned"

    # === M3 Optimizer ===
    optimizer_type: str = "m3"
    m3_beta_slow: float = 0.99
    m3_slow_freq: int = 50
    m3_slow_weight: float = 0.1


# =============================================================================
# Basic Components
# =============================================================================

class RMSNorm(nn.Module):
    """RMSNorm normalization layer"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with extended context support"""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._update_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype)
        )


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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
# Chunked Parallel Delta Memory
# =============================================================================

def normalize_for_delta(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


class ChunkedLinearDeltaMemory(nn.Module):
    """Linear associative memory with delta rule updates - chunked parallel."""

    def __init__(self, d_in: int, d_out: int, config: OnyxConfig):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.config = config
        self.chunk_size = config.memory_chunk_size

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

        self.register_buffer('inference_eta', torch.tensor(config.inference_memory_lr))
        self.register_buffer('inference_alpha', torch.tensor(config.inference_memory_decay))

        if config.use_memory_gate:
            self.gate_proj = nn.Linear(d_in, 1, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)

    @property
    def eta(self) -> Tensor:
        return torch.sigmoid(self.eta_raw) * self.config.max_memory_lr

    @property
    def alpha(self) -> Tensor:
        min_d = self.config.min_memory_decay
        return min_d + torch.sigmoid(self.alpha_raw) * (1 - min_d)

    def get_eta(self, inference_mode: bool = False) -> Tensor:
        if inference_mode:
            return self.inference_eta
        return self.eta

    def get_alpha(self, inference_mode: bool = False) -> Tensor:
        if inference_mode:
            return self.inference_alpha
        return self.alpha

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.M_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(dtype=dtype)

    def retrieve_batch(self, M: Tensor, queries: Tensor) -> Tensor:
        return torch.bmm(queries, M.transpose(-1, -2))

    def _process_chunk(
        self,
        M: Tensor,
        chunk: Tensor,
        eta: Tensor,
        alpha: Tensor,
        update_memory: bool
    ) -> Tuple[Tensor, Tensor]:
        B, C, d_in = chunk.shape
        outputs = self.retrieve_batch(M, chunk)

        if not update_memory:
            return outputs, M

        if self.config.normalize_keys:
            keys = normalize_for_delta(chunk, dim=-1, eps=self.config.norm_eps)
        else:
            keys = chunk

        if self.config.use_delta_rule:
            k_mean = keys.mean(dim=1)
            v_mean = outputs.mean(dim=1)
            Mk = torch.bmm(M, k_mean.unsqueeze(-1)).squeeze(-1)
            error = v_mean - Mk
            update = torch.bmm(error.unsqueeze(-1), k_mean.unsqueeze(1))
            chunk_scale = min(C / self.chunk_size, 1.0)
            M_new = alpha * M + eta * chunk_scale * update
        else:
            k_mean = keys.mean(dim=1)
            v_mean = outputs.mean(dim=1)
            update = torch.bmm(v_mean.unsqueeze(-1), k_mean.unsqueeze(1))
            chunk_scale = min(C / self.chunk_size, 1.0)
            M_new = alpha * M + eta * chunk_scale * update

        if hasattr(self, 'gate_proj'):
            gate = torch.sigmoid(self.gate_proj(k_mean))
            gate = gate.unsqueeze(-1)
            M_new = gate * M_new + (1 - gate) * M

        M_norm = torch.norm(M_new, dim=(-2, -1), keepdim=True)
        max_norm = self.config.memory_max_norm
        scale = torch.clamp(max_norm / (M_norm + 1e-6), max=1.0)
        M_new = M_new * scale

        return outputs, M_new

    def forward(
        self,
        x: Tensor,
        M: Optional[Tensor] = None,
        inference_mode: bool = False,
        update_memory: bool = True
    ) -> Tuple[Tensor, Tensor]:
        B, S, d_in = x.shape

        # Initialize or resize memory to match batch size
        if M is None or M.size(0) != B:
            M = self.init_state(B, x.device, x.dtype)

        eta = self.get_eta(inference_mode)
        alpha = self.get_alpha(inference_mode)

        outputs = []
        for start in range(0, S, self.chunk_size):
            end = min(start + self.chunk_size, S)
            chunk = x[:, start:end, :]
            chunk_out, M = self._process_chunk(M, chunk, eta, alpha, update_memory)
            outputs.append(chunk_out)

        return torch.cat(outputs, dim=1), M

    def get_memory_reg_loss(self, M: Tensor) -> Tensor:
        return M.pow(2).mean()


class ChunkedSelfReferentialMemory(nn.Module):
    """Memory that can generate its own training signal"""

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
        update_memory: bool = True
    ) -> Tuple[Tensor, Tensor]:
        B, S, _ = x.shape

        # Initialize or resize memory to match batch size
        if M is None or M.size(0) != B:
            M = self.init_state(B, x.device, x.dtype)

        outputs, M_new = self.memory(x, M, inference_mode, update_memory)

        if self.value_gen is not None:
            base_v = value_input if value_input is not None else outputs
            v_hat = self.value_gen(base_v) + base_v
            return v_hat, M_new

        return outputs, M_new

    def get_memory_reg_loss(self, M: Tensor) -> Tensor:
        return self.memory.get_memory_reg_loss(M)


# =============================================================================
# CMS FFN (Chunked Multi-Scale)
# =============================================================================

class CMSFFN(nn.Module):
    """Chunked Multi-Scale FFN"""

    def __init__(self, config: OnyxConfig):
        super().__init__()
        self.config = config
        self.num_levels = config.cms_num_levels
        self.d_model = config.d_model
        self.d_ff = config.d_ff

        self.chunk_sizes = [
            config.cms_base_chunk * (config.cms_chunk_multiplier ** i)
            for i in range(self.num_levels)
        ]

        self.level_ffns = nn.ModuleList([
            SwiGLUFFN(config.d_model, config.d_ff // self.num_levels, config.dropout)
            for _ in range(self.num_levels)
        ])

        if config.cms_aggregation == "learned":
            self.level_weights = nn.Parameter(torch.ones(self.num_levels) / self.num_levels)
        else:
            self.register_buffer('level_weights', torch.ones(self.num_levels) / self.num_levels)

        self.output_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        weights = F.softmax(self.level_weights, dim=0)

        level_outputs = []
        for level_idx, (chunk_size, ffn) in enumerate(zip(self.chunk_sizes, self.level_ffns)):
            if chunk_size >= S:
                out = ffn(x)
            else:
                num_chunks = (S + chunk_size - 1) // chunk_size
                padded_len = num_chunks * chunk_size
                if padded_len > S:
                    x_padded = F.pad(x, (0, 0, 0, padded_len - S))
                else:
                    x_padded = x

                chunks = x_padded.view(B, num_chunks, chunk_size, D)
                processed = ffn(chunks.view(B * num_chunks, chunk_size, D))
                processed = processed.view(B, num_chunks, chunk_size, D)
                out = processed.view(B, padded_len, D)[:, :S, :]

            level_outputs.append(out * weights[level_idx])

        combined = sum(level_outputs)
        return self.output_proj(combined)


# =============================================================================
# Attention Mechanisms
# =============================================================================

class HopeAttention(nn.Module):
    """Hope Attention with self-modifying K/V memory"""

    def __init__(self, config: OnyxConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)

        if config.use_short_conv:
            self.conv = nn.Conv1d(
                config.d_model, config.d_model,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.d_model
            )
        else:
            self.conv = None

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
        if hasattr(self, 'k_memory'):
            states['k'] = self.k_memory.init_state(batch_size, device, dtype)
        if hasattr(self, 'v_memory'):
            states['v'] = self.v_memory.init_state(batch_size, device, dtype)
        return states

    def get_memory_reg_loss(self, memory_states: Dict[str, Tensor]) -> Tensor:
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if hasattr(self, 'k_memory') and 'k' in memory_states:
            loss = loss + self.k_memory.get_memory_reg_loss(memory_states['k'])
        if hasattr(self, 'v_memory') and 'v' in memory_states:
            loss = loss + self.v_memory.get_memory_reg_loss(memory_states['v'])
        return loss

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[Dict[str, Tensor]] = None,
        update_memories: bool = True,
        inference_mode: bool = False,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        kv_cache: Optional[Dict[str, Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Dict[str, Tensor]]]:
        B, S, _ = x.shape

        if self.conv is not None:
            x_conv = self.conv(x.transpose(1, 2))[:, :, :S].transpose(1, 2)
            x = x + x_conv

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)

        new_memory_states = {}

        if hasattr(self, 'k_memory'):
            k_mem = memory_states.get('k') if memory_states else None
            k, new_k_mem = self.k_memory(
                x, k_mem,
                inference_mode=inference_mode,
                update_memory=update_memories
            )
            k = k.view(B, S, self.n_kv_heads, self.head_dim)
            new_memory_states['k'] = new_k_mem
        else:
            k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        if hasattr(self, 'v_memory'):
            v_mem = memory_states.get('v') if memory_states else None
            v, new_v_mem = self.v_memory(
                x, v_mem,
                inference_mode=inference_mode,
                update_memory=update_memories
            )
            v = v.view(B, S, self.n_kv_heads, self.head_dim)
            new_memory_states['v'] = new_v_mem
        else:
            v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        # Apply RoPE with position offset for KV cache
        total_seq_len = position_offset + S
        cos, sin = self.rotary(q, total_seq_len)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        q_rope = q.view(B * self.n_heads, S, self.head_dim)
        k_rope = k.view(B * self.n_kv_heads, S, self.head_dim)

        # Use position offset for RoPE
        cos = cos[position_offset:position_offset + S].unsqueeze(0)
        sin = sin[position_offset:position_offset + S].unsqueeze(0)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)

        q = q_rope.view(B, self.n_heads, S, self.head_dim)
        k = k_rope.view(B, self.n_kv_heads, S, self.head_dim)

        # KV cache handling
        new_kv_cache = None
        if kv_cache is not None or inference_mode:
            # Concat with cache if exists
            if kv_cache is not None and 'k' in kv_cache:
                k = torch.cat([kv_cache['k'], k], dim=2)
                v = torch.cat([kv_cache['v'], v], dim=2)
            # Store new cache
            new_kv_cache = {'k': k, 'v': v}

        # Get sequence length after potential cache concat
        kv_seq_len = k.shape[2]

        if self.n_rep > 1:
            k_expanded = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, kv_seq_len, self.head_dim)
            v_expanded = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, kv_seq_len, self.head_dim)
        else:
            k_expanded = k
            v_expanded = v

        if self.config.use_flash_attention and FLASH_ATTN_AVAILABLE and kv_cache is None:
            # Flash attention only for non-cached path (full sequence)
            if cu_seqlens is not None and max_seqlen is not None:
                q_fa = q.transpose(1, 2).reshape(B * S, self.n_heads, self.head_dim)
                k_fa = k_expanded.transpose(1, 2).reshape(B * kv_seq_len, self.n_heads, self.head_dim)
                v_fa = v_expanded.transpose(1, 2).reshape(B * kv_seq_len, self.n_heads, self.head_dim)

                attn_out = flash_attn_varlen_func(
                    q_fa, k_fa, v_fa,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                )
                attn_out = attn_out.reshape(B, S, -1)
            else:
                q_fa = q.transpose(1, 2)
                k_fa = k_expanded.transpose(1, 2)
                v_fa = v_expanded.transpose(1, 2)
                attn_out = flash_attn_func(q_fa, k_fa, v_fa, causal=True)
                attn_out = attn_out.reshape(B, S, -1)
        else:
            # Standard attention (supports KV cache)
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale

            # Causal mask - only mask future tokens relative to query position
            if kv_cache is not None:
                # For cached inference: query attends to all cached + current
                # No masking needed for single token generation
                pass
            else:
                # Full sequence: standard causal mask
                mask = torch.triu(torch.ones(S, kv_seq_len, device=x.device, dtype=torch.bool), diagonal=1)
                attn = attn.masked_fill(mask, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.config.attention_dropout, training=self.training)
            attn_out = torch.matmul(attn, v_expanded)
            attn_out = attn_out.transpose(1, 2).reshape(B, S, -1)

        output = self.o_proj(attn_out)
        return output, new_memory_states, new_kv_cache


class StandardAttention(nn.Module):
    """Standard GQA attention (no Hope memory)"""

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
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        cos, sin = self.rotary(q, S)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q_rope = q.view(B * self.n_heads, S, self.head_dim)
        k_rope = k.view(B * self.n_kv_heads, S, self.head_dim)

        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)

        q = q_rope.view(B, self.n_heads, S, self.head_dim)
        k = k_rope.view(B, self.n_kv_heads, S, self.head_dim)

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, S, self.head_dim)

        if self.config.use_flash_attention and FLASH_ATTN_AVAILABLE:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_out = flash_attn_func(q, k, v, causal=True)
            attn_out = attn_out.reshape(B, S, -1)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.config.attention_dropout, training=self.training)
            attn_out = torch.matmul(attn, v)
            attn_out = attn_out.transpose(1, 2).reshape(B, S, -1)

        return self.o_proj(attn_out), {}


# =============================================================================
# Transformer Block
# =============================================================================

class HopeBlock(nn.Module):
    """Transformer block with Hope Attention"""

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

        if config.use_cms_ffn:
            self.ffn = CMSFFN(config)
        else:
            self.ffn = SwiGLUFFN(config.d_model, config.d_ff, config.dropout)

    def init_memory_states(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        if hasattr(self.attention, 'init_memory_states'):
            return {'attention': self.attention.init_memory_states(batch_size, device, dtype)}
        return {}

    def get_memory_reg_loss(self, memory_states: Dict[str, Any]) -> Tensor:
        if hasattr(self.attention, 'get_memory_reg_loss') and 'attention' in memory_states:
            return self.attention.get_memory_reg_loss(memory_states['attention'])
        return torch.tensor(0.0)

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[Dict[str, Any]] = None,
        update_memories: bool = True,
        inference_mode: bool = False,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        kv_cache: Optional[Dict[str, Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[Tensor, Dict[str, Any], Optional[Dict[str, Tensor]]]:
        attn_mem = memory_states.get('attention', {}) if memory_states else {}
        layer_kv_cache = kv_cache.get('kv', None) if kv_cache else None

        attn_out, new_attn_mem, new_kv = self.attention(
            self.norm1(x),
            memory_states=attn_mem,
            update_memories=update_memories,
            inference_mode=inference_mode,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kv_cache=layer_kv_cache,
            position_offset=position_offset,
        )
        x = x + attn_out

        x = x + self.ffn(self.norm2(x))

        new_states = {'attention': new_attn_mem} if new_attn_mem else {}
        new_cache = {'kv': new_kv} if new_kv else None
        return x, new_states, new_cache


# =============================================================================
# Full Model
# =============================================================================

class Onyx(nn.Module):
    """Onyx Language Model with Persistent Memory"""

    def __init__(self, config: OnyxConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            HopeBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self._init_weights()
        self.gradient_checkpointing = config.gradient_checkpointing
        self._memory_update_count = 0

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.tie_embeddings:
            n_params -= self.embed.weight.numel()
        return n_params

    def init_memory_states(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> List[Dict[str, Any]]:
        return [layer.init_memory_states(batch_size, device, dtype) for layer in self.layers]

    def detach_memory_states(self, memory_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        detached = []
        for layer_mem in memory_states:
            layer_detached = {}
            for key, val in layer_mem.items():
                if isinstance(val, dict):
                    layer_detached[key] = {k: v.detach() if torch.is_tensor(v) else v for k, v in val.items()}
                elif torch.is_tensor(val):
                    layer_detached[key] = val.detach()
                else:
                    layer_detached[key] = val
            detached.append(layer_detached)
        return detached

    def get_memory_reg_loss(self, memory_states: List[Dict[str, Any]]) -> Tensor:
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer, layer_mem in zip(self.layers, memory_states):
            total_loss = total_loss + layer.get_memory_reg_loss(layer_mem)
        return total_loss * self.config.memory_reg_weight

    def save_memory_states(self, memory_states: List[Dict[str, Any]], path: str):
        cpu_states = []
        for layer_mem in memory_states:
            layer_cpu = {}
            for key, val in layer_mem.items():
                if isinstance(val, dict):
                    layer_cpu[key] = {k: v.cpu() if torch.is_tensor(v) else v for k, v in val.items()}
                elif torch.is_tensor(val):
                    layer_cpu[key] = val.cpu()
                else:
                    layer_cpu[key] = val
            cpu_states.append(layer_cpu)

        torch.save({
            'memory_states': cpu_states,
            'update_count': self._memory_update_count,
        }, path)

    def load_memory_states(self, path: str, device: torch.device, dtype: torch.dtype = None) -> List[Dict[str, Any]]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self._memory_update_count = checkpoint.get('update_count', 0)

        loaded_states = []
        for layer_mem in checkpoint['memory_states']:
            layer_loaded = {}
            for key, val in layer_mem.items():
                if isinstance(val, dict):
                    layer_loaded[key] = {
                        k: v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v
                        for k, v in val.items()
                    }
                elif torch.is_tensor(val):
                    layer_loaded[key] = val.to(device=device, dtype=dtype)
                else:
                    layer_loaded[key] = val
            loaded_states.append(layer_loaded)

        return loaded_states

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
        kv_cache: Optional[List[Dict[str, Tensor]]] = None,
        position_offset: int = 0,
    ) -> Dict[str, Any]:
        B, S = input_ids.shape

        x = self.embed(input_ids)
        x = self.embed_dropout(x)

        if memory_states is None:
            memory_states = [None] * len(self.layers)

        if kv_cache is None:
            kv_cache = [None] * len(self.layers)

        new_memory_states = []
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x, new_mem, new_kv = torch.utils.checkpoint.checkpoint(
                    layer, x, memory_states[i], update_memories, inference_mode,
                    cu_seqlens, max_seqlen, kv_cache[i], position_offset,
                    use_reentrant=False
                )
            else:
                x, new_mem, new_kv = layer(x, memory_states[i], update_memories, inference_mode,
                                           cu_seqlens, max_seqlen, kv_cache[i], position_offset)
            new_memory_states.append(new_mem)
            new_kv_cache.append(new_kv)

        if update_memories:
            self._memory_update_count += S

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        result = {
            "logits": logits,
            "loss": loss,
            "memory_states": new_memory_states,
            "kv_cache": new_kv_cache,
        }

        if return_memory_reg_loss:
            result["memory_reg_loss"] = self.get_memory_reg_loss(new_memory_states)

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        memory_states: Optional[List[Dict[str, Any]]] = None,
        memory_mode: str = "session",
        memory_path: Optional[str] = None,
        update_memory: bool = True,
        eos_token_id: Optional[int] = None,
        use_kv_cache: bool = True,
    ) -> Tuple[Tensor, List[Dict[str, Any]]]:
        self.eval()
        B, S = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        if memory_mode == "stateless":
            memory_states = self.init_memory_states(B, device, dtype)
            update_memory = False
        elif memory_mode == "session":
            if memory_states is None:
                memory_states = self.init_memory_states(B, device, dtype)
        elif memory_mode == "persistent":
            if memory_path and Path(memory_path).exists():
                memory_states = self.load_memory_states(memory_path, device, dtype)
            elif memory_states is None:
                memory_states = self.init_memory_states(B, device, dtype)

        # Process prompt
        outputs = self(
            input_ids,
            memory_states=memory_states,
            update_memories=update_memory,
            inference_mode=True,
            position_offset=0,
        )
        memory_states = outputs["memory_states"]
        kv_cache = outputs["kv_cache"] if use_kv_cache else None
        position_offset = S

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = outputs["logits"][:, -1, :]

            # [FIXED] Deterministic generation for temp=0
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # [FIXED] Strict EOS check
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            outputs = self(
                next_token,
                memory_states=memory_states,
                update_memories=update_memory,
                inference_mode=True,
                kv_cache=kv_cache,
                position_offset=position_offset,
            )
            memory_states = outputs["memory_states"]
            kv_cache = outputs["kv_cache"] if use_kv_cache else None
            position_offset += 1

        if memory_mode == "persistent" and memory_path and update_memory:
            self.save_memory_states(memory_states, memory_path)

        return generated, memory_states


# =============================================================================
# M3 Optimizer
# =============================================================================

class M3Optimizer(torch.optim.Optimizer):
    """Multi-scale Momentum optimizer (M3)"""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        beta_slow: float = 0.99,
        slow_freq: int = 50,
        slow_weight: float = 0.1,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            beta_slow=beta_slow, slow_freq=slow_freq, slow_weight=slow_weight
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta_slow = group['beta_slow']
            slow_freq = group['slow_freq']
            slow_weight = group['slow_weight']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("M3 does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['slow_avg'] = torch.zeros_like(p)

                state['step'] += 1

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                slow_avg = state['slow_avg']

                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                if state['step'] % slow_freq == 0:
                    slow_avg.mul_(beta_slow).add_(exp_avg, alpha=1 - beta_slow)

                combined = exp_avg + slow_weight * slow_avg
                p.data.addcdiv_(combined, denom, value=-step_size)

        return loss


# =============================================================================
# Factory Functions
# =============================================================================

def create_onyx(**kwargs) -> Onyx:
    """Create Onyx 11M model with optional config overrides."""
    config = OnyxConfig(**kwargs)
    return Onyx(config)


def get_param_groups(model: Onyx, weight_decay: float = 0.1, memory_lr_scale: float = 0.1):
    """Get parameter groups for optimizer"""
    decay_params = []
    no_decay_params = []
    memory_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'eta_raw' in name or 'alpha_raw' in name:
            memory_params.append(param)
        elif 'bias' in name or 'norm' in name or 'embed' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": memory_params, "weight_decay": 0.0, "lr_scale": memory_lr_scale},
    ]


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Onyx 11M Model Test")
    print("=" * 70)

    # Create model with smaller vocab for testing
    model = create_onyx(vocab_size=1000)
    config = model.config

    print(f"\nConfiguration:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_kv_heads: {config.n_kv_heads}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  Memory chunk size: {config.memory_chunk_size}")

    total_params = model.get_num_params()
    print(f"\nParameters: {total_params:,} ({total_params/1e6:.1f}M)")

    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 64

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    memory_states = model.init_memory_states(batch_size, input_ids.device, torch.float32)

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        memory_states=memory_states,
        update_memories=True,
        return_memory_reg_loss=True,
    )

    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Memory reg loss: {outputs['memory_reg_loss'].item():.6f}")
    print(f"  Memory states: {len(outputs['memory_states'])} layers")

    # Test batch size mismatch handling (the bug fix)
    print("\nTesting batch size change (bug fix validation)...")
    new_batch_size = 4
    new_input = torch.randint(0, config.vocab_size, (new_batch_size, seq_len))
    outputs2 = model(
        input_ids=new_input,
        memory_states=outputs['memory_states'],  # Different batch size - should auto-resize
        update_memories=True,
    )
    print(f"  New batch handled correctly: {outputs2['logits'].shape}")

    print("\nTesting memory detachment...")
    detached = model.detach_memory_states(outputs['memory_states'])
    print(f"  Detached states: {len(detached)} layers")

    print("\nTesting backward pass...")
    total_loss = outputs['loss'] + outputs['memory_reg_loss']
    total_loss.backward()

    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  Gradient norm: {grad_norm:.4f}")

    print("\nTesting generation...")
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 10))

    generated, final_memory = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        memory_mode="session",
    )
    print(f"  Generated shape: {generated.shape}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
