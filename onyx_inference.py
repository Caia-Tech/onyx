#!/usr/bin/env python3
"""
Onyx Inference Script

Handles model loading, sampling, and streaming generation with 
Hope/Titans memory management (Session vs Persistent).
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator, Tuple

import torch
import torch.nn.functional as F
import dataclasses

# Try importing Transformers for Tokenizer
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import your model
from onyx_model import Onyx, OnyxConfig


# =============================================================================
# Helper: Flatten Config
# =============================================================================

def _flatten_cfg(cfg_json: dict) -> dict:
    """Flattens nested config dictionaries (e.g. {'architecture': {...}})"""
    flat = {}
    for k, v in cfg_json.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    checkpoint_path: str,
    tokenizer=None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    model_config_path: Optional[str] = None,
):
    """
    Loads model weights and config. 
    Critically, it stabilizes 'vocab_size' to ensure weights fit, 
    handling mismatches between config, checkpoint, and tokenizer.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config: Optional[OnyxConfig] = None
    valid_fields = {f.name for f in dataclasses.fields(OnyxConfig)}

    # 1. Try loading explicit config file
    if model_config_path and Path(model_config_path).exists():
        print(f"Loading config from file: {model_config_path}")
        cfg_json = json.loads(Path(model_config_path).read_text())
        flat = _flatten_cfg(cfg_json)
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        config = OnyxConfig(**filtered)

    # 2. Try loading config from checkpoint dict
    if config is None and isinstance(ckpt, dict) and "config" in ckpt:
        cfg_data = ckpt["config"]
        
        # A. Handle indirect reference (e.g. {"model_config_path": "model.json"})
        if isinstance(cfg_data, dict) and "model_config_path" in cfg_data:
            ref_path = cfg_data["model_config_path"]
            # Try to find it relative to the checkpoint
            ckpt_dir = Path(checkpoint_path).parent
            candidates = [Path(ref_path), ckpt_dir / ref_path]
            for p in candidates:
                if p.exists():
                    print(f"Loading referenced config: {p}")
                    cfg_json = json.loads(p.read_text())
                    flat = _flatten_cfg(cfg_json)
                    filtered = {k: v for k, v in flat.items() if k in valid_fields}
                    config = OnyxConfig(**filtered)
                    break

        # B. Handle direct dict (Standard case)
        if config is None and isinstance(cfg_data, dict):
            filtered = {k: v for k, v in cfg_data.items() if k in valid_fields}
            # Only use if we have enough fields to look like a real config
            if len(filtered) > 0: 
                config = OnyxConfig(**filtered)
                
        # C. Handle OnyxConfig object
        elif config is None and isinstance(cfg_data, OnyxConfig):
            config = cfg_data

    # 3. Fallback to default
    if config is None:
        print("[Warn] No config found in checkpoint or args, using defaults.")
        config = OnyxConfig()

    # --- Vocabulary Stabilization ---
    # We must ensure the model structure matches the weights we are about to load.
    
    # Get state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    # Determine required vocab size
    tok_vocab = int(len(tokenizer)) if tokenizer is not None else 0
    
    # Check what size the checkpoint actually has
    ckpt_vocab = 0
    if isinstance(state, dict) and "embed.weight" in state:
        ckpt_vocab = int(state["embed.weight"].shape[0])
    elif int(config.vocab_size) > 0:
        ckpt_vocab = int(config.vocab_size)

    # The model must be large enough to handle ALL potential tokens
    target_vocab = max(int(config.vocab_size), tok_vocab, ckpt_vocab)

    if config.vocab_size != target_vocab:
        print(f"Resizing config.vocab_size: {config.vocab_size} -> {target_vocab}")
        config.vocab_size = target_vocab

    # Initialize Model
    model = Onyx(config)

    # Resize checkpoint weights if they are smaller than target
    def _pad_or_trunc(key: str):
        if key not in state: return
        w = state[key]
        if w.ndim != 2: return
        cur_rows = w.shape[0]
        
        if cur_rows == target_vocab:
            return
        
        if cur_rows < target_vocab:
            print(f"Padding {key}: {cur_rows} -> {target_vocab}")
            pad_rows = target_vocab - cur_rows
            pad = torch.zeros((pad_rows, w.shape[1]), dtype=w.dtype, device=w.device)
            state[key] = torch.cat([w, pad], dim=0)
        else:
            # Usually we don't truncate unless we are forcing a smaller vocab
            print(f"Truncating {key}: {cur_rows} -> {target_vocab}")
            state[key] = w[:target_vocab, :]

    if isinstance(state, dict):
        _pad_or_trunc("embed.weight")
        _pad_or_trunc("lm_head.weight")

    # Load weights
    keys = model.load_state_dict(state, strict=False)
    if keys.missing_keys:
        # Filter out ignored keys (like dynamic hyperparams we might have just added)
        real_missing = [k for k in keys.missing_keys if not ("eta_" in k or "alpha_" in k)]
        if real_missing:
            print(f"[Warn] Missing keys: {real_missing[:5]}...")
        
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    return model, config


# =============================================================================
# Sampling Logic
# =============================================================================

def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[object] = None,
) -> torch.Tensor:
    """Standard sampling with temp, top-k, top-p, and repetition penalty."""
    
    # Helper to handle different generated_tokens formats
    def _iter_token_ids(src):
        if src is None: return ()
        if isinstance(src, set): return src
        if isinstance(src, list): return src
        if torch.is_tensor(src): return src.tolist()
        return ()

    # Repetition Penalty
    if repetition_penalty != 1.0 and generated_tokens is not None:
        for tid in _iter_token_ids(generated_tokens):
            tid = int(tid)
            if tid < logits.size(-1):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

    # Temperature
    if temperature <= 1e-5:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature

    # Top-K
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")

    # Min-P (Alternative to Top-P)
    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        top_prob = probs.max(dim=-1, keepdim=True).values
        mask = probs < (top_prob * min_p)
        logits[mask] = float("-inf")

    # Top-P (Nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# =============================================================================
# Generator (Streaming)
# =============================================================================

def generate_stream(
    model: Onyx,
    input_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.0,
    repetition_penalty: float = 1.1,
    memory_states: Optional[List[Dict[str, Any]]] = None,
    update_memory: bool = True,
    eos_token_id: Optional[int] = None,
    stop_tokens: Optional[List[int]] = None,
    use_kv_cache: bool = True,
    min_tokens_before_eos: int = 2,
    stop_on_eos: bool = True,
) -> Generator[Tuple[torch.Tensor, List[Dict[str, Any]]], None, None]:
    """
    Yields (token, memory_states) step-by-step.
    
    Nested Learning Detail:
    This function processes the new token and immediately returns the 
    UPDATED memory state, reflecting the 'Session' learning.
    """
    model.eval()
    B, S = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype

    if memory_states is None:
        memory_states = model.init_memory_states(B, device, dtype)

    # Setup Stop Tokens
    stop_tokens = list(stop_tokens or [])
    if stop_on_eos and eos_token_id is not None and eos_token_id not in stop_tokens:
        stop_tokens.append(eos_token_id)

    generated_count = 0
    seen_token_ids = set()

    # 1. Prefill
    with torch.no_grad():
        outputs = model(
            input_ids,
            memory_states=memory_states,
            update_memories=update_memory,
            inference_mode=True,
            position_offset=0
        )
        memory_states = outputs["memory_states"]
        kv_cache = outputs.get("kv_cache") if use_kv_cache else None
        position_offset = S

    # 2. Generation Loop
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = outputs["logits"][:, -1, :]

            # Constraint: Don't EOS too early
            if eos_token_id is not None and generated_count < min_tokens_before_eos:
                logits[:, eos_token_id] = float("-inf")

            next_token = sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=seen_token_ids if repetition_penalty != 1.0 else None
            )

            tid = int(next_token.item())
            if repetition_penalty != 1.0:
                seen_token_ids.add(tid)
            
            generated_count += 1

            # 3. Step Forward (Update Memory & Cache)
            # This is where the model 'learns' the newly generated token in-context
            next_outputs = model(
                next_token,
                memory_states=memory_states,
                update_memories=update_memory,
                inference_mode=True,
                kv_cache=kv_cache,
                position_offset=position_offset
            )
            
            # Capture updated state
            memory_states = next_outputs["memory_states"]
            kv_cache = next_outputs.get("kv_cache") if use_kv_cache else None
            position_offset += 1
            
            # Yield token and the NEW memory state
            yield next_token, memory_states
            
            outputs = next_outputs

            if tid in stop_tokens:
                break


# =============================================================================
# Chat Interface
# =============================================================================

def chat(
    model: Onyx,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    memory_mode: str = "session",
    memory_path: Optional[str] = None,
    learning: bool = False,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.0,
    repetition_penalty: float = 1.1,
    max_tokens: int = 512,
    stream: bool = True,
    system_prompt: Optional[str] = None,
):
    print(f"\n=== Onyx Chat (Memory: {memory_mode}) ===")
    print("Commands: /save, /clear, /exit")
    
    memory_states = None
    
    # Load persistent memory if available
    if memory_mode == "persistent" and memory_path and Path(memory_path).exists():
        try:
            memory_states = model.load_memory_states(memory_path, device, dtype)
            print(f"[System] Loaded persistent memory from {memory_path}")
        except Exception as e:
            print(f"[Error] Could not load memory: {e}")

    # Ensure stop tokens
    eos = tokenizer.eos_token_id
    stop_tokens = [eos] if eos is not None else []
    
    conversation_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input: continue

        # Handle Commands
        if user_input.lower() in ("/exit", "/quit"):
            break
        if user_input.lower() == "/clear":
            memory_states = None
            conversation_history = []
            print("[System] Memory cleared.")
            continue
        if user_input.lower() == "/save":
            if memory_path and memory_states:
                model.save_memory_states(memory_states, memory_path)
                print(f"[System] Memory saved to {memory_path}")
            else:
                print("[System] No memory path or empty memory.")
            continue

        # Build Prompt (Simple Chat Format)
        prompt = ""
        if system_prompt:
            prompt += f"System: {system_prompt}\n\n"
        
        # Add recent history context (only visual, memory handles the actual state)
        # Note: If memory_mode != stateless, we rely on internal memory, 
        # so we don't strictly need to replay history in input_ids, 
        # but for robustness we often keep a small sliding window.
        prompt += f"User: {user_input}\nAssistant:"
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Decide if we update memory
        update_memory = (memory_mode != "stateless")
        
        print("Onyx: ", end="", flush=True)
        generated_text = ""
        
        gen_gen = generate_stream(
            model=model,
            input_ids=input_ids,
            tokenizer=tokenizer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            memory_states=memory_states, # Pass previous state
            update_memory=update_memory,
            eos_token_id=eos,
            stop_tokens=stop_tokens
        )
        
        for token, new_mem_state in gen_gen:
            tid = int(token.item())
            if tid == eos: break
            
            word = tokenizer.decode([tid], skip_special_tokens=True)
            print(word, end="", flush=True)
            generated_text += word
            
            # Update loop state
            memory_states = new_mem_state
            
        print("") # Newline
        conversation_history.append(f"User: {user_input}\nAssistant: {generated_text}")
        
        # Auto-save if learning is enabled
        if learning and memory_mode == "persistent" and memory_path:
            model.save_memory_states(memory_states, memory_path)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Hermes-2-Pro-Llama-3-8B")
    
    parser.add_argument("--memory", type=str, default="session", choices=["stateless", "session", "persistent"])
    parser.add_argument("--memory_path", type=str, default=None)
    parser.add_argument("--learning", action="store_true")
    
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    
    # [FIX] Added --stream / --no-stream arguments required by test harness
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stream", action="store_true", default=True, help="Enable streaming (default)")
    group.add_argument("--no_stream", action="store_false", dest="stream", help="Disable streaming")

    args = parser.parse_args()
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    dtype = getattr(torch, args.dtype)
    
    if not TRANSFORMERS_AVAILABLE:
        print("Error: 'transformers' library not found.")
        return

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    # Load Model
    model, _ = load_model(args.checkpoint, tokenizer, device=device, dtype=dtype, model_config_path=args.model_config)
    
    # Run
    if args.prompt:
        # One-off generation
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
        print(f"Prompt: {args.prompt}")
        print("Output:", end=" ")
        
        # Handle streaming config from CLI
        if args.stream:
            for token, _ in generate_stream(model, input_ids, tokenizer, max_new_tokens=args.max_tokens, temperature=args.temperature):
                print(tokenizer.decode(token[0], skip_special_tokens=True), end="", flush=True)
            print()
        else:
            # Accumulate output if no stream
            output_tokens = []
            for token, _ in generate_stream(model, input_ids, tokenizer, max_new_tokens=args.max_tokens, temperature=args.temperature):
                output_tokens.append(token.item())
            print(tokenizer.decode(output_tokens, skip_special_tokens=True))
    else:
        # Chat mode
        chat(
            model, tokenizer, device, dtype,
            memory_mode=args.memory,
            memory_path=args.memory_path,
            learning=args.learning,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=args.stream
        )

if __name__ == "__main__":
    main()
