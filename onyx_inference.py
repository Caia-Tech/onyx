#!/usr/bin/env python3
"""
Onyx Inference Script
Handles model loading, sampling, and streaming generation.
"""

import argparse
import json
import time
import sys
import dataclasses
import re
import io
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator, Tuple
import torch
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from onyx_model import Onyx, OnyxConfig

_TURN_MARKERS = ("\nUser:", "\nAssistant:", "\nSystem:")
_LEADING_ASSISTANT_RE = re.compile(r"^\s*(assistant\s*:+\s*)+", flags=re.IGNORECASE)


def _supports_color() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _configure_utf8_stdout() -> None:
    if not sys.stdout:
        return
    if getattr(sys.stdout, "encoding", "").lower() == "utf-8":
        return
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        return
    if isinstance(sys.stdout, io.TextIOBase):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def _onyx_blue(text: str) -> str:
    if not _supports_color():
        return text
    # Lighter/brighter sky-blue (256-color). Works well on both dark/light themes.
    return f"\033[1;38;5;81m{text}\033[0m"


def _dim(text: str) -> str:
    if not _supports_color():
        return text
    return f"\033[2m{text}\033[0m"


def _strip_leading_assistant(text: str) -> str:
    return _LEADING_ASSISTANT_RE.sub("", text)


def _flatten_cfg(cfg_json: dict) -> dict:
    flat = {}
    for k, v in cfg_json.items():
        if isinstance(v, dict): flat.update(v)
        else: flat[k] = v
    return flat

def sample_token(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9, min_p: float = 0.0, repetition_penalty: float = 1.0, generated_tokens: Optional[object] = None) -> torch.Tensor:
    def _iter_token_ids(src):
        if src is None: return ()
        if isinstance(src, set): return src
        if isinstance(src, list): return src
        if torch.is_tensor(src): return src.tolist()
        return ()
    if repetition_penalty != 1.0 and generated_tokens is not None:
        for tid in _iter_token_ids(generated_tokens):
            tid = int(tid)
            if tid < logits.size(-1):
                if logits[0, tid] > 0: logits[0, tid] /= repetition_penalty
                else: logits[0, tid] *= repetition_penalty
    if temperature <= 1e-5: return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")
    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        top_prob = probs.max(dim=-1, keepdim=True).values
        mask = probs < (top_prob * min_p)
        logits[mask] = float("-inf")
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def load_model(checkpoint_path: str, tokenizer=None, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32, model_config_path: Optional[str] = None):
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config: Optional[OnyxConfig] = None
    valid_fields = {f.name for f in dataclasses.fields(OnyxConfig)}
    if model_config_path and Path(model_config_path).exists():
        print(f"Loading config from file: {model_config_path}")
        cfg_json = json.loads(Path(model_config_path).read_text())
        flat = _flatten_cfg(cfg_json)
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        config = OnyxConfig(**filtered)
    if config is None and isinstance(ckpt, dict) and "config" in ckpt:
        cfg_data = ckpt["config"]
        if isinstance(cfg_data, dict) and cfg_data.get("model_config_path"):
            ref_path = cfg_data["model_config_path"]
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
        if config is None and isinstance(cfg_data, dict):
            filtered = {k: v for k, v in cfg_data.items() if k in valid_fields}
            if len(filtered) > 0: config = OnyxConfig(**filtered)
        elif config is None and isinstance(cfg_data, OnyxConfig):
            config = cfg_data
    if config is None:
        print("[Warn] No config found in checkpoint or args, using defaults.")
        config = OnyxConfig()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt: state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt: state = ckpt["model"]
    else: state = ckpt
    tok_vocab = int(len(tokenizer)) if tokenizer is not None else 0
    ckpt_vocab = 0
    if isinstance(state, dict) and "embed.weight" in state: ckpt_vocab = int(state["embed.weight"].shape[0])
    elif int(config.vocab_size) > 0: ckpt_vocab = int(config.vocab_size)
    target_vocab = max(int(config.vocab_size), tok_vocab, ckpt_vocab)
    if config.vocab_size != target_vocab:
        print(f"Resizing config.vocab_size: {config.vocab_size} -> {target_vocab}")
        config.vocab_size = target_vocab
    model = Onyx(config)
    def _pad_or_trunc(key: str):
        if key not in state: return
        w = state[key]
        if w.ndim != 2: return
        cur_rows = w.shape[0]
        if cur_rows == target_vocab: return
        if cur_rows < target_vocab:
            print(f"Padding {key}: {cur_rows} -> {target_vocab}")
            pad_rows = target_vocab - cur_rows
            pad = torch.zeros((pad_rows, w.shape[1]), dtype=w.dtype, device=w.device)
            state[key] = torch.cat([w, pad], dim=0)
        else:
            print(f"Truncating {key}: {cur_rows} -> {target_vocab}")
            state[key] = w[:target_vocab, :]
    if isinstance(state, dict):
        _pad_or_trunc("embed.weight")
        _pad_or_trunc("lm_head.weight")
    keys = model.load_state_dict(state, strict=False)
    if keys.missing_keys:
        real_missing = [k for k in keys.missing_keys if not ("eta_" in k or "alpha_" in k)]
        if real_missing: print(f"[Warn] Missing keys: {real_missing[:5]}...")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, config

def generate_stream(model: Onyx, input_ids: torch.Tensor, tokenizer, max_new_tokens: int = 512, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9, min_p: float = 0.0, repetition_penalty: float = 1.1, memory_states: Optional[List[Dict[str, Any]]] = None, update_memory: bool = True, eos_token_id: Optional[int] = None, stop_tokens: Optional[List[int]] = None, use_kv_cache: bool = True, min_tokens_before_eos: int = 2, stop_on_eos: bool = True) -> Generator[Tuple[torch.Tensor, List[Dict[str, Any]]], None, None]:
    model.eval()
    B, S = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype
    if memory_states is None: memory_states = model.init_memory_states(B, device, dtype)
    stop_tokens = list(stop_tokens or [])
    if stop_on_eos and eos_token_id is not None and eos_token_id not in stop_tokens: stop_tokens.append(eos_token_id)
    generated_count = 0
    seen_token_ids = set()
    with torch.no_grad():
        outputs = model(input_ids, memory_states=memory_states, update_memories=update_memory, inference_mode=True, position_offset=0)
        memory_states = outputs["memory_states"]
        kv_cache = outputs.get("kv_cache") if use_kv_cache else None
        position_offset = S
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = outputs["logits"][:, -1, :]
            if eos_token_id is not None and generated_count < min_tokens_before_eos: logits[:, eos_token_id] = float("-inf")
            next_token = sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty, generated_tokens=seen_token_ids if repetition_penalty != 1.0 else None)
            tid = int(next_token.item())
            if repetition_penalty != 1.0: seen_token_ids.add(tid)
            generated_count += 1
            next_outputs = model(next_token, memory_states=memory_states, update_memories=update_memory, inference_mode=True, kv_cache=kv_cache, position_offset=position_offset)
            memory_states = next_outputs["memory_states"]
            kv_cache = next_outputs.get("kv_cache") if use_kv_cache else None
            position_offset += 1
            yield next_token, memory_states
            outputs = next_outputs
            if tid in stop_tokens: break

def _stop_seqs(tokenizer) -> List[List[int]]:
    seqs: List[List[int]] = []
    for m in _TURN_MARKERS:
        ids = tokenizer.encode(m, add_special_tokens=False)
        if ids:
            seqs.append(ids)
    return seqs


def _decode_incremental(tokenizer, token_ids: List[int], printed_len: int) -> tuple[str, int]:
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    text = _strip_leading_assistant(text).lstrip("\n ")
    if len(text) <= printed_len:
        return "", printed_len
    return text[printed_len:], len(text)


def chat(model: Onyx, tokenizer, device: torch.device, dtype: torch.dtype, memory_mode: str = "session", memory_path: Optional[str] = None, learning: bool = False, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9, min_p: float = 0.0, repetition_penalty: float = 1.1, max_tokens: int = 512, stream: bool = True, system_prompt: Optional[str] = None):
    onyx_tag = _onyx_blue("onyx")
    device_str = getattr(device, "type", str(device))
    dtype_str = str(dtype).replace("torch.", "")
    print(f"\n{_dim('═' * 72)}")
    print(f" {onyx_tag}  {_dim('|')}  device: {device_str}  {_dim('|')}  dtype: {dtype_str}  {_dim('|')}  memory: {memory_mode}")
    print(f" {_dim('commands:')} /save  /clear  /exit")
    print(f"{_dim('═' * 72)}")
    memory_states = None
    if memory_mode == "persistent" and memory_path and Path(memory_path).exists():
        try:
            memory_states = model.load_memory_states(memory_path, device, dtype)
            print(f"system> loaded persistent memory from {memory_path}")
        except Exception as e:
            print(f"error> could not load memory: {e}")
    eos = tokenizer.eos_token_id
    stop_tokens = [eos] if eos is not None else []
    conversation_history: List[str] = []
    max_history_turns = 8
    stop_seqs = _stop_seqs(tokenizer)
    max_stop_len = max((len(s) for s in stop_seqs), default=0)
    while True:
        try:
            user_input = input(f"\n{_dim('you:')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input: continue
        if user_input.lower() in ("/exit", "/quit"): break
        if user_input.lower() == "/clear":
            memory_states = None
            conversation_history = []
            print("system> memory cleared")
            continue
        if user_input.lower() == "/save":
            if memory_path and memory_states:
                model.save_memory_states(memory_states, memory_path)
                print(f"system> memory saved to {memory_path}")
            else:
                print("system> no memory path or empty memory")
            continue
        prompt = ""
        if system_prompt:
            prompt += f"System: {system_prompt}\n\n"
        if conversation_history:
            prompt += "\n\n".join(conversation_history[-max_history_turns:]).rstrip() + "\n\n"
        prompt += f"User: {user_input}\nAssistant: "
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        update_memory = (memory_mode != "stateless")
        print(f"{onyx_tag}{_dim(':')} ", end="", flush=True)
        generated_text = ""
        out_token_ids: List[int] = []
        printed_len = 0
        gen_gen = generate_stream(model=model, input_ids=input_ids, tokenizer=tokenizer, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty, memory_states=memory_states, update_memory=update_memory, eos_token_id=eos, stop_tokens=stop_tokens)
        for token, new_mem_state in gen_gen:
            tid = int(token.item())
            if tid == eos: break
            out_token_ids.append(tid)

            # Stop if the model starts emitting a new turn marker; drop the marker from output.
            stop_now = False
            if max_stop_len and len(out_token_ids) >= 1:
                for seq in stop_seqs:
                    if len(seq) <= len(out_token_ids) and out_token_ids[-len(seq) :] == seq:
                        del out_token_ids[-len(seq) :]
                        stop_now = True
                        break

            delta, printed_len = _decode_incremental(tokenizer, out_token_ids, printed_len)
            if delta:
                print(delta, end="", flush=True)
                generated_text += delta

            memory_states = new_mem_state
            if stop_now:
                break
        print("")
        conversation_history.append(f"User: {user_input}\nAssistant: {generated_text}")
        if learning and memory_mode == "persistent" and memory_path:
            model.save_memory_states(memory_states, memory_path)

def main():
    _configure_utf8_stdout()
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stream", action="store_true", default=True, help="Enable streaming (default)")
    group.add_argument("--no_stream", action="store_false", dest="stream", help="Disable streaming")
    args = parser.parse_args()
    if args.device: device = torch.device(args.device)
    elif torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    dtype = getattr(torch, args.dtype)
    if not TRANSFORMERS_AVAILABLE:
        print("Error: 'transformers' library not found.")
        return
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model, _ = load_model(args.checkpoint, tokenizer, device=device, dtype=dtype, model_config_path=args.model_config)
    if args.prompt:
        onyx_tag = _onyx_blue("onyx")
        prompt = args.prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        print(f"{_dim('prompt:')} {prompt}")
        print(f"{onyx_tag}{_dim(':')} ", end="", flush=True)

        eos = tokenizer.eos_token_id
        stop_tokens = [eos] if eos is not None else []
        stop_seqs = _stop_seqs(tokenizer)
        max_stop_len = max((len(s) for s in stop_seqs), default=0)

        out_token_ids: List[int] = []
        printed_len = 0

        gen = generate_stream(
            model=model,
            input_ids=input_ids,
            tokenizer=tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=eos,
            stop_tokens=stop_tokens,
        )

        if args.stream:
            for token, _ in gen:
                tid = int(token.item())
                if tid == eos:
                    break
                out_token_ids.append(tid)

                stop_now = False
                if max_stop_len:
                    for seq in stop_seqs:
                        if len(seq) <= len(out_token_ids) and out_token_ids[-len(seq) :] == seq:
                            del out_token_ids[-len(seq) :]
                            stop_now = True
                            break

                delta, printed_len = _decode_incremental(tokenizer, out_token_ids, printed_len)
                if delta:
                    print(delta, end="", flush=True)

                if stop_now:
                    break
            print()
        else:
            for token, _ in gen:
                tid = int(token.item())
                if tid == eos:
                    break
                out_token_ids.append(tid)

                if max_stop_len:
                    for seq in stop_seqs:
                        if len(seq) <= len(out_token_ids) and out_token_ids[-len(seq) :] == seq:
                            del out_token_ids[-len(seq) :]
                            break
            text = tokenizer.decode(out_token_ids, skip_special_tokens=True)
            text = _strip_leading_assistant(text).lstrip("\n ")
            print(text)
    else:
        chat(model, tokenizer, device, dtype, memory_mode=args.memory, memory_path=args.memory_path, learning=args.learning, temperature=args.temperature, max_tokens=args.max_tokens, stream=args.stream, system_prompt=args.system)

if __name__ == "__main__":
    main()
