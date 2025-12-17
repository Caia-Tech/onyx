#!/usr/bin/env python3
"""
Onyx Inference Script

Examples:
  python onyx_inference.py --checkpoint /path/to/checkpoint.pt
  python onyx_inference.py --checkpoint /path/to/checkpoint.pt --memory persistent --memory_path ./mem.pt --learning
  python onyx_inference.py --checkpoint /path/to/checkpoint.pt --prompt "Hello"
"""

import argparse
import time
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


# =============================================================================
# Loading
# =============================================================================

def _flatten_cfg(cfg_json: dict) -> dict:
    flat = {}
    for k, v in cfg_json.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def load_model(
    checkpoint_path: str,
    tokenizer=None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    model_config_path: Optional[str] = None,
):
    """
    Loads model + config, and stabilizes vocab sizing:
      vocab_size := max(config_vocab, checkpoint_vocab, len(tokenizer or []))
    Then pads/truncates embed/lm_head weights accordingly.
    """
    import json
    import dataclasses

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config: Optional[OnyxConfig] = None

    # 1) optional explicit model_config
    if model_config_path and Path(model_config_path).exists():
        print(f"Loading model config from: {model_config_path}")
        cfg_json = json.loads(Path(model_config_path).read_text())
        flat = _flatten_cfg(cfg_json)
        valid = {f.name for f in dataclasses.fields(OnyxConfig)}
        filtered = {k: v for k, v in flat.items() if k in valid}
        config = OnyxConfig(**filtered)

    # 2) checkpoint referenced model_config_path
    if config is None and isinstance(ckpt, dict) and "config" in ckpt:
        cfg = ckpt["config"]
        if isinstance(cfg, dict):
            mcp = cfg.get("model_config_path")
            if mcp:
                ckpt_dir = Path(checkpoint_path).parent
                candidates = [ckpt_dir / mcp, ckpt_dir.parent / mcp, Path(mcp)]
                for p in candidates:
                    if p.exists():
                        print(f"Loading model config from checkpoint reference: {p}")
                        cfg_json = json.loads(p.read_text())
                        flat = _flatten_cfg(cfg_json)
                        valid = {f.name for f in dataclasses.fields(OnyxConfig)}
                        filtered = {k: v for k, v in flat.items() if k in valid}
                        config = OnyxConfig(**filtered)
                        break

    # 3) fallback: config stored directly in checkpoint
    if config is None:
        if isinstance(ckpt, dict) and "config" in ckpt:
            cfg = ckpt["config"]
            if isinstance(cfg, dict):
                valid = {f.name for f in dataclasses.fields(OnyxConfig)}
                filtered = {k: v for k, v in cfg.items() if k in valid}
                config = OnyxConfig(**filtered) if filtered else OnyxConfig()
            else:
                config = cfg
        else:
            config = OnyxConfig()

    # determine state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    # stabilize vocab (never shrink below config vocab)
    tok_vocab = int(len(tokenizer)) if tokenizer is not None else 0
    ckpt_vocab = int(state["embed.weight"].shape[0]) if isinstance(state, dict) and "embed.weight" in state else int(config.vocab_size)
    target_vocab = max(int(config.vocab_size), tok_vocab, ckpt_vocab)

    if config.vocab_size != target_vocab:
        print(f"Setting config.vocab_size: {config.vocab_size} -> {target_vocab} (max(tokenizer, checkpoint))")
        config.vocab_size = target_vocab

    print(f"Model config: d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}, vocab={config.vocab_size}")
    model = Onyx(config)

    # resize embed/lm_head weights in the loaded state dict to match config.vocab_size
    def _pad_or_trunc(key: str):
        if key not in state:
            return
        w = state[key]
        if w.ndim != 2:
            return
        cur = w.shape[0]
        if cur == config.vocab_size:
            return
        if cur < config.vocab_size:
            pad_rows = config.vocab_size - cur
            pad = torch.zeros((pad_rows, w.shape[1]), dtype=w.dtype, device=w.device)
            state[key] = torch.cat([w, pad], dim=0)
            print(f"  - Padded {key}: {cur} -> {config.vocab_size}")
        else:
            state[key] = w[: config.vocab_size, :]
            print(f"  - Truncated {key}: {cur} -> {config.vocab_size}")

    if isinstance(state, dict):
        _pad_or_trunc("embed.weight")
        _pad_or_trunc("lm_head.weight")

    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, config


# =============================================================================
# Sampling
# =============================================================================

def sample_token(
    logits: torch.Tensor,  # [1, vocab]
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[object] = None,
) -> torch.Tensor:
    def _iter_penalty_token_ids(src: object):
        if src is None:
            return ()
        if isinstance(src, set):
            return src
        if isinstance(src, (list, tuple)):
            return src
        if torch.is_tensor(src):
            if src.numel() == 0:
                return ()
            t = src.detach()
            # MPS unique can be slow; always compute on CPU.
            if t.device.type != "cpu":
                t = t.to("cpu")
            return [int(x) for x in torch.unique(t)]
        return ()

    if repetition_penalty != 1.0:
        for tid in _iter_penalty_token_ids(generated_tokens):
            tid = int(tid)
            if 0 <= tid < logits.size(-1):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")

    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        top_prob = probs.max(dim=-1, keepdim=True).values
        thresh = top_prob * min_p
        logits[probs < thresh] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = 0
        mask = remove.scatter(1, sorted_idx, remove)
        logits[mask] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# =============================================================================
# Streaming generation (FIXED memory yield)
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
    min_tokens_before_eos: int = 5,
    stop_on_eos: bool = True,
) -> Generator[Tuple[torch.Tensor, List[Dict[str, Any]]], None, None]:
    """
    FIX: yields memory_states AFTER consuming the sampled token.
    This makes session/persistent memory *actually* advance in the caller.
    """
    model.eval()
    B, S = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype

    if memory_states is None:
        memory_states = model.init_memory_states(B, device, dtype)

    stop_tokens = list(stop_tokens or [])
    if stop_on_eos and eos_token_id is not None and eos_token_id not in stop_tokens:
        stop_tokens.append(eos_token_id)

    generated_count = 0
    seen_token_ids: set[int] = set()

    with torch.no_grad():
        outputs = model(
            input_ids,
            memory_states=memory_states,
            update_memories=update_memory,
            inference_mode=True,
            position_offset=0,
        )
        memory_states = outputs["memory_states"]
        kv_cache = outputs.get("kv_cache") if use_kv_cache else None
        position_offset = S

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = outputs["logits"][:, -1, :]  # [1, vocab]

            if eos_token_id is not None and generated_count < min_tokens_before_eos:
                logits[:, eos_token_id] = float("-inf")

            next_token = sample_token(
                logits.clone(),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=(seen_token_ids if repetition_penalty != 1.0 else None),
            )  # [1,1]

            tid = int(next_token.item())
            generated_count += 1
            if repetition_penalty != 1.0:
                seen_token_ids.add(tid)

            # Consume the token to advance memory + kv_cache first
            next_outputs = model(
                next_token,
                memory_states=memory_states,
                update_memories=update_memory,
                inference_mode=True,
                kv_cache=kv_cache,
                position_offset=position_offset,
            )
            memory_states = next_outputs["memory_states"]
            kv_cache = next_outputs.get("kv_cache") if use_kv_cache else None
            position_offset += 1

            # Now yield (token, UPDATED memory)
            yield next_token, memory_states

            outputs = next_outputs

            if tid in stop_tokens:
                break


# =============================================================================
# Chat
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
    print("\n" + "=" * 60)
    print("  Onyx Chat")
    print("=" * 60)
    print(f"  Memory: {memory_mode} | Learning: {'ON' if learning else 'OFF'} | Stream: {'ON' if stream else 'OFF'}")
    print("-" * 60)
    print("  Commands: /quit /exit /clear /save /learning on|off /temp <v> /system <text>")
    print("=" * 60 + "\n")

    memory_states = None
    conversation_history = []

    if memory_mode == "persistent" and memory_path and Path(memory_path).exists():
        memory_states = model.load_memory_states(memory_path, device, dtype)
        print(f"[Loaded memory from {memory_path}]\n")

    eos_token_id = tokenizer.eos_token_id
    stop_tokens = [eos_token_id] if eos_token_id is not None else []
    print(f"[Debug] Stop Tokens: {stop_tokens}")

    while True:
        try:
            user_input = input("\033[92mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd_parts = user_input[1:].split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else None

            if cmd in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if cmd == "clear":
                memory_states = None
                conversation_history = []
                print("[Memory cleared]\n")
                continue
            if cmd == "save":
                if memory_mode == "persistent" and memory_path:
                    if memory_states is not None:
                        model.save_memory_states(memory_states, memory_path)
                        print(f"[Memory saved to {memory_path}]\n")
                    else:
                        print("[No memory to save]\n")
                else:
                    print("[Not in persistent mode / no memory_path]\n")
                continue
            if cmd == "learning" and arg:
                if arg.lower() in ("on", "true", "1"):
                    learning = True
                    print("[Learning: ON]\n")
                elif arg.lower() in ("off", "false", "0"):
                    learning = False
                    print("[Learning: OFF]\n")
                continue
            if cmd == "temp" and arg:
                try:
                    temperature = float(arg)
                    print(f"[Temperature: {temperature}]\n")
                except ValueError:
                    print("[Invalid temperature]\n")
                continue
            if cmd == "system" and arg:
                system_prompt = arg
                print("[System prompt set]\n")
                continue

            print("[Unknown command]\n")
            continue

        # Build prompt
        prompt = ""
        if system_prompt:
            prompt += f"System: {system_prompt}\n\n"
        for turn in conversation_history[-4:]:
            prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
        prompt += f"User: {user_input}\nAssistant:"

        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        if tokenizer.bos_token_id is not None:
            bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
            input_ids = torch.cat([bos, input_ids], dim=1)

        # Semantics:
        # - Memory updates in RAM for session/persistent (unless stateless).
        # - Disk persistence only when persistent+learning (or /save).
        update_memory = (memory_mode != "stateless")

        print("\033[94mOnyx:\033[0m ", end="", flush=True)
        start = time.time()
        generated_text = ""
        token_count = 0

        try:
            if stream:
                for token, memory_states in generate_stream(
                    model=model,
                    input_ids=input_ids,
                    tokenizer=tokenizer,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    memory_states=memory_states,
                    update_memory=update_memory,
                    eos_token_id=eos_token_id,
                    stop_tokens=stop_tokens,
                    min_tokens_before_eos=8,
                    stop_on_eos=True,
                ):
                    tid = int(token.item())
                    if eos_token_id is not None and tid == eos_token_id:
                        continue
                    txt = tokenizer.decode(token[0], skip_special_tokens=True)
                    print(txt, end="", flush=True)
                    generated_text += txt
                    token_count += 1
            else:
                # Non-stream: do a quick greedy-style loop via generate_stream accumulation
                for token, memory_states in generate_stream(
                    model=model,
                    input_ids=input_ids,
                    tokenizer=tokenizer,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    memory_states=memory_states,
                    update_memory=update_memory,
                    eos_token_id=eos_token_id,
                    stop_tokens=stop_tokens,
                    min_tokens_before_eos=8,
                    stop_on_eos=True,
                ):
                    tid = int(token.item())
                    if eos_token_id is not None and tid == eos_token_id:
                        continue
                    txt = tokenizer.decode(token[0], skip_special_tokens=True)
                    generated_text += txt
                    token_count += 1
                print(generated_text, end="", flush=True)
        except KeyboardInterrupt:
            print("\n[Interrupted generation]\n", flush=True)

        elapsed = time.time() - start
        print(f"\n\033[90m[{token_count} tokens, {token_count / max(elapsed, 1e-8):.1f} tok/s]\033[0m\n")

        conversation_history.append({"user": user_input, "assistant": generated_text.strip()})

        # Persist only if (persistent + learning)
        if memory_mode == "persistent" and memory_path and learning and memory_states is not None:
            model.save_memory_states(memory_states, memory_path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Onyx Inference")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Hermes-2-Pro-Llama-3-8B")

    parser.add_argument("--memory", type=str, default="session", choices=["stateless", "session", "persistent"])
    parser.add_argument("--memory_path", type=str, default=None)
    parser.add_argument("--learning", action="store_true", help="Allow saving memory to disk in persistent mode")

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_tokens", type=int, default=512)

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--stream", action="store_true", help="Stream tokens (interactive default)")
    g.add_argument("--no_stream", action="store_true", help="Disable streaming")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--system", type=str, default=None)

    args = parser.parse_args()

    stream = True if (not args.no_stream) else False
    if args.stream:
        stream = True

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    print(f"Device: {device} | Dtype: {dtype}")

    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed. Run: pip install transformers")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, _ = load_model(args.checkpoint, tokenizer, device, dtype, args.model_config)
    print(f"Model parameters: {model.get_num_params():,}")

    if args.prompt is not None:
        prompt = args.prompt
        if args.system:
            prompt = f"System: {args.system}\n\nUser: {prompt}\nAssistant:"

        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        if tokenizer.bos_token_id is not None:
            bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
            input_ids = torch.cat([bos, input_ids], dim=1)

        eos = tokenizer.eos_token_id
        stop_tokens = [eos] if eos is not None else []

        # For single prompt, update memory in RAM if memory != stateless
        update_memory = (args.memory != "stateless")

        out_text = ""
        if stream:
            for token, _mem in generate_stream(
                model=model,
                input_ids=input_ids,
                tokenizer=tokenizer,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
                memory_states=None,
                update_memory=update_memory,
                eos_token_id=eos,
                stop_tokens=stop_tokens,
                min_tokens_before_eos=8,
                stop_on_eos=True,
            ):
                tid = int(token.item())
                if eos is not None and tid == eos:
                    continue
                chunk = tokenizer.decode(token[0], skip_special_tokens=True)
                print(chunk, end="", flush=True)
                out_text += chunk
            print()
        else:
            # non-stream path: just collect
            for token, _mem in generate_stream(
                model=model,
                input_ids=input_ids,
                tokenizer=tokenizer,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
                memory_states=None,
                update_memory=update_memory,
                eos_token_id=eos,
                stop_tokens=stop_tokens,
                min_tokens_before_eos=8,
                stop_on_eos=True,
            ):
                tid = int(token.item())
                if eos is not None and tid == eos:
                    continue
                out_text += tokenizer.decode(token[0], skip_special_tokens=True)
            print(out_text)
    else:
        chat(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            memory_mode=args.memory,
            memory_path=args.memory_path,
            learning=args.learning,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens,
            stream=stream,
            system_prompt=args.system,
        )


if __name__ == "__main__":
    main()
