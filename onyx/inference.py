#!/usr/bin/env python3
"""
Onyx Inference Script
Handles model loading, sampling, and streaming generation.

Changes vs your original:
- Stronger anti-loop decoding:
  - windowed repetition penalty
  - frequency penalty
  - hard run-ban (kills "the the the...")
  - optional no-repeat ngram ban
- Do NOT update memory during generation by default (prevents delta-memory lock-in).
  - Only updates memory when: memory_mode == "persistent" AND learning == True
"""

import argparse
import json
import sys
import dataclasses
import re
from pathlib import Path
from contextlib import nullcontext
from typing import Optional, List, Dict, Any, Generator, Tuple
from collections import deque, Counter

import torch
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from onyx.model import Onyx, OnyxConfig

_TURN_MARKERS = ("\nUser:", "\nAssistant:", "\nSystem:")
_LEADING_ASSISTANT_RE = re.compile(r"^\s*(assistant\s*:+\s*)+", flags=re.IGNORECASE)


def _supports_color() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


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
    flat: Dict[str, Any] = {}
    for k, v in cfg_json.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def _get_ngrams(seq: List[int], n: int) -> set[tuple[int, ...]]:
    if n <= 0 or len(seq) < n:
        return set()
    return {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)}


def _ban_repeat_ngrams(
    logits: torch.Tensor,
    recent_tokens: List[int],
    *,
    no_repeat_ngram_size: int = 3,
) -> None:
    """
    If the last (n-1) tokens form a prefix that has occurred before, ban tokens
    that would complete a previously-seen ngram.
    """
    if no_repeat_ngram_size <= 1:
        return
    n = no_repeat_ngram_size
    if len(recent_tokens) < n - 1:
        return

    prefix = tuple(recent_tokens[-(n - 1) :])
    seen = _get_ngrams(recent_tokens, n)

    # Collect all tokens that would recreate an n-gram with this prefix
    banned_next = set()
    for ng in seen:
        if ng[:-1] == prefix:
            banned_next.add(ng[-1])

    if not banned_next:
        return

    V = logits.size(-1)
    for tid in banned_next:
        if 0 <= tid < V:
            logits[0, tid] = float("-inf")


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.05,
    repetition_penalty: float = 1.15,
    generated_tokens: Optional[Any] = None,
    recent_tokens: Optional[List[int]] = None,
    rep_window: int = 128,
    freq_penalty: float = 0.3,
    ban_run_len: int = 4,
    no_repeat_ngram_size: int = 3,
) -> torch.Tensor:
    """
    logits: [1, V]
    """
    assert logits.dim() == 2 and logits.size(0) == 1

    # Back-compat: support generated_tokens alias used by older callsites/tests.
    if recent_tokens is None and generated_tokens is not None:
        if isinstance(generated_tokens, torch.Tensor):
            recent_tokens = generated_tokens.detach().view(-1).tolist()
        elif isinstance(generated_tokens, set):
            recent_tokens = list(generated_tokens)
        elif isinstance(generated_tokens, (list, tuple)):
            recent_tokens = list(generated_tokens)
        else:
            try:
                recent_tokens = list(generated_tokens)  # type: ignore[arg-type]
            except Exception:
                recent_tokens = []

    # ---- Hard run-ban (kills "the the the...") ----
    if recent_tokens and ban_run_len > 0:
        last = recent_tokens[-1]
        run = 1
        for i in range(len(recent_tokens) - 2, -1, -1):
            if recent_tokens[i] == last:
                run += 1
            else:
                break
        if run >= ban_run_len and 0 <= last < logits.size(-1):
            logits[0, last] = float("-inf")

    # ---- No-repeat ngram ban ----
    if recent_tokens and no_repeat_ngram_size and no_repeat_ngram_size > 1:
        _ban_repeat_ngrams(logits, recent_tokens, no_repeat_ngram_size=no_repeat_ngram_size)

    # ---- Windowed repetition + frequency penalty ----
    if recent_tokens:
        window = recent_tokens[-rep_window:] if rep_window > 0 else recent_tokens
        counts = Counter(window)
        V = logits.size(-1)

        # classic repetition penalty on recent window (not entire history)
        if repetition_penalty != 1.0:
            for tid in counts.keys():
                if 0 <= tid < V:
                    if logits[0, tid] > 0:
                        logits[0, tid] /= repetition_penalty
                    else:
                        logits[0, tid] *= repetition_penalty

        # frequency penalty: subtract from logits proportional to recent count
        if freq_penalty > 0.0:
            for tid, c in counts.items():
                if 0 <= tid < V:
                    logits[0, tid] -= freq_penalty * float(c)

    # Greedy (after penalties)
    if temperature <= 1e-6:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # temperature scaling
    logits = logits / max(temperature, 1e-8)

    # top-k
    if top_k and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = float("-inf")

    # min-p (relative to current max prob)
    if min_p and min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        top_prob = probs.max(dim=-1, keepdim=True).values
        logits[probs < (top_prob * min_p)] = float("-inf")

    # top-p
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = 0
        remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
        logits[remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def load_model(
    checkpoint_path: str,
    tokenizer=None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    model_config_path: Optional[str] = None,
    experimental_mhc: Optional[bool] = None,
    mhc_n: Optional[int] = None,
    mhc_mode: Optional[str] = None,
    mhc_sinkhorn: Optional[bool] = None,
    mhc_sinkhorn_iters: Optional[int] = None,
):
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
    elif model_config_path:
        print(f"[Warn] model_config not found, ignoring: {model_config_path}")

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
            if len(filtered) > 0:
                config = OnyxConfig(**filtered)
        elif config is None and isinstance(cfg_data, OnyxConfig):
            config = cfg_data

    if config is None:
        print("[Warn] No config found in checkpoint or args, using defaults.")
        config = OnyxConfig()

    train_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if not isinstance(train_cfg, dict):
        train_cfg = {}

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    tok_vocab = int(len(tokenizer)) if tokenizer is not None else 0
    ckpt_vocab = 0
    if isinstance(state, dict) and "embed.weight" in state:
        ckpt_vocab = int(state["embed.weight"].shape[0])
    elif int(config.vocab_size) > 0:
        ckpt_vocab = int(config.vocab_size)

    target_vocab = max(int(config.vocab_size), tok_vocab, ckpt_vocab)
    if config.vocab_size != target_vocab:
        print(f"Resizing config.vocab_size: {config.vocab_size} -> {target_vocab}")
        config.vocab_size = target_vocab

    if experimental_mhc is None:
        experimental_mhc = bool(train_cfg.get("experimental_mhc", False))

    if experimental_mhc:
        from onyx.experimental import OnyxMHC

        mhc_n = int(mhc_n if mhc_n is not None else train_cfg.get("mhc_n", 2))
        mhc_mode = str(mhc_mode if mhc_mode is not None else train_cfg.get("mhc_mode", "mhc"))
        if mhc_sinkhorn is None:
            mhc_sinkhorn = bool(train_cfg.get("mhc_sinkhorn", True))
        mhc_sinkhorn_iters = int(
            mhc_sinkhorn_iters if mhc_sinkhorn_iters is not None else train_cfg.get("mhc_sinkhorn_iters", 10)
        )
        print(
            "[Experimental] Using OnyxMHC mhc_n={n} mhc_mode={mode} sinkhorn={sinkhorn} iters={iters}".format(
                n=mhc_n,
                mode=mhc_mode,
                sinkhorn=mhc_sinkhorn,
                iters=mhc_sinkhorn_iters,
            )
        )
        model = OnyxMHC(
            config,
            mhc_n=mhc_n,
            mhc_mode=mhc_mode,
            mhc_sinkhorn=mhc_sinkhorn,
            mhc_sinkhorn_iters=mhc_sinkhorn_iters,
        )
    else:
        model = Onyx(config)

    def _pad_or_trunc(key: str):
        if key not in state:
            return
        w = state[key]
        if w.ndim != 2:
            return
        cur_rows = w.shape[0]
        if cur_rows == target_vocab:
            return
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
        if real_missing:
            print(f"[Warn] Missing keys: {real_missing[:5]}...")

    # On Apple MPS, casting weights to fp16/bf16 can trigger hard failures in some
    # matmul kernels when mixed with fp32 accumulations. Keep weights in fp32 and
    # use autocast for activations instead.
    requested_dtype = dtype
    autocast_dtype: Optional[torch.dtype] = None
    if device.type == "mps" and requested_dtype in (torch.float16, torch.bfloat16):
        autocast_dtype = requested_dtype
        dtype = torch.float32
        print(
            f"[Warn] MPS requested dtype={requested_dtype}; keeping weights in float32 and using autocast({autocast_dtype})."
        )

    model = model.to(device=device, dtype=dtype)
    model.eval()
    if autocast_dtype is not None:
        setattr(model, "_inference_autocast_dtype", autocast_dtype)
    return model, config


def generate_stream(
    model: Onyx,
    input_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.05,
    repetition_penalty: float = 1.15,
    memory_states: Optional[List[Dict[str, Any]]] = None,
    update_memory: bool = False,
    eos_token_id: Optional[int] = None,
    stop_tokens: Optional[List[int]] = None,
    use_kv_cache: bool = True,
    min_tokens_before_eos: int = 2,
    stop_on_eos: bool = True,
    rep_window: int = 128,
    freq_penalty: float = 0.3,
    ban_run_len: int = 4,
    no_repeat_ngram_size: int = 3,
) -> Generator[Tuple[torch.Tensor, List[Dict[str, Any]]], None, None]:
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
    recent = deque(maxlen=max(64, rep_window * 2))  # keep enough for ngram + window penalties

    autocast_dtype = getattr(model, "_inference_autocast_dtype", None)
    autocast_ctx = (
        torch.autocast(device_type="mps", dtype=autocast_dtype)
        if (autocast_dtype is not None and device.type == "mps")
        else nullcontext()
    )

    with torch.no_grad(), autocast_ctx:
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
        with torch.no_grad(), autocast_ctx:
            logits = outputs["logits"][:, -1, :]

            if eos_token_id is not None and generated_count < min_tokens_before_eos:
                logits[:, eos_token_id] = float("-inf")

            next_token = sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                recent_tokens=list(recent),
                rep_window=rep_window,
                freq_penalty=freq_penalty,
                ban_run_len=ban_run_len,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            tid = int(next_token.item())
            recent.append(tid)
            generated_count += 1

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

            yield next_token, memory_states
            outputs = next_outputs

            if tid in stop_tokens:
                break


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


def chat(
    model: Onyx,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    memory_mode: str = "session",
    memory_path: Optional[str] = None,
    learning: bool = False,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.05,
    repetition_penalty: float = 1.15,
    max_tokens: int = 512,
    stream: bool = True,
    system_prompt: Optional[str] = None,
    rep_window: int = 128,
    freq_penalty: float = 0.3,
    ban_run_len: int = 4,
    no_repeat_ngram_size: int = 3,
    chat_template: str = "onyx",
):
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

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit"):
            break

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
        if chat_template == "onyx":
            if system_prompt:
                prompt += f"System: {system_prompt}\n\n"
            if conversation_history:
                prompt += "\n\n".join(conversation_history[-max_history_turns:]).rstrip() + "\n\n"
            prompt += f"User: {user_input}\nAssistant: "
        elif chat_template == "raw":
            if system_prompt:
                prompt += system_prompt.rstrip() + "\n\n"
            if conversation_history:
                prompt += "\n\n".join(conversation_history[-max_history_turns:]).rstrip() + "\n\n"
            prompt += user_input
        else:
            raise ValueError(f"Unknown chat_template: {chat_template}")

        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        # For early checkpoints: do NOT update memory during generation.
        # Only update memory if explicitly "learning" into persistent memory.
        update_memory = (memory_mode == "persistent" and learning)

        print(f"{onyx_tag}{_dim(':')} ", end="", flush=True)
        generated_text = ""
        out_token_ids: List[int] = []
        printed_len = 0

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
            memory_states=memory_states,
            update_memory=update_memory,
            eos_token_id=eos,
            stop_tokens=stop_tokens,
            rep_window=rep_window,
            freq_penalty=freq_penalty,
            ban_run_len=ban_run_len,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        for token, new_mem_state in gen_gen:
            tid = int(token.item())
            if tid == eos:
                break
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
        if chat_template == "onyx":
            conversation_history.append(f"User: {user_input}\nAssistant: {generated_text}")
        else:
            conversation_history.append(f"{user_input}\n{generated_text}")

        if learning and memory_mode == "persistent" and memory_path:
            model.save_memory_states(memory_states, memory_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Hermes-2-Pro-Llama-3-8B")
    parser.add_argument("--memory", type=str, default="session", choices=["stateless", "session", "persistent"])
    parser.add_argument("--memory_path", type=str, default=None)
    parser.add_argument("--learning", action="store_true")

    # Experimental: load OnyxMHC checkpoints (matches --experimental_mhc training)
    parser.add_argument("--experimental_mhc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mhc_n", type=int, default=None)
    parser.add_argument("--mhc_mode", type=str, choices=["mhc", "hc"], default=None)
    parser.add_argument("--mhc_sinkhorn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mhc_sinkhorn_iters", type=int, default=None)

    # Decoding params
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=0.05)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--rep_window", type=int, default=128)
    parser.add_argument("--freq_penalty", type=float, default=0.3)
    parser.add_argument("--ban_run_len", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)

    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--chat_template", type=str, choices=["onyx", "raw"], default="onyx")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stream", action="store_true", default=True, help="Enable streaming (default)")
    group.add_argument("--no_stream", action="store_false", dest="stream", help="Disable streaming")

    args = parser.parse_args()

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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    try:
        model, _ = load_model(
            args.checkpoint,
            tokenizer,
            device=device,
            dtype=dtype,
            model_config_path=args.model_config,
            experimental_mhc=args.experimental_mhc,
            mhc_n=args.mhc_n,
            mhc_mode=args.mhc_mode,
            mhc_sinkhorn=args.mhc_sinkhorn,
            mhc_sinkhorn_iters=args.mhc_sinkhorn_iters,
        )
    except TypeError:
        # Back-compat with monkeypatched/older load_model implementations.
        model, _ = load_model(
            args.checkpoint,
            tokenizer,
            device=device,
            dtype=dtype,
            model_config_path=args.model_config,
        )

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
            rep_window=args.rep_window,
            freq_penalty=args.freq_penalty,
            ban_run_len=args.ban_run_len,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
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
        chat(
            model,
            tokenizer,
            device,
            dtype,
            memory_mode=args.memory,
            memory_path=args.memory_path,
            learning=args.learning,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens,
            stream=args.stream,
            system_prompt=args.system,
            rep_window=args.rep_window,
            freq_penalty=args.freq_penalty,
            ban_run_len=args.ban_run_len,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            chat_template=args.chat_template,
        )


if __name__ == "__main__":
    main()
