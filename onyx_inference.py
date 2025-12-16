"""
Onyx Inference Script

Usage:
  python onyx_inference.py --checkpoint path/to/checkpoint.pt
  python onyx_inference.py --checkpoint path/to/checkpoint.pt --learning
  python onyx_inference.py --checkpoint path/to/checkpoint.pt --stream
  python onyx_inference.py --checkpoint path/to/checkpoint.pt --prompt "Hello"
"""

import argparse
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator, Tuple

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from onyx_model import Onyx, OnyxConfig


def load_model(checkpoint_path: str, device: torch.device, dtype: torch.dtype, model_config_path: str = None):
    """Load model from checkpoint with auto-vocab resizing"""
    import json
    import dataclasses

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config = None

    if model_config_path and Path(model_config_path).exists():
        print(f"Loading model config from: {model_config_path}")
        with open(model_config_path) as f:
            cfg_json = json.load(f)
        flat_cfg = {}
        for key, value in cfg_json.items():
            if isinstance(value, dict):
                flat_cfg.update(value)
            else:
                flat_cfg[key] = value
        valid_fields = {f.name for f in dataclasses.fields(OnyxConfig)}
        filtered_cfg = {k: v for k, v in flat_cfg.items() if k in valid_fields}
        config = OnyxConfig(**filtered_cfg)

    elif 'config' in checkpoint:
        cfg = checkpoint['config']
        if isinstance(cfg, dict) and 'model_config_path' in cfg:
            mcp = cfg['model_config_path']
            # Some checkpoints store None/empty for model_config_path; skip in that case
            if isinstance(mcp, (str, Path)) and mcp:
                ckpt_dir = Path(checkpoint_path).parent
                possible_paths = [
                    ckpt_dir / mcp,
                    ckpt_dir.parent / mcp,
                    Path(mcp),
                ]
                for p in possible_paths:
                    if p.exists():
                        print(f"Loading model config from checkpoint reference: {p}")
                        with open(p) as f:
                            cfg_json = json.load(f)
                        flat_cfg = {}
                        for key, value in cfg_json.items():
                            if isinstance(value, dict):
                                flat_cfg.update(value)
                            else:
                                flat_cfg[key] = value
                        valid_fields = {f.name for f in dataclasses.fields(OnyxConfig)}
                        filtered_cfg = {k: v for k, v in flat_cfg.items() if k in valid_fields}
                        config = OnyxConfig(**filtered_cfg)
                        break

    if config is None:
        if 'config' in checkpoint:
            cfg = checkpoint['config']
            if isinstance(cfg, dict):
                valid_fields = {f.name for f in dataclasses.fields(OnyxConfig)}
                filtered_cfg = {k: v for k, v in cfg.items() if k in valid_fields}
                if filtered_cfg:
                    config = OnyxConfig(**filtered_cfg)
                else:
                    print("Warning: No valid model config found, using default config")
                    config = OnyxConfig()
            else:
                config = cfg
        else:
            print("Warning: No config in checkpoint, using default config")
            config = OnyxConfig()

    print(f"Model config: d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
    model = Onyx(config)

    # Determine state dict
    state_dict = None
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # [FIX] Handle Vocab Mismatch (e.g., 128256 vs 128258)
    if 'embed.weight' in state_dict:
        ckpt_vocab = state_dict['embed.weight'].shape[0]
        model_vocab = config.vocab_size
        
        if ckpt_vocab != model_vocab:
            print(f"Warning: Checkpoint vocab ({ckpt_vocab}) != Config ({model_vocab}). Auto-resizing...")
            
            for key in ['embed.weight', 'lm_head.weight']:
                if key in state_dict:
                    w = state_dict[key]
                    if w.shape[0] < model_vocab:
                        # Pad with zeros
                        print(f"  - Padding {key}: {w.shape} -> ({model_vocab}, {w.shape[1]})")
                        pad_size = model_vocab - w.shape[0]
                        pad = torch.zeros((pad_size, w.shape[1]), dtype=w.dtype, device=w.device)
                        state_dict[key] = torch.cat([w, pad], dim=0)
                    elif w.shape[0] > model_vocab:
                        # Truncate
                        print(f"  - Truncating {key}: {w.shape} -> ({model_vocab}, {w.shape[1]})")
                        state_dict[key] = w[:model_vocab, :]

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model, config


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample next token with various strategies"""

    # Repetition penalty
    if repetition_penalty != 1.0 and generated_tokens is not None and generated_tokens.numel() > 0:
        for token_id in generated_tokens.unique():
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

    # Deterministic generation for temp=0
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    # Top-k
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float('-inf')

    # Min-p (dynamic threshold based on top token probability)
    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        top_prob = probs.max(dim=-1, keepdim=True).values
        min_prob_threshold = top_prob * min_p
        logits[probs < min_prob_threshold] = float('-inf')

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_stream(
    model: Onyx,
    input_ids: torch.Tensor,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.0,
    repetition_penalty: float = 1.1,
    memory_states: Optional[List[Dict[str, Any]]] = None,
    memory_mode: str = "session",
    update_memory: bool = True,
    eos_token_id: Optional[int] = None,
    stop_tokens: Optional[List[int]] = None,
    use_kv_cache: bool = True,
) -> Generator[Tuple[torch.Tensor, List[Dict[str, Any]]], None, None]:
    """Generate tokens one at a time, yielding each token"""

    model.eval()
    B, S = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype

    if memory_states is None:
        memory_states = model.init_memory_states(B, device, dtype)

    # Process prompt
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

    generated_tokens = torch.tensor([], dtype=torch.long, device=device)
    stop_tokens = stop_tokens or []
    if eos_token_id is not None:
        stop_tokens.append(eos_token_id)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = outputs["logits"][:, -1:, :]

            next_token = sample_token(
                logits.squeeze(1),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated_tokens,
            )

            generated_tokens = torch.cat([generated_tokens, next_token.squeeze(0)])

            yield next_token, memory_states

            if next_token.item() in stop_tokens:
                break

            outputs = model(
                next_token,
                memory_states=memory_states,
                update_memories=update_memory,
                inference_mode=True,
                kv_cache=kv_cache,
                position_offset=position_offset,
            )
            memory_states = outputs["memory_states"]
            kv_cache = outputs.get("kv_cache") if use_kv_cache else None
            position_offset += 1


def chat(
    model: Onyx,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    memory_mode: str = "session",
    memory_path: str = None,
    learning: bool = False,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.0,
    repetition_penalty: float = 1.1,
    max_tokens: int = 512,
    stream: bool = True,
    system_prompt: str = None,
):
    """Interactive chat loop with streaming"""

    print("\n" + "="*60)
    print("  Onyx Chat")
    print("="*60)
    print(f"  Memory: {memory_mode} | Learning: {'ON' if learning else 'OFF'} | Stream: {'ON' if stream else 'OFF'}")
    print("-"*60)
    print("  Commands:")
    print("    /quit, /exit    - End session")
    print("    /clear          - Reset memory")
    print("    /save           - Save memory (if persistent)")
    print("    /learning on|off- Toggle learning mode")
    print("    /temp <value>   - Set temperature")
    print("    /system <text>  - Set system prompt")
    print("="*60 + "\n")

    memory_states = None
    conversation_history = []

    if memory_mode == "persistent" and memory_path and Path(memory_path).exists():
        memory_states = model.load_memory_states(memory_path, device, dtype)
        print(f"[Loaded memory from {memory_path}]\n")

    # Get special tokens
    eos_token_id = tokenizer.eos_token_id
    stop_tokens = [eos_token_id] if eos_token_id else []

    # Explicit Llama-3 Stop Tokens
    llama3_stops = ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|end|>", "</s>"]
    for token in llama3_stops:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id and token_id not in stop_tokens:
            stop_tokens.append(token_id)
            
    print(f"[Debug] Stop Tokens: {stop_tokens}")

    while True:
        try:
            user_input = input("\033[92mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith('/'):
            cmd_parts = user_input[1:].split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else None

            if cmd in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif cmd == 'clear':
                memory_states = None
                conversation_history = []
                print("[Memory cleared]\n")
                continue
            elif cmd == 'save' and memory_path:
                if memory_states:
                    model.save_memory_states(memory_states, memory_path)
                    print(f"[Memory saved to {memory_path}]\n")
                else:
                    print("[No memory to save]\n")
                continue
            elif cmd == 'learning' and cmd_arg:
                if cmd_arg.lower() in ['on', 'true', '1']:
                    learning = True
                    print("[Learning: ON]\n")
                elif cmd_arg.lower() in ['off', 'false', '0']:
                    learning = False
                    print("[Learning: OFF]\n")
                continue
            elif cmd == 'temp' and cmd_arg:
                try:
                    temperature = float(cmd_arg)
                    print(f"[Temperature: {temperature}]\n")
                except ValueError:
                    print("[Invalid temperature value]\n")
                continue
            elif cmd == 'system' and cmd_arg:
                system_prompt = cmd_arg
                print(f"[System prompt set]\n")
                continue
            else:
                print(f"[Unknown command: {cmd}]\n")
                continue

        # Build prompt
        prompt = ""
        if system_prompt:
            prompt += f"System: {system_prompt}\n\n"

        # Add conversation history (last few turns)
        for turn in conversation_history[-4:]:
            prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

        prompt += f"User: {user_input}\nAssistant:"

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        update_memory = learning or (memory_mode != "stateless")

        print("\033[94mOnyx:\033[0m ", end="", flush=True)

        start_time = time.time()
        generated_text = ""
        token_count = 0

        if stream:
            for token, memory_states in generate_stream(
                model,
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                memory_states=memory_states,
                memory_mode=memory_mode,
                update_memory=update_memory,
                eos_token_id=eos_token_id,
                stop_tokens=stop_tokens,
            ):
                token_text = tokenizer.decode(token[0], skip_special_tokens=True)
                print(token_text, end="", flush=True)
                generated_text += token_text
                token_count += 1
        else:
            # Non-streaming fallback
            output_ids, memory_states = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                memory_states=memory_states,
                memory_mode=memory_mode if not learning else "session",
                memory_path=memory_path if memory_mode == "persistent" and not learning else None,
                update_memory=update_memory,
                eos_token_id=eos_token_id,
            )
            generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            token_count = output_ids.shape[1] - input_ids.shape[1]
            print(generated_text, end="")

        elapsed = time.time() - start_time
        tok_per_sec = token_count / elapsed if elapsed > 0 else 0

        print(f"\n\033[90m[{token_count} tokens, {tok_per_sec:.1f} tok/s]\033[0m\n")

        # Save to history
        conversation_history.append({
            'user': user_input,
            'assistant': generated_text.strip()
        })

        # Save memory if persistent and learning
        if memory_mode == "persistent" and memory_path and learning:
            model.save_memory_states(memory_states, memory_path)


def main():
    parser = argparse.ArgumentParser(description="Onyx Inference")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--model_config", type=str, default=None,
                       help="Path to model config JSON")
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Hermes-2-Pro-Llama-3-8B")

    # Memory
    parser.add_argument("--memory", type=str, default="session",
                       choices=["stateless", "session", "persistent"])
    parser.add_argument("--memory_path", type=str, default=None)
    parser.add_argument("--learning", action="store_true",
                       help="Enable learning mode")

    # Generation
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=0.0,
                       help="Min-p sampling threshold")
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_tokens", type=int, default=512)

    # Output
    parser.add_argument("--stream", action="store_true", default=True,
                       help="Stream output tokens")
    parser.add_argument("--no_stream", action="store_true",
                       help="Disable streaming")

    # Device
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float16", "bfloat16"])

    # Single prompt mode
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single prompt (non-interactive)")
    parser.add_argument("--system", type=str, default=None,
                       help="System prompt")

    args = parser.parse_args()

    # Handle stream flag
    stream = args.stream and not args.no_stream

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"Device: {device} | Dtype: {dtype}")

    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed. Run: pip install transformers")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device, dtype, args.model_config)
    print(f"Model parameters: {model.get_num_params():,}")

    if args.prompt:
        # Single prompt mode with streaming
        prompt = args.prompt
        if args.system:
            prompt = f"System: {args.system}\n\nUser: {prompt}\nAssistant:"

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Stop tokens for single prompt
        eos_token_id = tokenizer.eos_token_id
        stop_tokens = [eos_token_id] if eos_token_id else []
        llama3_stops = ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"]
        for token in llama3_stops:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id and token_id not in stop_tokens:
                stop_tokens.append(token_id)

        if stream:
            for token, _ in generate_stream(
                model,
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
                memory_mode=args.memory,
                update_memory=args.learning,
                eos_token_id=eos_token_id,
                stop_tokens=stop_tokens
            ):
                print(tokenizer.decode(token[0], skip_special_tokens=True), end="", flush=True)
            print()
        else:
            output_ids, _ = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                memory_mode=args.memory,
                update_memory=args.learning,
                eos_token_id=eos_token_id,
            )
            print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
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
