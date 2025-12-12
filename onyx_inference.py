"""
Onyx Inference Script

Usage:
  python onyx_inference.py --checkpoint path/to/checkpoint.pt
  python onyx_inference.py --checkpoint path/to/checkpoint.pt --learning  # enables memory updates
  python onyx_inference.py --checkpoint path/to/checkpoint.pt --memory persistent --memory_path memory.pt
"""

import argparse
import torch
from pathlib import Path

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from onyx_model import Onyx, OnyxConfig


def load_model(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get config from checkpoint or use default
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        if isinstance(cfg, dict):
            # Filter to only OnyxConfig fields
            import dataclasses
            valid_fields = {f.name for f in dataclasses.fields(OnyxConfig)}
            filtered_cfg = {k: v for k, v in cfg.items() if k in valid_fields}
            config = OnyxConfig(**filtered_cfg)
        else:
            config = cfg
    else:
        config = OnyxConfig()

    model = Onyx(config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Try loading directly
        model.load_state_dict(checkpoint)

    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model, config


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
    max_tokens: int = 512,
):
    """Interactive chat loop"""
    print("\n" + "="*60)
    print("Onyx Chat")
    print("="*60)
    print(f"Memory mode: {memory_mode}")
    print(f"Learning: {'ON' if learning else 'OFF'}")
    if memory_path:
        print(f"Memory path: {memory_path}")
    print("Type 'quit' or 'exit' to end, 'clear' to reset memory")
    print("="*60 + "\n")

    memory_states = None

    # Load persistent memory if exists
    if memory_mode == "persistent" and memory_path and Path(memory_path).exists():
        memory_states = model.load_memory_states(memory_path, device, dtype)
        print(f"Loaded memory from {memory_path}")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit']:
            break

        if user_input.lower() == 'clear':
            memory_states = None
            print("Memory cleared.\n")
            continue

        if user_input.lower() == 'save' and memory_path:
            if memory_states:
                model.save_memory_states(memory_states, memory_path)
                print(f"Memory saved to {memory_path}\n")
            else:
                print("No memory to save.\n")
            continue

        if user_input.lower().startswith('learning '):
            val = user_input.split()[1].lower()
            if val in ['on', 'true', '1']:
                learning = True
                print("Learning: ON\n")
            elif val in ['off', 'false', '0']:
                learning = False
                print("Learning: OFF\n")
            continue

        # Tokenize input
        input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)

        # Generate
        update_memory = learning or (memory_mode != "stateless")

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
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode response (skip input tokens)
        response = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Onyx: {response}\n")

        # Save memory if persistent and learning
        if memory_mode == "persistent" and memory_path and learning:
            model.save_memory_states(memory_states, memory_path)


def main():
    parser = argparse.ArgumentParser(description="Onyx Inference")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Hermes-2-Pro-Llama-3-8B")

    # Memory
    parser.add_argument("--memory", type=str, default="session",
                       choices=["stateless", "session", "persistent"],
                       help="Memory mode")
    parser.add_argument("--memory_path", type=str, default=None,
                       help="Path for persistent memory")
    parser.add_argument("--learning", action="store_true",
                       help="Enable learning mode (updates memory during inference)")

    # Generation
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)

    # Device
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/mps/cpu)")
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float16", "bfloat16"])

    # Single prompt mode
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single prompt (non-interactive mode)")

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Setup dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    # Load tokenizer
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed. Run: pip install transformers")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device, dtype)
    print(f"Model parameters: {model.get_num_params():,}")

    # Single prompt or chat
    if args.prompt:
        input_ids = tokenizer.encode(args.prompt, return_tensors='pt').to(device)
        output_ids, _ = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            memory_mode=args.memory,
            memory_path=args.memory_path,
            update_memory=args.learning,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(response)
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
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
