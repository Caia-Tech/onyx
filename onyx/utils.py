#!/usr/bin/env python3
"""
Utility functions for Onyx model.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Union


@torch.no_grad()
def generate_diagnostic_sample(
    model,
    tokenizer,
    prompt: str = "Hello",
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> str:
    """
    Generate a text sample for manual inspection (diagnostic purposes).

    This is a simple sampling function meant to be called manually when you want
    to see what the model is generating - NOT meant to be called during training.

    Args:
        model: The Onyx model (should be in eval mode or will be set to eval)
        tokenizer: The tokenizer (e.g., AutoTokenizer)
        prompt: The input prompt string (default: "Hello")
        max_tokens: Maximum number of tokens to generate (default: 50)
        temperature: Sampling temperature, higher = more random (default: 1.0)
        top_k: If set, only sample from top-k tokens (default: None)
        top_p: If set, nucleus sampling with cumulative probability p (default: None)
        repetition_penalty: Penalty for repeating tokens, >1.0 = less repetition (default: 1.0)
        device: Device to run on (default: infer from model)

    Returns:
        Generated text string (includes the prompt)

    Example usage:
        ```python
        # In a notebook or debug session
        from onyx.utils import generate_diagnostic_sample
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # ... load your model ...

        # Generate a sample
        sample = generate_diagnostic_sample(
            model, tokenizer,
            prompt="The quick brown",
            max_tokens=50,
            temperature=0.8
        )
        print(sample)
        ```
    """
    # Set model to eval mode
    was_training = model.training
    model.eval()

    # Infer device if not provided
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    B, S = input_ids.shape

    # Initialize memory states
    memory_states = model.init_memory_states(
        batch_size=B,
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    # Track tokens we've generated (for repetition penalty)
    generated_tokens = input_ids[0].tolist()

    # Generate tokens one at a time
    for _ in range(max_tokens):
        # Forward pass
        out = model(
            input_ids=input_ids,
            memory_states=memory_states,
            update_memories=True,
        )

        logits = out["logits"]  # [B, S, V]
        memory_states = out["memory_states"]

        # Get logits for last position
        next_token_logits = logits[0, -1, :]  # [V]

        # Apply repetition penalty if requested
        if repetition_penalty != 1.0:
            for token_id in set(generated_tokens):
                # If token was already generated, divide its logit by penalty
                next_token_logits[token_id] /= repetition_penalty

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Convert to probabilities
        probs = F.softmax(next_token_logits, dim=-1)

        # Apply top-k filtering
        if top_k is not None:
            topk_probs, topk_indices = probs.topk(top_k)
            # Zero out probabilities outside top-k
            probs_filtered = torch.zeros_like(probs)
            probs_filtered[topk_indices] = topk_probs
            probs = probs_filtered
            # Renormalize
            probs = probs / probs.sum()

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum_probs = sorted_probs.cumsum(dim=-1)

            # Find cutoff index
            cutoff_idx = (cumsum_probs > top_p).nonzero(as_tuple=True)[0]
            if len(cutoff_idx) > 0:
                cutoff_idx = cutoff_idx[0].item() + 1  # Include the token that pushes over threshold
            else:
                cutoff_idx = len(sorted_probs)

            # Keep only top-p tokens
            probs_filtered = torch.zeros_like(probs)
            probs_filtered[sorted_indices[:cutoff_idx]] = sorted_probs[:cutoff_idx]
            probs = probs_filtered
            # Renormalize
            probs = probs / probs.sum()

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Append to input_ids
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        generated_tokens.append(next_token.item())

        # Note: For efficiency, we could detach memory states here
        # but since this is diagnostic/manual use, clarity > efficiency
        memory_states = model.detach_memory_states(memory_states)

    # Decode generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    # Restore training mode if needed
    if was_training:
        model.train()

    return generated_text


def print_sample(
    model,
    tokenizer,
    prompt: str = "Hello",
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> None:
    """
    Generate and print a sample (convenience wrapper).

    Same arguments as generate_diagnostic_sample, but prints the result
    with nice formatting.

    Example:
        ```python
        from onyx.utils import print_sample

        print_sample(model, tokenizer, prompt="Once upon a time", max_tokens=100)
        ```
    """
    sample = generate_diagnostic_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        device=device,
    )

    print("\n" + "=" * 80)
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens} | Temperature: {temperature}")
    if top_k:
        print(f"Top-k: {top_k}")
    if top_p:
        print(f"Top-p: {top_p}")
    if repetition_penalty != 1.0:
        print(f"Repetition penalty: {repetition_penalty}")
    print("=" * 80)
    print(sample)
    print("=" * 80 + "\n")
