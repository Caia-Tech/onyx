#!/usr/bin/env python3
"""Count total tokens in a JSONL dataset using the Onyx tokenizer."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers")
    sys.exit(1)


def read_jsonl(path: str) -> Iterator[dict]:
    """Read JSONL file line by line."""
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}", file=sys.stderr)
                continue


def extract_text(entry: dict, text_field: str = "text") -> str:
    """Extract text from a JSONL entry."""
    if text_field in entry:
        return str(entry[text_field])

    # Try common alternatives
    for field in ["content", "text", "message", "body"]:
        if field in entry:
            return str(entry[field])

    # If no known field, try to stringify the whole entry
    return str(entry)


def main():
    parser = argparse.ArgumentParser(description="Count tokens in JSONL dataset")
    parser.add_argument("dataset", help="Path to JSONL dataset file")
    parser.add_argument("--tokenizer", default="/Users/owner/Desktop/caiatech/datasets/tokenizer",
                        help="Path to tokenizer directory")
    parser.add_argument("--text_field", default="text",
                        help="JSON field containing text (default: 'text')")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Only process first N documents (for testing)")
    parser.add_argument("--report_every", type=int, default=10000,
                        help="Report progress every N documents")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    # Count tokens
    print(f"Counting tokens in: {args.dataset}")
    print(f"Text field: {args.text_field}")
    if args.max_docs:
        print(f"Max documents: {args.max_docs:,}")
    print("-" * 70)

    total_tokens = 0
    total_docs = 0
    total_chars = 0

    try:
        for entry in read_jsonl(args.dataset):
            if args.max_docs and total_docs >= args.max_docs:
                break

            text = extract_text(entry, args.text_field)
            total_chars += len(text)

            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)
            total_docs += 1

            if total_docs % args.report_every == 0:
                print(f"Processed {total_docs:,} docs | {total_tokens:,} tokens | "
                      f"Avg: {total_tokens/total_docs:.1f} tok/doc")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    except Exception as e:
        print(f"\nError processing dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # Final report
    print("=" * 70)
    print(f"Total documents:  {total_docs:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total tokens:     {total_tokens:,}")
    if total_docs > 0:
        print(f"Avg tokens/doc:   {total_tokens/total_docs:.1f}")
        print(f"Avg chars/token:  {total_chars/total_tokens:.2f}")
    print("=" * 70)

    # Time estimates
    if total_tokens > 0:
        print("\nTraining time estimates (approximate):")
        # Tokens per step = batch_size * seq_len
        tokens_per_step = 4 * 1024  # Current config
        total_steps = total_tokens / tokens_per_step

        # At ~3600 tok/s
        throughput = 3600
        seconds = total_tokens / throughput
        hours = seconds / 3600

        print(f"  Steps (batch=4, seq=1024): ~{total_steps:,.0f}")
        print(f"  Time at 3,600 tok/s: ~{hours:.1f} hours ({seconds/60:.1f} minutes)")

        # At different throughputs
        for tps in [5000, 10000, 50000]:
            h = (total_tokens / tps) / 3600
            print(f"  Time at {tps:,} tok/s: ~{h:.1f} hours")


if __name__ == "__main__":
    main()
