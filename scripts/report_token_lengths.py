#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from tokenizers import Tokenizer


_ROLE_PREFIX_RE = re.compile(r"^\\s*(system|user|assistant)\\s*:\\s*", flags=re.IGNORECASE)


def _strip_leading_role_prefixes(text: str) -> str:
    s = text
    for _ in range(4):
        m = _ROLE_PREFIX_RE.match(s)
        if not m:
            break
        s = s[m.end() :]
    return s.lstrip()


def _jsonl_record_to_doc(data: Dict[str, Any]) -> Optional[Union[str, Dict[str, Any]]]:
    """
    Copied (lightly) from `onyx/train.py` to match training-time behavior.
    """
    if not isinstance(data, dict):
        return None

    chat = data.get("chat")
    if not isinstance(chat, list):
        chat = data.get("chats")
    if not isinstance(chat, list):
        chat = data.get("messages")
    if isinstance(chat, list) and chat:
        role_content_msgs: List[Tuple[str, str]] = []
        for turn in chat:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role") or turn.get("from") or turn.get("speaker")
            content = turn.get("content") or turn.get("text") or turn.get("value")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            role = role.strip().lower()
            content = content.strip()
            if not content:
                continue
            if role in ("system", "user", "assistant"):
                role_content_msgs.append((role, content))

        if role_content_msgs:
            turns: List[Dict[str, str]] = []
            current_style = ""
            pending_user: Optional[str] = None

            for role, content in role_content_msgs:
                if role == "system":
                    current_style = content
                    continue
                if role == "user":
                    pending_user = content
                    continue
                if role == "assistant":
                    if not pending_user:
                        continue
                    turns.append({"user": pending_user, "assistant": content, "style": current_style})
                    pending_user = None

            if turns:
                if len(turns) == 1:
                    t = turns[0]
                    return {"system": t.get("style", ""), "user": t["user"], "assistant": t["assistant"]}
                return {"chat": turns}

            joined = "\n\n".join([c for _, c in role_content_msgs]).strip()
            return joined if joined else None

        cleaned: List[Dict[str, str]] = []
        for turn in chat:
            if not isinstance(turn, dict):
                continue
            user = turn.get("user")
            assistant = turn.get("assistant")
            style = turn.get("style", "")
            if not isinstance(user, str) or not isinstance(assistant, str):
                continue
            if not user or not assistant:
                continue
            if not isinstance(style, str):
                style = ""
            cleaned.append({"user": user, "assistant": assistant, "style": style})
        if cleaned:
            return {"chat": cleaned}
        return None

    text = data.get("text") or data.get("content") or data.get("input") or ""
    if not text:
        system = data.get("system")
        user = data.get("user")
        assistant = data.get("assistant")
        if isinstance(user, str) and isinstance(assistant, str):
            return {
                "system": system if isinstance(system, str) else "",
                "user": user,
                "assistant": assistant,
            }
        return None

    if isinstance(text, str) and text:
        return text
    return None


def _iter_jsonl_docs(filepath: str) -> Iterator[Union[str, Dict[str, Any]]]:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = _jsonl_record_to_doc(data)
            if doc is not None:
                yield doc


def _load_eos_token_id(tokenizer_dir: Path, tok: Tokenizer) -> Optional[int]:
    cfg_path = tokenizer_dir / "tokenizer_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            eos = cfg.get("eos_token")
            if isinstance(eos, str):
                return tok.token_to_id(eos)
            if isinstance(eos, dict) and isinstance(eos.get("content"), str):
                return tok.token_to_id(eos["content"])
        except Exception:
            pass
    for candidate in ("[EOS]", "</s>", "<|endoftext|>"):
        tid = tok.token_to_id(candidate)
        if tid is not None:
            return tid
    return None


def _encode_ids(tok: Tokenizer, text: str) -> List[int]:
    return tok.encode(text).ids


def _doc_token_len(tok: Tokenizer, doc: Union[str, Dict[str, Any]], *, eod_token_id: int) -> Optional[int]:
    if isinstance(doc, str):
        ids = _encode_ids(tok, doc)
        if not ids:
            return None
        return len(ids) + 1  # +EOD

    if isinstance(doc, dict) and ("user" in doc and "assistant" in doc):
        system = doc.get("system") if isinstance(doc.get("system"), str) else ""
        user = doc.get("user") if isinstance(doc.get("user"), str) else ""
        assistant = doc.get("assistant") if isinstance(doc.get("assistant"), str) else ""

        system = _strip_leading_role_prefixes(system.strip())
        user = _strip_leading_role_prefixes(user.strip())
        assistant = _strip_leading_role_prefixes(assistant.strip())

        if not user or not assistant:
            return None

        prompt = ""
        if system:
            prompt += f"System: {system}\n\n"
        prompt += f"User: {user}\nAssistant: "

        prompt_ids = _encode_ids(tok, prompt)
        assistant_ids = _encode_ids(tok, assistant)
        if not prompt_ids or not assistant_ids:
            return None
        return len(prompt_ids) + len(assistant_ids) + 1  # +EOD

    if isinstance(doc, dict) and "chat" in doc and isinstance(doc["chat"], list):
        total = 0
        prev_style = ""
        first_turn = True
        any_turn = False

        for turn in doc["chat"]:
            if not isinstance(turn, dict):
                continue
            user = turn.get("user") if isinstance(turn.get("user"), str) else ""
            assistant = turn.get("assistant") if isinstance(turn.get("assistant"), str) else ""
            style = turn.get("style") if isinstance(turn.get("style"), str) else ""

            user = _strip_leading_role_prefixes(user.strip())
            assistant = _strip_leading_role_prefixes(assistant.strip())
            style = _strip_leading_role_prefixes(style.strip())
            if not user or not assistant:
                continue

            prompt = ""
            if not first_turn:
                prompt += "\n\n"
            if style and (first_turn or style != prev_style):
                prompt += f"System: {style}\n\n"
            prompt += f"User: {user}\nAssistant: "

            prompt_ids = _encode_ids(tok, prompt)
            assistant_ids = _encode_ids(tok, assistant)
            if not prompt_ids or not assistant_ids:
                continue

            total += len(prompt_ids) + len(assistant_ids)
            prev_style = style
            first_turn = False
            any_turn = True

        if not any_turn:
            return None
        return total + 1  # +EOD

    return None


@dataclass
class Report:
    total_entries: int = 0
    total_tokenized: int = 0
    total_skipped: int = 0
    le_256: int = 0
    le_512: int = 0
    le_1024: int = 0
    le_2048: int = 0
    gt_2048: int = 0
    token_lens: List[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.token_lens is None:
            self.token_lens = []

    def add(self, n: int) -> None:
        self.total_tokenized += 1
        self.token_lens.append(n)
        if n <= 256:
            self.le_256 += 1
        if n <= 512:
            self.le_512 += 1
        if n <= 1024:
            self.le_1024 += 1
        if n <= 2048:
            self.le_2048 += 1
        if n > 2048:
            self.gt_2048 += 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Report token lengths for a JSONL using the Onyx training doc format.")
    ap.add_argument("--data", required=True, help="Path to .jsonl file")
    ap.add_argument("--tokenizer_dir", required=True, help="HF tokenizer directory (must contain tokenizer.json)")
    ap.add_argument("--max_entries", type=int, default=0, help="If >0, only process this many entries")
    ap.add_argument("--json_out", type=str, default="", help="Optional path to write JSON report")
    args = ap.parse_args()

    data_path = Path(args.data)
    tok_dir = Path(args.tokenizer_dir)
    tok_path = tok_dir / "tokenizer.json"
    if not data_path.exists():
        raise SystemExit(f"Data not found: {data_path}")
    if not tok_path.exists():
        raise SystemExit(f"tokenizer.json not found: {tok_path}")

    tok = Tokenizer.from_file(str(tok_path))
    eos_id = _load_eos_token_id(tok_dir, tok)
    if eos_id is None:
        raise SystemExit("Could not determine eos/eod token id from tokenizer_dir")

    rep = Report()
    rep.total_entries = 0

    for doc in _iter_jsonl_docs(str(data_path)):
        rep.total_entries += 1
        n = _doc_token_len(tok, doc, eod_token_id=eos_id)
        if n is None:
            rep.total_skipped += 1
        else:
            rep.add(n)

        if args.max_entries and rep.total_entries >= args.max_entries:
            break

    stats: Dict[str, Any] = {
        "data": str(data_path),
        "tokenizer_dir": str(tok_dir),
        "eod_token_id": eos_id,
        "total_entries": rep.total_entries,
        "tokenized": rep.total_tokenized,
        "skipped": rep.total_skipped,
        "counts": {
            "<=256": rep.le_256,
            "<=512": rep.le_512,
            "<=1024": rep.le_1024,
            "<=2048": rep.le_2048,
            ">2048": rep.gt_2048,
        },
    }

    if rep.token_lens:
        stats["length_stats"] = {
            "min": min(rep.token_lens),
            "p50": int(statistics.median(rep.token_lens)),
            "p95": int(statistics.quantiles(rep.token_lens, n=20)[18]),  # 95th percentile
            "max": max(rep.token_lens),
            "mean": round(sum(rep.token_lens) / len(rep.token_lens), 3),
        }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(stats, indent=2, sort_keys=False), encoding="utf-8")
        os.replace(tmp, out_path)

    print(json.dumps(stats, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()

