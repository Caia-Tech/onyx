#!/usr/bin/env python3
import argparse
import json
import os
import platform
import sys
import time
import traceback
import unicodedata
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import tokenizers as tokenizers_pkg
import transformers as transformers_pkg
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


# ----------------------------
# Built-in seed tokens (code + logs + JSON glue), conservative-but-useful.
# These are NOT "special tokens" (they behave like normal tokens).
# Keep this list reasonably sized to avoid stealing too much vocab budget.
# ----------------------------
DEFAULT_SEED_TOKENS: List[str] = [
    # --- whitespace / structure ---
    "\n", "\n\n", "\n\n\n",
    "\r\n",
    "\t",
    "  ", "    ",  # 2 and 4 spaces
    "\n  ", "\n    ", "\n\t",

    # --- JSON / config glue ---
    '{"', '"}', "{\n", "}\n", "}\n\n",
    '":', '": ', '",', '", ',
    '["', '"]',
    "\\n", "\\t", '\\"', "\\\\",
    "null", "true", "false",

    # --- common code operators ---
    "==", "!=", "<=", ">=",
    "===", "!==",
    "&&", "||", "??",
    "->", "=>", "::",
    "+=", "-=", "*=", "/=", "%=",
    "++", "--",
    "<<", ">>",
    "&=", "|=", "^=",
    "&&=", "||=",

    # --- comments / docstrings ---
    "//", "/*", "*/",
    "# ",
    '"""', "'''",

    # --- bracket + separator glue ---
    "();", ");", ");\n",
    "{}", "{ }",
    "[]", "[ ]",
    "()", "( )",
    "},", "},\n",
    "];", "];\n",
    "),", "),\n",
    "):", "):\n",
    ":\n", ":\n  ", ":\n    ",  # YAML/Python blocks
    " = ", " == ", " != ", " <= ", " >= ",
    " -> ", " => ", " :: ",

    # --- common language-ish punctuation combos ---
    ".)", ").", ".,", ".\n",
    ", ", "; ", ": ",
    " ,", " ;", " :",  # sometimes appears in messy logs/text
    "...\n", "...",

    # --- paths / urls ---
    "://", "http://", "https://", "www.",
    "./", "../",
    "/usr/", "/etc/", "/var/", "/home/",
    "C:\\", "\\\\", "\\",

    # --- file extensions (high frequency across code/logs) ---
    ".json", ".yaml", ".yml", ".toml",
    ".py", ".ipynb", ".js", ".ts", ".tsx",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp",
    ".rs", ".go", ".java", ".kt", ".swift",
    ".sh", ".bash", ".zsh",
    ".md", ".txt", ".log", ".csv",

    # --- logging / error patterns ---
    "INFO", "WARN", "WARNING", "ERROR", "DEBUG", "TRACE",
    " INFO ", " WARN ", " ERROR ", " DEBUG ",  # spaced variants common in logs
    "Exception", "Exception: ",
    "Traceback (most recent call last):",
    "Caused by: ",
    " at ",  # Java-ish
    "AssertionError", "KeyError", "ValueError", "TypeError", "RuntimeError",

    # --- HTTP-ish (logs) ---
    "GET ", "POST ", "PUT ", "DELETE ", "PATCH ",
    "HTTP/1.1", "HTTP/2",
    " 200 ", " 201 ", " 204 ", " 301 ", " 302 ", " 400 ", " 401 ", " 403 ", " 404 ", " 409 ", " 429 ", " 500 ", " 502 ", " 503 ",

    # --- a few extremely common code keywords WITH trailing space (more useful than bare words) ---
    "def ", "class ", "return ", "import ", "from ", "as ",
    "if ", "elif ", "else ", "for ", "while ", "try:", "except ", "finally:",
    "function ", "const ", "let ", "var ",
    "public ", "private ", "protected ", "static ",
    "async ", "await ",
]


def _utc_iso(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _truncate(s: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _is_glob_pattern(p: str) -> bool:
    return any(ch in p for ch in ("*", "?", "["))


def _file_info(path: str) -> Dict[str, Any]:
    try:
        st = os.stat(path)
        return {
            "path": path,
            "size_bytes": int(st.st_size),
            "mtime_utc": _utc_iso(st.st_mtime),
        }
    except Exception as e:
        return {"path": path, "stat_error": repr(e)}


def resolve_input_files(data_glob: str) -> List[str]:
    p = Path(data_glob)
    files: List[str]
    if p.exists() and p.is_dir():
        files = sorted(glob(str(p / "**" / "*.jsonl"), recursive=True))
    elif p.exists() and p.is_file():
        files = [str(p)]
    elif _is_glob_pattern(data_glob):
        files = sorted(glob(data_glob, recursive=True))
    else:
        files = [data_glob]

    files = [f for f in files if os.path.isfile(f)]
    return files


def _extract_text(obj: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Return (kind, text) or None."""
    text = (
        obj.get("text")
        or obj.get("content")
        or obj.get("input")
        or obj.get("prompt")
        or ""
    )
    if isinstance(text, str) and text.strip():
        return ("text_field", text)

    prompt = obj.get("prompt")
    completion = obj.get("completion")
    if isinstance(prompt, str) and isinstance(completion, str):
        p = prompt.strip()
        c = completion.strip()
        if p and c:
            return ("prompt_completion", f"{p}\n{c}")

    user = obj.get("user")
    assistant = obj.get("assistant")
    system = obj.get("system", "")
    if isinstance(user, str) and isinstance(assistant, str):
        system = system if isinstance(system, str) else ""
        system = system.strip()
        user = user.strip()
        assistant = assistant.strip()
        if user and assistant:
            if system:
                return (
                    "user_assistant",
                    f"System: {system}\n\nUser: {user}\nAssistant: {assistant}",
                )
            return ("user_assistant", f"User: {user}\nAssistant: {assistant}")

    chat = obj.get("chat")
    if isinstance(chat, list):
        parts: List[str] = []
        for turn in chat:
            if not isinstance(turn, dict):
                continue

            u = turn.get("user")
            a = turn.get("assistant")
            if isinstance(u, str) and u.strip():
                parts.append(f"User: {u.strip()}")
            if isinstance(a, str) and a.strip():
                parts.append(f"Assistant: {a.strip()}")

            role = turn.get("role") or turn.get("from")
            content = turn.get("content") or turn.get("value") or turn.get("text")
            if isinstance(role, str) and isinstance(content, str) and content.strip():
                role_label = role.strip().capitalize() or "Message"
                parts.append(f"{role_label}: {content.strip()}")

        if parts:
            return ("chat_list", "\n".join(parts))

    messages = obj.get("messages")
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(role, str) and isinstance(content, str) and content.strip():
                role_label = role.strip().capitalize() or "Message"
                parts.append(f"{role_label}: {content.strip()}")
        if parts:
            return ("messages", "\n".join(parts))

    return None


def _clean_text(text: str, normalize: str) -> str:
    """Avoid learning garbage tokens: normalize (optional) + drop most control chars (keep \\n and \\t)."""
    if normalize == "nfkc":
        text = unicodedata.normalize("NFKC", text)

    out_chars: List[str] = []
    for ch in text:
        if ch in ("\n", "\t"):
            out_chars.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C"):  # control/surrogate/private-use/etc.
            continue
        out_chars.append(ch)
    return "".join(out_chars)


def _replacement_frac(text: str) -> float:
    if not text:
        return 1.0
    return text.count("\uFFFD") / len(text)


def _unescape_seed_token(t: str) -> str:
    # Minimal unescape so a file can contain "\n" and "\t" literals.
    # We do NOT do full python unicode-escape (keeps this predictable).
    return (
        t.replace("\\r\\n", "\r\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace("\\\\", "\\")
    )


def _load_seed_tokens(path: str, *, unescape: bool) -> List[str]:
    toks: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw:
                continue
            s = raw.lstrip()
            if s.startswith("#"):
                continue
            toks.append(_unescape_seed_token(raw) if unescape else raw)

    # stable unique preserving order
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _stable_unique(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in seq:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


class CorpusIterator:
    def __init__(
        self,
        files: List[str],
        report: Dict[str, Any],
        *,
        normalize: str,
        max_replacement_frac: float,
        max_text_samples: int,
        max_dropped_samples: int,
        truncate_chars: int,
        status_every: int,
    ) -> None:
        self._files = files
        self._report = report
        self._normalize = normalize
        self._max_replacement_frac = max_replacement_frac
        self._max_text_samples = max_text_samples
        self._max_dropped_samples = max_dropped_samples
        self._truncate_chars = truncate_chars
        self._status_every = status_every

    def _note_dropped(self, path: str, line_no: int, reason: str, raw: str) -> None:
        dropped = self._report["corpus"]["dropped_samples"]
        if len(dropped) >= self._max_dropped_samples:
            return
        dropped.append(
            {
                "file": path,
                "line": line_no,
                "reason": reason,
                "raw": _truncate(raw.rstrip("\n"), self._truncate_chars),
            }
        )

    def __iter__(self) -> Iterator[str]:
        totals = self._report["corpus"]["totals"]
        per_file = self._report["corpus"]["per_file"]
        yielded = 0

        for path in self._files:
            pf = per_file.setdefault(
                path,
                {
                    "lines_total": 0,
                    "lines_empty": 0,
                    "json_errors": 0,
                    "non_dict": 0,
                    "no_text": 0,
                    "yielded": 0,
                    "kinds": {},
                },
            )

            try:
                f = open(path, "r", encoding="utf-8", errors="replace")
            except Exception as e:
                self._note_dropped(path, 0, f"open_error:{type(e).__name__}", repr(e))
                totals["open_errors"] += 1
                continue

            with f:
                for line_no, raw in enumerate(f, start=1):
                    pf["lines_total"] += 1
                    totals["lines_total"] += 1

                    line = raw.strip()
                    if not line:
                        pf["lines_empty"] += 1
                        totals["lines_empty"] += 1
                        continue

                    try:
                        obj = json.loads(line)
                    except Exception:
                        pf["json_errors"] += 1
                        totals["json_errors"] += 1
                        self._note_dropped(path, line_no, "json_error", raw)
                        continue

                    if not isinstance(obj, dict):
                        pf["non_dict"] += 1
                        totals["non_dict"] += 1
                        self._note_dropped(path, line_no, "non_dict", raw)
                        continue

                    extracted = _extract_text(obj)
                    if extracted is None:
                        pf["no_text"] += 1
                        totals["no_text"] += 1
                        self._note_dropped(path, line_no, "no_text", raw)
                        continue

                    kind, text = extracted
                    text = text.strip()
                    if not text:
                        pf["no_text"] += 1
                        totals["no_text"] += 1
                        self._note_dropped(path, line_no, "empty_text", raw)
                        continue

                    text = _clean_text(text, self._normalize).strip()
                    if not text:
                        pf["no_text"] += 1
                        totals["no_text"] += 1
                        self._note_dropped(path, line_no, "empty_after_clean", raw)
                        continue

                    if self._max_replacement_frac > 0:
                        frac = _replacement_frac(text)
                        if frac > self._max_replacement_frac:
                            pf["no_text"] += 1
                            totals["no_text"] += 1
                            self._note_dropped(
                                path,
                                line_no,
                                f"too_many_replacement_chars:{frac:.4f}",
                                raw,
                            )
                            continue

                    pf["yielded"] += 1
                    totals["yielded"] += 1
                    pf["kinds"][kind] = pf["kinds"].get(kind, 0) + 1
                    totals["kinds"][kind] = totals["kinds"].get(kind, 0) + 1

                    examples = self._report["corpus"]["text_samples"]
                    if len(examples) < self._max_text_samples:
                        examples.append(_truncate(text, self._truncate_chars))

                    yielded += 1
                    if self._status_every > 0 and yielded % self._status_every == 0:
                        print(
                            f"[make_tokenizer] yielded={yielded} lines_total={totals['lines_total']} "
                            f"json_errors={totals['json_errors']} no_text={totals['no_text']}",
                            file=sys.stderr,
                        )
                    yield text


def build_tokenizer(normalize: str, add_prefix_space: bool, bpe_dropout: float) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="[UNK]", dropout=bpe_dropout))
    if normalize == "nfkc":
        tok.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
    tok.decoder = decoders.ByteLevel()
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", required=True, help="Path, directory, or glob for .jsonl")
    ap.add_argument("--out_dir", required=True)

    # Key knobs for “more general” tokenizers:
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--min_frequency", type=int, default=5)
    ap.add_argument(
        "--bpe_dropout",
        type=float,
        default=0.0,
        help="BPE dropout for subword regularization (try 0.05-0.2 for training; use 0.0 for deterministic)",
    )
    ap.add_argument(
        "--max_replacement_frac",
        type=float,
        default=0.01,
        help="Drop samples where U+FFFD replacement-char fraction exceeds this (0 disables)",
    )

    ap.add_argument(
        "--normalize",
        choices=["none", "nfkc"],
        default="nfkc",
        help="Unicode normalization (use 'none' for very code-heavy corpora)",
    )
    ap.add_argument("--add_prefix_space", action="store_true", default=True)
    ap.add_argument(
        "--no_add_prefix_space",
        action="store_false",
        dest="add_prefix_space",
        help="Disable ByteLevel add_prefix_space",
    )
    ap.add_argument(
        "--add_bos_eos",
        action="store_true",
        help="Add a post-processor that inserts [BOS]/[EOS] around sequences",
    )

    # Seeding
    ap.add_argument(
        "--seed_builtin",
        action="store_true",
        default=True,
        help="Add built-in seed tokens for code/logs/JSON glue (default: on)",
    )
    ap.add_argument(
        "--no_seed_builtin",
        action="store_false",
        dest="seed_builtin",
        help="Disable built-in seed tokens",
    )
    ap.add_argument(
        "--seed_tokens",
        default=None,
        help="Path to newline-delimited extra seed tokens (optional). Lines starting with # are ignored.",
    )
    ap.add_argument(
        "--seed_unescape",
        action="store_true",
        default=True,
        help=r"Unescape \n \t \r \\ sequences in --seed_tokens file (default: on)",
    )
    ap.add_argument(
        "--no_seed_unescape",
        action="store_false",
        dest="seed_unescape",
        help="Disable unescaping for --seed_tokens file",
    )

    ap.add_argument(
        "--report_path",
        default=None,
        help="Write a JSON report here (default: OUT_DIR/tokenizer_report.json)",
    )
    ap.add_argument("--status_every", type=int, default=200000)
    ap.add_argument("--report_max_text_samples", type=int, default=5)
    ap.add_argument("--report_max_dropped_samples", type=int, default=20)
    ap.add_argument("--report_truncate_chars", type=int, default=400)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path) if args.report_path else (out_dir / "tokenizer_report.json")

    report: Dict[str, Any] = {
        "status": "started",
        "started_utc": _utc_iso(),
        "finished_utc": None,
        "elapsed_sec": None,
        "args": {
            "data_glob": args.data_glob,
            "out_dir": str(out_dir),
            "vocab_size": args.vocab_size,
            "min_frequency": args.min_frequency,
            "bpe_dropout": args.bpe_dropout,
            "max_replacement_frac": args.max_replacement_frac,
            "normalize": args.normalize,
            "add_prefix_space": bool(args.add_prefix_space),
            "add_bos_eos": bool(args.add_bos_eos),
            "seed_builtin": bool(args.seed_builtin),
            "seed_tokens": args.seed_tokens,
            "seed_unescape": bool(args.seed_unescape),
            "report_path": str(report_path),
            "status_every": args.status_every,
            "report_max_text_samples": args.report_max_text_samples,
            "report_max_dropped_samples": args.report_max_dropped_samples,
            "report_truncate_chars": args.report_truncate_chars,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "tokenizers_version": getattr(tokenizers_pkg, "__version__", None),
            "transformers_version": getattr(transformers_pkg, "__version__", None),
        },
        "input": {"files": [], "file_info": []},
        "corpus": {
            "totals": {
                "lines_total": 0,
                "lines_empty": 0,
                "json_errors": 0,
                "non_dict": 0,
                "no_text": 0,
                "open_errors": 0,
                "yielded": 0,
                "kinds": {},
            },
            "per_file": {},
            "text_samples": [],
            "dropped_samples": [],
        },
        "tokenizer": {},
        "warnings": [],
        "errors": [],
    }

    start = time.time()

    def write_report() -> None:
        report["finished_utc"] = _utc_iso()
        report["elapsed_sec"] = round(time.time() - start, 3)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_report = report_path.with_suffix(report_path.suffix + ".tmp")
        with open(tmp_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=False)
        os.replace(tmp_report, report_path)

    try:
        files = resolve_input_files(args.data_glob)
        report["input"]["files"] = files
        report["input"]["file_info"] = [_file_info(p) for p in files]
        if not files:
            raise SystemExit(f"No input files matched: {args.data_glob}")

        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

        tok = build_tokenizer(args.normalize, args.add_prefix_space, args.bpe_dropout)
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        corpus_iter = CorpusIterator(
            files,
            report,
            normalize=args.normalize,
            max_replacement_frac=args.max_replacement_frac,
            max_text_samples=args.report_max_text_samples,
            max_dropped_samples=args.report_max_dropped_samples,
            truncate_chars=args.report_truncate_chars,
            status_every=args.status_every,
        )

        tok.train_from_iterator(corpus_iter, trainer=trainer)

        # Add seed tokens AFTER training as normal AddedTokens (NOT special tokens).
        # Note: this will increase vocab size beyond args.vocab_size; that’s expected.
        seed_tokens: List[str] = []
        if args.seed_builtin:
            seed_tokens.extend(DEFAULT_SEED_TOKENS)
        if args.seed_tokens:
            seed_tokens.extend(_load_seed_tokens(args.seed_tokens, unescape=args.seed_unescape))
        seed_tokens = _stable_unique(seed_tokens)

        if seed_tokens:
            added = [AddedToken(t, normalized=False, special=False) for t in seed_tokens]
            tok.add_tokens(added)

        if args.add_bos_eos:
            bos_id = tok.token_to_id("[BOS]")
            eos_id = tok.token_to_id("[EOS]")
            if bos_id is None or eos_id is None:
                report["warnings"].append(
                    {"type": "missing_special_token_ids", "bos_id": bos_id, "eos_id": eos_id}
                )
            else:
                tok.post_processor = TemplateProcessing(
                    single="[BOS] $A [EOS]",
                    pair="[BOS] $A [EOS] $B:1 [EOS]:1",
                    special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
                )

        hf_tok = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
        )
        hf_tok.clean_up_tokenization_spaces = False
        hf_tok.save_pretrained(out_dir)

        saved_files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
        vocab = hf_tok.get_vocab()

        report["tokenizer"] = {
            "type": "bytelevel_bpe",
            "normalize": args.normalize,
            "add_prefix_space": bool(args.add_prefix_space),
            "add_bos_eos": bool(args.add_bos_eos),
            "vocab_size_target": args.vocab_size,
            "vocab_size_actual": len(hf_tok),
            "min_frequency": args.min_frequency,
            "bpe_dropout": args.bpe_dropout,
            "max_replacement_frac": args.max_replacement_frac,
            "special_tokens": special_tokens,
            "special_token_ids": {
                "pad": hf_tok.pad_token_id,
                "unk": hf_tok.unk_token_id,
                "bos": hf_tok.bos_token_id,
                "eos": hf_tok.eos_token_id,
            },
            "special_tokens_in_vocab": {t: (t in vocab) for t in special_tokens},
            "seed_tokens_builtin": bool(args.seed_builtin),
            "seed_tokens_path": args.seed_tokens,
            "seed_tokens_count": len(seed_tokens),
            "seed_tokens_preview": seed_tokens[:40],
            "saved_files": saved_files,
            "sanity_samples": [],
        }

        sample_texts = [
            "hello world",
            "foo\nbar",
            "naïve café",
            "x = 1_000_000 # comment",
            "{" + '"k"' + ": " + '"v"' + "}",
            "Traceback (most recent call last):\n  File \"x.py\", line 1\n    raise ValueError(\"bad\")",
            "GET /api/v1/items?id=123 HTTP/1.1\nHost: example.com\n",
            "if (a != b && c <= d) { return a->x::y; }",
        ]
        for s in sample_texts:
            enc = hf_tok(s)
            ids = enc.get("input_ids")
            toks = hf_tok.convert_ids_to_tokens(ids) if isinstance(ids, list) else None
            decoded = hf_tok.decode(ids) if isinstance(ids, list) else None
            report["tokenizer"]["sanity_samples"].append(
                {"text": s, "input_ids": ids, "tokens": toks, "decoded": decoded}
            )

        report["status"] = "ok"
        write_report()

        print(f"Saved tokenizer to: {out_dir}")
        print(f"len(tokenizer) = {len(hf_tok)}")
        print(
            f"PAD={hf_tok.pad_token_id} UNK={hf_tok.unk_token_id} "
            f"BOS={hf_tok.bos_token_id} EOS={hf_tok.eos_token_id}"
        )
        print(f"Seed tokens added: {len(seed_tokens)} (builtin={args.seed_builtin}, file={bool(args.seed_tokens)})")
        print(f"Wrote report to: {report_path}")

    except SystemExit as e:
        report["status"] = "error"
        report["errors"].append({"type": "SystemExit", "message": str(e)})
        write_report()
        raise
    except Exception as e:
        report["status"] = "error"
        report["errors"].append(
            {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        )
        write_report()
        raise


if __name__ == "__main__":
    main()
