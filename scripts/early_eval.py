#!/usr/bin/env python3
"""
Early eval harness for near-scratch Onyx models.

Focus: training health, learning onset, pipeline correctness, and early copy/format behavior.
Writes a JSON report per checkpoint and prints a short console summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from _bootstrap import add_repo_root

add_repo_root()

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from onyx.inference import load_model, generate_stream


_LEADING_ASSISTANT_RE = re.compile(r"^\s*(assistant\s*:+\s*)+", flags=re.IGNORECASE)
_STEP_RE = re.compile(r"checkpoint_(final_)?step_(\d+)\.pt$")


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: str
    tokenizer: str
    model_config: Optional[str]
    val_data: str
    out_dir: Path
    copy_tests: Path
    gen_prompts: Path
    max_seq_len: int
    batch_size: int
    val_max_tokens: int
    val_max_docs: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    device: str
    dtype: str
    seed: int
    train_loss: Optional[float]
    slope_window: int
    divergence_threshold: float
    repeat_3gram_threshold: float
    unique_token_threshold: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _strip_leading_assistant(text: str) -> str:
    return _LEADING_ASSISTANT_RE.sub("", text)


def _normalize_roundtrip(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_checkpoint_step(path: str) -> Optional[int]:
    name = Path(path).name
    m = _STEP_RE.search(name)
    if not m:
        return None
    return int(m.group(2))


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


_ROLE_PREFIX_RE = re.compile(r"^\s*(system|user|assistant)\s*:\s*", flags=re.IGNORECASE)


def _strip_leading_role_prefixes(text: str) -> str:
    s = text
    for _ in range(4):
        m = _ROLE_PREFIX_RE.match(s)
        if not m:
            break
        s = s[m.end() :]
    return s.lstrip()


def _iter_val_docs(path: Path) -> Iterator[object]:
    for obj in _iter_jsonl(path):
        text = obj.get("text") or obj.get("content") or obj.get("input") or ""
        if isinstance(text, str) and text.strip():
            yield text.strip()
            continue
        user = obj.get("user")
        assistant = obj.get("assistant")
        system = obj.get("system")
        if isinstance(user, str) and isinstance(assistant, str):
            yield {
                "system": system if isinstance(system, str) else "",
                "user": user,
                "assistant": assistant,
            }


def _tokenize_doc(doc: object, *, tokenizer, eos_token_id: int) -> Optional[Tuple[List[int], List[int]]]:
    if isinstance(doc, str):
        toks = tokenizer.encode(doc, add_special_tokens=False)
        if not toks:
            return None
        toks.append(eos_token_id)
        return toks, list(toks)

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

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        assistant_ids = tokenizer.encode(assistant, add_special_tokens=False)
        if not prompt_ids or not assistant_ids:
            return None

        toks = prompt_ids + assistant_ids + [eos_token_id]
        labels = ([-100] * len(prompt_ids)) + assistant_ids + [eos_token_id]
        return toks, labels

    return None


def _chunk_tokens(
    tokens: List[int],
    labels: List[int],
    *,
    max_seq_len: int,
    pad_token_id: int,
) -> Iterator[Tuple[List[int], List[int]]]:
    if len(tokens) != len(labels):
        return
    start = 0
    while start < len(tokens):
        end = min(start + max_seq_len, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_labels = labels[start:end]
        if len(chunk_tokens) < max_seq_len:
            pad = max_seq_len - len(chunk_tokens)
            chunk_tokens = chunk_tokens + [pad_token_id] * pad
            chunk_labels = chunk_labels + ([-100] * pad)
        yield chunk_tokens, chunk_labels
        start = end


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _edit_similarity(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    dist = _levenshtein(a, b)
    return max(0.0, 1.0 - (dist / float(denom)))


def _repeat_ngram_rate(tokens: List[int], n: int) -> float:
    if n <= 0:
        return 0.0
    total = max(0, len(tokens) - n + 1)
    if total == 0:
        return 0.0
    counts: Dict[Tuple[int, ...], int] = {}
    for i in range(total):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / float(total)


def _max_run_length(tokens: List[int]) -> int:
    if not tokens:
        return 0
    max_run = 1
    cur_run = 1
    prev = tokens[0]
    for tid in tokens[1:]:
        if tid == prev:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            prev = tid
            cur_run = 1
    return max_run


def _load_jsonl_items(path: Path) -> List[dict]:
    return list(_iter_jsonl(path))


def _generate_text(
    *,
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Tuple[str, List[int]]:
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    eos = tokenizer.eos_token_id
    stop_tokens = [eos] if eos is not None else []
    out_token_ids: List[int] = []

    gen = generate_stream(
        model=model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos,
        stop_tokens=stop_tokens,
        min_tokens_before_eos=1,
        stop_on_eos=True,
    )
    for token, _ in gen:
        tid = int(token.item())
        if eos is not None and tid == eos:
            break
        out_token_ids.append(tid)

    text = tokenizer.decode(out_token_ids, skip_special_tokens=True)
    text = _strip_leading_assistant(text).lstrip("\n ")
    return text, out_token_ids


def _compute_sanity(model, tokenizer) -> Dict[str, Any]:
    tok_vocab = len(tokenizer)
    model_vocab = int(getattr(model.config, "vocab_size", 0) or 0)
    embed_vocab = int(model.embed.weight.shape[0]) if hasattr(model, "embed") else 0
    vocab_matches = (tok_vocab == embed_vocab) and (model_vocab == 0 or model_vocab == tok_vocab)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id
    unk_id = tokenizer.unk_token_id
    ids = [i for i in (pad_id, eos_id, bos_id, unk_id) if i is not None]
    ids_in_range = all(0 <= int(i) < tok_vocab for i in ids) if tok_vocab > 0 else False
    has_eos_or_pad = (eos_id is not None) or (pad_id is not None)
    special_tokens_ok = bool(ids_in_range and has_eos_or_pad)

    roundtrip_ok = True
    for s in ("Hello world.", "User: hi\nAssistant:", "(() )[]{}"):
        enc = tokenizer.encode(s, add_special_tokens=False)
        dec = tokenizer.decode(enc, skip_special_tokens=True)
        if _normalize_roundtrip(s) != _normalize_roundtrip(dec):
            roundtrip_ok = False
            break

    broken_pipeline = not (vocab_matches and special_tokens_ok and roundtrip_ok)
    return {
        "vocab_matches": vocab_matches,
        "special_tokens_ok": special_tokens_ok,
        "roundtrip_ok": roundtrip_ok,
        "broken_pipeline": broken_pipeline,
    }


def _evaluate_loss(
    *,
    model,
    tokenizer,
    val_path: Path,
    max_seq_len: int,
    batch_size: int,
    val_max_tokens: int,
    val_max_docs: int,
) -> Tuple[Optional[float], int]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id
    if eos is None:
        eos = 0
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    loss_sum = 0.0
    token_count = 0
    batch_tokens: List[List[int]] = []
    batch_labels: List[List[int]] = []

    def _flush_batch():
        nonlocal loss_sum, token_count, batch_tokens, batch_labels
        if not batch_tokens:
            return
        input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
        with torch.no_grad():
            memory_states = model.init_memory_states(input_ids.shape[0], device, dtype)
            out = model(
                input_ids,
                labels=None,
                memory_states=memory_states,
                update_memories=True,
                inference_mode=False,
            )
            logits = out["logits"][:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
        loss_sum += float(loss.item())
        token_count += int((shift_labels != -100).sum().item())
        batch_tokens = []
        batch_labels = []

    docs_seen = 0
    for doc in _iter_val_docs(val_path):
        if val_max_docs and docs_seen >= val_max_docs:
            break
        docs_seen += 1
        tok_pair = _tokenize_doc(doc, tokenizer=tokenizer, eos_token_id=eos)
        if tok_pair is None:
            continue
        toks, labels = tok_pair
        for chunk_toks, chunk_labels in _chunk_tokens(
            toks, labels, max_seq_len=max_seq_len, pad_token_id=pad
        ):
            batch_tokens.append(chunk_toks)
            batch_labels.append(chunk_labels)
            if len(batch_tokens) >= batch_size:
                _flush_batch()
            if val_max_tokens and token_count >= val_max_tokens:
                break
        if val_max_tokens and token_count >= val_max_tokens:
            break
    _flush_batch()

    if token_count == 0:
        return None, 0
    return loss_sum / float(token_count), token_count


def _evaluate_copy_tests(
    *,
    model,
    tokenizer,
    device: torch.device,
    items: List[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Dict[str, Any]:
    results = []
    exact = 0
    edit_sims = []
    json_ok = 0
    json_total = 0

    for item in items:
        prompt = str(item.get("prompt") or "")
        target = str(item.get("target") or "")
        kind = str(item.get("kind") or "")
        output, _ = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        out_norm = output.strip()
        tgt_norm = target.strip()
        is_exact = out_norm == tgt_norm
        if is_exact:
            exact += 1
        sim = _edit_similarity(out_norm, tgt_norm)
        edit_sims.append(sim)

        json_success = None
        if kind == "json":
            json_total += 1
            try:
                json.loads(out_norm)
                json_ok += 1
                json_success = True
            except Exception:
                json_success = False

        results.append(
            {
                "id": item.get("id"),
                "prompt": prompt,
                "target": target,
                "output": output,
                "exact": bool(is_exact),
                "edit_similarity": sim,
                "json_ok": json_success,
            }
        )

    total = len(results)
    exact_rate = (exact / total) if total else 0.0
    avg_edit = (sum(edit_sims) / total) if total else 0.0
    json_rate = (json_ok / json_total) if json_total else 0.0

    return {
        "exact_match_rate": exact_rate,
        "avg_edit_similarity": avg_edit,
        "json_parse_success_rate": json_rate,
        "by_item": results,
    }


def _evaluate_generation_metrics(
    *,
    model,
    tokenizer,
    device: torch.device,
    items: List[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Tuple[Dict[str, Any], List[dict]]:
    all_tokens: List[int] = []
    samples: List[dict] = []
    for item in items:
        prompt = str(item.get("prompt") or "")
        output, token_ids = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        samples.append({"prompt": prompt, "output": output})
        all_tokens.extend(token_ids)

    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens)) if total_tokens else 0
    unique_ratio = (unique_tokens / total_tokens) if total_tokens else 0.0
    repeat_1 = _repeat_ngram_rate(all_tokens, 1)
    repeat_3 = _repeat_ngram_rate(all_tokens, 3)
    max_run = _max_run_length(all_tokens)

    metrics = {
        "unique_token_ratio": unique_ratio,
        "repeat_1gram_rate": repeat_1,
        "repeat_3gram_rate": repeat_3,
        "max_run_length": max_run,
        "total_tokens": total_tokens,
        "num_prompts": len(items),
    }
    return metrics, samples


def _load_history(out_dir: Path, limit: int) -> List[dict]:
    if not out_dir.exists():
        return []
    reports: List[dict] = []
    for p in sorted(out_dir.glob("early_eval_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "loss" in data:
            reports.append(data)
    if limit > 0:
        return reports[-limit:]
    return reports


def _loss_slope(history: List[dict]) -> Optional[float]:
    points = []
    for h in history:
        loss = h.get("loss", {}).get("val_loss")
        step = h.get("checkpoint_step")
        if isinstance(loss, (int, float)):
            x = step if isinstance(step, int) else len(points)
            points.append((float(x), float(loss)))
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in points)
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def _compute_alerts(
    *,
    current: dict,
    prev: Optional[dict],
    divergence_threshold: float,
    repeat_3gram_threshold: float,
    unique_token_threshold: float,
) -> List[str]:
    alerts = []
    loss = current.get("loss", {}).get("val_loss")
    if isinstance(loss, (int, float)):
        if math.isnan(loss) or math.isinf(loss):
            alerts.append("ALERT_DIVERGENCE")
        if prev and isinstance(prev.get("loss", {}).get("val_loss"), (int, float)):
            prev_loss = prev["loss"]["val_loss"]
            if prev_loss > 0 and ((loss - prev_loss) / prev_loss) > divergence_threshold:
                alerts.append("ALERT_DIVERGENCE")

    gen = current.get("generation_metrics", {})
    repeat_3 = gen.get("repeat_3gram_rate")
    unique_ratio = gen.get("unique_token_ratio")
    if isinstance(repeat_3, (int, float)) and repeat_3 > repeat_3gram_threshold:
        alerts.append("ALERT_LOOPING")
    if isinstance(unique_ratio, (int, float)) and unique_ratio < unique_token_threshold:
        alerts.append("ALERT_COLLAPSE")

    sanity = current.get("sanity", {})
    if sanity.get("broken_pipeline"):
        alerts.append("ALERT_TOKENIZER_MISMATCH")

    return alerts


def run(cfg: EvalConfig) -> dict:
    if not TRANSFORMERS_AVAILABLE:
        raise SystemExit("Error: transformers not installed (pip install transformers)")

    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, trust_remote_code=True)
    model, _ = load_model(
        cfg.checkpoint,
        tokenizer,
        device=device,
        dtype=dtype,
        model_config_path=cfg.model_config,
    )
    model.eval()

    sanity = _compute_sanity(model, tokenizer)

    val_loss, val_tokens = _evaluate_loss(
        model=model,
        tokenizer=tokenizer,
        val_path=Path(cfg.val_data),
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        val_max_tokens=cfg.val_max_tokens,
        val_max_docs=cfg.val_max_docs,
    )
    val_ppl = math.exp(val_loss) if isinstance(val_loss, (int, float)) else None

    copy_items = _load_jsonl_items(cfg.copy_tests)
    copy_metrics = _evaluate_copy_tests(
        model=model,
        tokenizer=tokenizer,
        device=device,
        items=copy_items,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
    )

    gen_items = _load_jsonl_items(cfg.gen_prompts)
    gen_metrics, samples = _evaluate_generation_metrics(
        model=model,
        tokenizer=tokenizer,
        device=device,
        items=gen_items,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
    )

    checkpoint_step = _parse_checkpoint_step(cfg.checkpoint)
    report = {
        "checkpoint": cfg.checkpoint,
        "checkpoint_step": checkpoint_step,
        "timestamp": _now_iso(),
        "loss": {
            "train_loss": cfg.train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "val_tokens": val_tokens,
        },
        "copy_tests": copy_metrics,
        "generation_metrics": gen_metrics,
        "sanity": sanity,
        "samples": samples,
    }

    history = _load_history(cfg.out_dir, max(cfg.slope_window - 1, 0))
    history_with_current = history + [report]
    slope = _loss_slope(history_with_current)
    report["loss"]["val_loss_slope"] = slope
    report["loss"]["val_loss_slope_window"] = cfg.slope_window

    prev = history[-1] if history else None
    alerts = _compute_alerts(
        current=report,
        prev=prev,
        divergence_threshold=cfg.divergence_threshold,
        repeat_3gram_threshold=cfg.repeat_3gram_threshold,
        unique_token_threshold=cfg.unique_token_threshold,
    )
    report["alerts"] = alerts

    return report


def _print_summary(report: dict, prev: Optional[dict]) -> None:
    loss = report.get("loss", {}).get("val_loss")
    copy_rate = report.get("copy_tests", {}).get("exact_match_rate")
    repeat_3 = report.get("generation_metrics", {}).get("repeat_3gram_rate")
    uniq = report.get("generation_metrics", {}).get("unique_token_ratio")

    delta = None
    if prev and isinstance(prev.get("loss", {}).get("val_loss"), (int, float)) and isinstance(loss, (int, float)):
        delta = loss - prev["loss"]["val_loss"]

    parts = []
    if isinstance(loss, (int, float)):
        if delta is None:
            parts.append(f"val_loss={loss:.4f}")
        else:
            parts.append(f"val_loss={loss:.4f} (delta {delta:+.4f})")
    if isinstance(copy_rate, (int, float)):
        parts.append(f"copy_exact={copy_rate:.3f}")
    if isinstance(repeat_3, (int, float)):
        parts.append(f"repeat_3gram={repeat_3:.3f}")
    if isinstance(uniq, (int, float)):
        parts.append(f"unique_ratio={uniq:.3f}")
    alert_str = ",".join(report.get("alerts") or [])
    if alert_str:
        parts.append(f"alerts={alert_str}")
    print(" | ".join(parts))


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Run early eval metrics for Onyx checkpoints")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    p.add_argument("--tokenizer", required=True, help="Tokenizer name or local path")
    p.add_argument("--model_config", default=None, help="Optional model config JSON path")
    p.add_argument("--val_data", required=True, help="Frozen validation JSONL")
    repo_root = Path(__file__).resolve().parents[1]
    p.add_argument("--out_dir", default=str(repo_root / "early_eval_reports"))
    p.add_argument("--copy_tests", default=str(repo_root / "early_eval" / "copy_tests.jsonl"))
    p.add_argument("--gen_prompts", default=str(repo_root / "early_eval" / "gen_prompts.jsonl"))
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--val_max_tokens", type=int, default=20000)
    p.add_argument("--val_max_docs", type=int, default=0, help="0 = unlimited")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--train_loss", type=float, default=None)
    p.add_argument("--slope_window", type=int, default=5)

    p.add_argument("--divergence_threshold", type=float, default=0.10)
    p.add_argument("--repeat_3gram_threshold", type=float, default=0.50)
    p.add_argument("--unique_token_threshold", type=float, default=0.05)

    args = p.parse_args(argv)

    cfg = EvalConfig(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        model_config=args.model_config,
        val_data=args.val_data,
        out_dir=Path(args.out_dir),
        copy_tests=Path(args.copy_tests),
        gen_prompts=Path(args.gen_prompts),
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        val_max_tokens=args.val_max_tokens,
        val_max_docs=args.val_max_docs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        train_loss=args.train_loss,
        slope_window=args.slope_window,
        divergence_threshold=args.divergence_threshold,
        repeat_3gram_threshold=args.repeat_3gram_threshold,
        unique_token_threshold=args.unique_token_threshold,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    report = run(cfg)
    history = _load_history(cfg.out_dir, 1)
    prev = history[-1] if history else None
    _print_summary(report, prev)

    ckpt_name = Path(cfg.checkpoint).stem
    ts_compact = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = cfg.out_dir / f"early_eval_{ckpt_name}_{ts_compact}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote_report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
