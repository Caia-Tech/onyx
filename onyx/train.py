#!/usr/bin/env python3
"""
Caia Tech

"""

import argparse
import atexit
import gc
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
import resource
from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Optional, Iterator, List, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from onyx.model import (
    Onyx,
    OnyxConfig,
    M3Optimizer,
    get_param_groups,
)
from onyx.experimental import OnyxMHC
from onyx.fp32_utils import enforce_fp32_everywhere

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


class _SimpleLogger:
    def __init__(self, path: str, tee: bool, flush_logs: bool):
        self.path = path
        self.tee = tee
        self.flush_logs = flush_logs
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, msg: str) -> None:
        self._file.write(msg)
        if self.flush_logs:
            self._file.flush()
        if self.tee:
            self._stdout.write(msg)
            if self.flush_logs:
                self._stdout.flush()

    def close(self) -> None:
        try:
            if self.flush_logs:
                self._file.flush()
            self._file.close()
        except Exception:
            pass


_LOGGER: Optional[_SimpleLogger] = None


def _configure_logger(log_file: str, tee: bool, flush_logs: bool) -> None:
    global _LOGGER
    if not log_file:
        _LOGGER = None
        return
    if _LOGGER is not None:
        return
    _LOGGER = _SimpleLogger(log_file, tee=tee, flush_logs=flush_logs)
    atexit.register(_LOGGER.close)


def log(msg: str = "", *, end: str = "\n") -> None:
    if _LOGGER is None:
        print(msg, end=end)
        return
    _LOGGER.write(f"{msg}{end}")


def _get_rss_gb() -> float:
    if psutil is not None:
        try:
            rss_bytes = psutil.Process(os.getpid()).memory_info().rss
            return rss_bytes / (1024 ** 3)
        except Exception:
            pass
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss_bytes = rss
        else:
            rss_bytes = rss * 1024
        return rss_bytes / (1024 ** 3)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
        rss_kb = int(out.strip())
        return (rss_kb * 1024) / (1024 ** 3)
    except Exception:
        return 0.0


def _get_mps_mem_gb() -> Tuple[Optional[float], Optional[float]]:
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            current = torch.mps.current_allocated_memory()
        except Exception:
            current = None
        try:
            driver = torch.mps.driver_allocated_memory()
        except Exception:
            driver = None
        cur_gb = current / (1024 ** 3) if current is not None else None
        drv_gb = driver / (1024 ** 3) if driver is not None else None
        return cur_gb, drv_gb
    return None, None


# =============================================================================
# Configuration
# =============================================================================

class CMSFrequencyManager:
    """
    Manages the multi-timescale updates for CMS layers.
    Paper Section 7.1, Eq 71.

    This masks gradients for CMS levels that are not scheduled to update at the
    current optimizer step.
    """

    def __init__(self, model: nn.Module, config: OnyxConfig):
        self.param_schedules: List[Dict[str, Any]] = []
        if config is None:
            return

        base = int(getattr(config, "cms_update_every_base_steps", 1) or 1)
        mult = int(getattr(config, "cms_update_every_multiplier", 2) or 2)
        if base < 1:
            base = 1
        if mult < 1:
            mult = 1

        for name, param in model.named_parameters():
            if "ffn.level_ffns" not in name:
                continue
            parts = name.split(".")
            try:
                idx = parts.index("level_ffns") + 1
                level = int(parts[idx])
            except Exception:
                continue

            update_every = int(base * (mult ** level))
            update_every = max(1, update_every)
            self.param_schedules.append(
                {"param": param, "update_every": update_every, "name": name, "level": level}
            )

    def mask_gradients(self, global_step: int) -> None:
        """
        Call this before gradient clipping / optimizer.step().
        Zeroes out gradients for levels that shouldn't update this step.
        """
        step = int(global_step)
        for item in self.param_schedules:
            if step % item["update_every"] != 0:
                if item["param"].grad is not None:
                    item["param"].grad = None


@dataclass
class TrainingConfig:
    # === Data ===
    data_glob: str = "./data.jsonl"
    tokenizer_name: str = "NousResearch/Hermes-2-Pro-Llama-3-8B"

    # === Sequence ===
    max_seq_len: int = 256
    pack_sequences: bool = True
    drop_remainder: bool = False
    shuffle_buffer_docs: int = 2048  # buffer-shuffle docs without loading all into RAM

    # === Batch & Accumulation ===
    batch_size: int = 4
    tokens_per_step: int = 8192

    # === Training Duration ===
    num_epochs: int = 1
    train_tokens_target: Optional[int] = None
    max_steps: Optional[int] = None

    # === Learning Rate ===
    learning_rate: float = 3e-4
    min_lr: float = 1e-4
    memory_lr_scale: float = 0.1
    warmup_steps: int = 50
    warmup_ratio: Optional[float] = None
    weight_decay: float = 0.1

    # === Memory ===
    memory_reg_weight: float = 0.0001
    persist_memory_across_steps: bool = True
    reset_memory_every_steps: int = 0
    reset_memory_on_epoch: bool = True

    # === Loss & Regularization ===
    label_smoothing: float = 0.0
    entropy_reg_weight: float = 0.0
    feedback_strength: float = 1.0

    # === Monitoring ===
    monitor_diversity: bool = True
    monitor_memory_states: bool = False
    monitor_every: int = 0  # 0 = use log_every
    alert_top10_mass: float = 0.7
    alert_entropy_ratio: float = 0.3
    alert_effective_vocab: int = 100
    alert_memory_norm: float = 100.0

    # === Optimizer ===
    use_m3_optimizer: bool = True
    m3_beta_slow: float = 0.99
    m3_slow_freq: int = 50
    m3_slow_weight: float = 0.1

    # === Precision ===
    use_amp: bool = False
    amp_dtype: str = "float16"
    amp_flag: Optional[bool] = None

    # === Memory Savings ===
    gradient_checkpointing: bool = False
    # === Compilation ===
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"

    # === Experimental mHC ===
    experimental_mhc: bool = False
    mhc_n: int = 2
    mhc_sinkhorn: bool = True
    mhc_sinkhorn_iters: int = 10
    mhc_mode: str = "mhc"
    mhc_debug_finite_checks: bool = False
    mhc_debug_every: int = 0

    # === Gradient ===
    gradient_clip: float = 1.0

    # === Benchmarking ===
    bench_steps: int = 0
    disable_saves_during_bench: bool = False

    # === Checkpointing ===
    save_dir: str = "./checkpoints"
    save_every_epoch: bool = True
    save_every_steps: int = 0
    keep_last_n: int = 5
    keep_every_steps: int = 0  # if >0, keep step checkpoints at multiples of this forever
    resume: Optional[str] = None
    init_checkpoint: Optional[str] = None  # load weights only
    init_strict: bool = True  # if False, allow partial warm-start when architecture changes
    model_config_path: Optional[str] = None
    stage_name: str = ""

    # === Logging ===
    log_every: int = 50
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # === Debug ===
    dry_run: bool = False
    seed: int = 42

    log_file: str = ""
    tee: bool = False
    flush_logs: bool = True
    mem_report_every: int = 50
    max_rss_gb: float = 0.0
    max_mps_alloc_gb: float = 0.0
    auto_interrupt_on_mem: bool = True
    mps_empty_cache_every: int = 50
    gc_collect_every: int = 200
    dataset_state_mode: str = "light"
    quiet: bool = False
    mps_debug_memory: bool = False
    mps_debug_steps: int = 3


# =============================================================================
# Dataset
# =============================================================================

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
    if not isinstance(data, dict):
        return None

    chat = data.get("chat")
    if not isinstance(chat, list):
        chat = data.get("chats")
    if not isinstance(chat, list):
        chat = data.get("messages")
    if isinstance(chat, list) and chat:
        # Handle role/content chats: {"chat": [{"role": "user", "content": "..."}, ...]}
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

        # Handle paired user/assistant chat turns: {"chat": [{"user": "...", "assistant": "..."}]}
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
            # If the record is chat-shaped, yield a structured doc so we can
            # mask loss on the prompt and train only on the assistant span.
            return {
                "system": system if isinstance(system, str) else "",
                "user": user,
                "assistant": assistant,
            }
        return None

    if isinstance(text, str) and text:
        return text
    return None


def _iter_jsonl_texts(filepath: str) -> Iterator[Union[str, Dict[str, Any]]]:
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


class StreamingPackedDataset(IterableDataset):
    """
    Streaming dataset that packs documents into fixed-length sequences.
    Uses buffer shuffle (bounded memory) rather than list(all_docs).
    """

    def __init__(
        self,
        data_glob: str,
        tokenizer,
        max_seq_len: int = 2048,
        pack: bool = True,
        seed: int = 42,
        drop_remainder: bool = False,
        emit_cu_seqlens: bool = True,
        shuffle_buffer_docs: int = 2048,
    ):
        self.data_glob = data_glob
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pack = pack
        self.seed = seed
        self.drop_remainder = drop_remainder
        self.emit_cu_seqlens = emit_cu_seqlens
        self.shuffle_buffer_docs = max(0, int(shuffle_buffer_docs))

        self.data_files = self._get_files()
        if not self.data_files:
            raise ValueError(f"No files found matching: {data_glob}")

        log(f"Found {len(self.data_files)} data file(s)")
        for f in self.data_files[:5]:
            log(f"  - {f}")

        self.eod_token_id = tokenizer.eos_token_id
        if self.eod_token_id is None:
            self.eod_token_id = tokenizer.pad_token_id or 0

        self._doc_rng = random.Random(self.seed)
        self._doc_state: Optional[Dict[str, Any]] = None
        self._pack_state: Optional[Dict[str, Any]] = None

    def _get_files(self) -> List[str]:
        from glob import glob
        g = self.data_glob
        if "*" not in g and "?" not in g:
            return [g] if os.path.exists(g) else []
        return sorted(glob(g))

    def _init_doc_state(self) -> Dict[str, Any]:
        self._doc_rng = random.Random(self.seed)
        files = list(self.data_files)
        self._doc_rng.shuffle(files)
        self._doc_state = {
            "files": files,
            "file_index": 0,
            "file_pos": 0,
            "buffer": [],
            "buffer_index": 0,
            "exhausted": False,
        }
        return self._doc_state

    def _reset_stream_state(self) -> None:
        self._doc_state = None
        self._pack_state = None

    def state_dict(self, mode: str = "full") -> Dict[str, Any]:
        ds_state = self._doc_state or {}
        pack_state = self._pack_state or {}
        if mode not in ("full", "light"):
            raise ValueError(f"Unknown dataset_state_mode: {mode}")
        buffer = list(ds_state.get("buffer", [])) if mode == "full" else []
        return {
            "version": 1,
            "data_glob": self.data_glob,
            "data_files": list(self.data_files),
            "pack": self.pack,
            "max_seq_len": self.max_seq_len,
            "drop_remainder": self.drop_remainder,
            "shuffle_buffer_docs": self.shuffle_buffer_docs,
            "seed": self.seed,
            "state_mode": mode,
            "doc_state": {
                "files": list(ds_state.get("files", [])),
                "file_index": int(ds_state.get("file_index", 0)),
                "file_pos": ds_state.get("file_pos", 0),
                "buffer": buffer,
                "buffer_index": int(ds_state.get("buffer_index", 0)) if mode == "full" else 0,
                "exhausted": bool(ds_state.get("exhausted", False)),
            } if ds_state else None,
            "pack_state": {
                "current_seq": list(pack_state.get("current_seq", [])),
                "current_labels": list(pack_state.get("current_labels", [])),
                "current_boundaries": list(pack_state.get("current_boundaries", [])),
            } if pack_state else None,
            "rng_state": self._doc_rng.getstate() if self._doc_rng is not None else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        if state.get("pack") != self.pack or state.get("max_seq_len") != self.max_seq_len:
            log("[WARN] Dataset state mismatch (pack/max_seq_len); ignoring resume state.")
            return
        if state.get("drop_remainder") != self.drop_remainder or state.get("shuffle_buffer_docs") != self.shuffle_buffer_docs:
            log("[WARN] Dataset state mismatch (drop_remainder/shuffle_buffer_docs); ignoring resume state.")
            return
        if state.get("data_files") and list(state.get("data_files")) != list(self.data_files):
            log("[WARN] Dataset files differ from checkpoint; ignoring resume state.")
            return

        self._doc_state = state.get("doc_state") or None
        self._pack_state = state.get("pack_state") or None
        self._doc_rng = random.Random(self.seed)
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._doc_rng.setstate(rng_state)
        if self._doc_state is not None:
            if "buffer" not in self._doc_state:
                self._doc_state["buffer"] = []
                self._doc_state["buffer_index"] = 0

        if self._doc_state:
            buf_len = len(self._doc_state.get("buffer", []))
            seq_len = len((self._pack_state or {}).get("current_seq", []))
            log(f"[Resume] Dataset state loaded: file_index={self._doc_state.get('file_index', 0)}, buffer={buf_len}, seq={seq_len}")

    def _iter_documents(self) -> Iterator[Union[str, Dict[str, Any]]]:
        state = self._doc_state
        if state is None or state.get("exhausted"):
            state = self._init_doc_state()

        rng = self._doc_rng
        file_handle = None

        def _open_current_file() -> bool:
            nonlocal file_handle
            if file_handle is not None:
                file_handle.close()
            if state["file_index"] >= len(state["files"]):
                file_handle = None
                return False
            path = state["files"][state["file_index"]]
            file_handle = open(path, "r", encoding="utf-8")
            pos = state.get("file_pos", 0) or 0
            if pos:
                file_handle.seek(pos)
            return True

        def _read_next_doc() -> Optional[Union[str, Dict[str, Any]]]:
            nonlocal file_handle
            while True:
                if file_handle is None:
                    if not _open_current_file():
                        return None
                line = file_handle.readline()
                if not line:
                    file_handle.close()
                    file_handle = None
                    state["file_index"] += 1
                    state["file_pos"] = 0
                    continue
                state["file_pos"] = file_handle.tell()
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc = _jsonl_record_to_doc(data)
                if doc is None:
                    continue
                return doc

        try:
            while True:
                if self.shuffle_buffer_docs <= 1:
                    doc = _read_next_doc()
                    if doc is None:
                        state["exhausted"] = True
                        break
                    yield doc
                    continue

                if state["buffer_index"] < len(state["buffer"]):
                    doc = state["buffer"][state["buffer_index"]]
                    state["buffer_index"] += 1
                    if state["buffer_index"] >= len(state["buffer"]):
                        state["buffer"] = []
                        state["buffer_index"] = 0
                    yield doc
                    continue

                while len(state["buffer"]) < self.shuffle_buffer_docs:
                    doc = _read_next_doc()
                    if doc is None:
                        break
                    state["buffer"].append(doc)

                if state["buffer"]:
                    rng.shuffle(state["buffer"])
                    doc = state["buffer"][0]
                    state["buffer_index"] = 1
                    if state["buffer_index"] >= len(state["buffer"]):
                        state["buffer"] = []
                        state["buffer_index"] = 0
                    yield doc
                    continue

                state["exhausted"] = True
                break
        finally:
            if file_handle is not None:
                file_handle.close()

    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _tokenize_doc(self, doc: object) -> Optional[Tuple[List[int], List[int]]]:
        """
        Returns (input_ids, labels) for a single document.

        If the doc is chat-shaped (system/user/assistant), mask the prompt span
        with -100 so loss is computed only on assistant tokens (SFT-style).
        """
        # Plain text doc: train on all tokens.
        if isinstance(doc, str):
            toks = self._tokenize(doc)
            if not toks:
                return None
            toks.append(self.eod_token_id)
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

            prompt_ids = self._tokenize(prompt)
            assistant_ids = self._tokenize(assistant)
            if not prompt_ids or not assistant_ids:
                return None

            toks = prompt_ids + assistant_ids + [self.eod_token_id]
            labels = ([-100] * len(prompt_ids)) + assistant_ids + [self.eod_token_id]
            return toks, labels

        if isinstance(doc, dict) and "chat" in doc and isinstance(doc["chat"], list):
            toks: List[int] = []
            labels: List[int] = []
            prev_style = ""
            first_turn = True

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

                prompt_ids = self._tokenize(prompt)
                assistant_ids = self._tokenize(assistant)
                if not prompt_ids or not assistant_ids:
                    continue

                toks.extend(prompt_ids)
                labels.extend([-100] * len(prompt_ids))
                toks.extend(assistant_ids)
                labels.extend(assistant_ids)

                prev_style = style
                first_turn = False

            if not toks:
                return None
            toks.append(self.eod_token_id)
            labels.append(self.eod_token_id)
            return toks, labels

        return None

    def _pack_sequences_with_boundaries(self) -> Iterator[Tuple[List[int], List[int], List[int], Optional[int]]]:
        """Pack documents into fixed-length sequences and track doc boundaries.

        The returned `boundaries` is a cu_seqlens-style list of start offsets for
        each packed document segment within the fixed-length chunk, always
        starting at 0 and ending at `max_seq_len`.
        """
        if self._pack_state is None:
            self._pack_state = {
                "current_seq": [],
                "current_labels": [],
                "current_boundaries": [0],
            }

        current_seq: List[int] = self._pack_state.get("current_seq", [])
        current_labels: List[int] = self._pack_state.get("current_labels", [])
        current_boundaries: List[int] = self._pack_state.get("current_boundaries", [0]) or [0]

        def _dedup_consecutive(xs: List[int]) -> List[int]:
            if not xs:
                return []
            out = [xs[0]]
            for v in xs[1:]:
                if v != out[-1]:
                    out.append(v)
            return out

        for doc in self._iter_documents():
            tok_pair = self._tokenize_doc(doc)
            if tok_pair is None:
                continue
            toks, labels = tok_pair

            doc_start_in_seq = len(current_seq)
            current_seq.extend(toks)
            current_labels.extend(labels)

            # Track where the new document begins (if not already at 0)
            if doc_start_in_seq != 0:
                if current_boundaries[-1] != doc_start_in_seq:
                    current_boundaries.append(doc_start_in_seq)

            while len(current_seq) >= self.max_seq_len:
                seq = current_seq[: self.max_seq_len]
                lab = current_labels[: self.max_seq_len]

                b = [v for v in current_boundaries if v < self.max_seq_len]
                if not b or b[0] != 0:
                    b.insert(0, 0)
                if b[-1] != self.max_seq_len:
                    b.append(self.max_seq_len)
                b = _dedup_consecutive(b)

                # Update state before yielding so resume continues at the next chunk.
                current_seq = current_seq[self.max_seq_len :]
                current_labels = current_labels[self.max_seq_len :]
                shifted = [v - self.max_seq_len for v in current_boundaries if v >= self.max_seq_len]
                if not shifted or shifted[0] != 0:
                    shifted.insert(0, 0)  # doc continuation starts at 0 in the new chunk
                current_boundaries = _dedup_consecutive(shifted)
                self._pack_state["current_seq"] = current_seq
                self._pack_state["current_labels"] = current_labels
                self._pack_state["current_boundaries"] = current_boundaries

                yield seq, lab, b, None

            self._pack_state["current_seq"] = current_seq
            self._pack_state["current_labels"] = current_labels
            self._pack_state["current_boundaries"] = current_boundaries

        if current_seq and not self.drop_remainder:
            pad_start = None
            if len(current_seq) < self.max_seq_len:
                pad_start = len(current_seq)
                pad = self.max_seq_len - len(current_seq)
                current_seq.extend([self.eod_token_id] * pad)
                current_labels.extend([-100] * pad)

            b = [v for v in current_boundaries if v < self.max_seq_len]
            if not b or b[0] != 0:
                b.insert(0, 0)
            if b[-1] != self.max_seq_len:
                b.append(self.max_seq_len)
            b = _dedup_consecutive(b)

            self._pack_state["current_seq"] = []
            self._pack_state["current_labels"] = []
            self._pack_state["current_boundaries"] = [0]

            yield current_seq[: self.max_seq_len], current_labels[: self.max_seq_len], b, pad_start

        if self._doc_state and self._doc_state.get("exhausted"):
            self._reset_stream_state()

    def _simple_sequences(self) -> Iterator[Tuple[List[int], List[int], Optional[int]]]:
        if self._pack_state is None:
            self._pack_state = {"current_seq": [], "current_labels": []}

        buf: List[int] = self._pack_state.get("current_seq", [])
        buf_labels: List[int] = self._pack_state.get("current_labels", [])
        for doc in self._iter_documents():
            tok_pair = self._tokenize_doc(doc)
            if tok_pair is None:
                continue
            toks, labels = tok_pair
            buf.extend(toks)
            buf_labels.extend(labels)

            while len(buf) >= self.max_seq_len:
                seq = buf[: self.max_seq_len]
                lab = buf_labels[: self.max_seq_len]
                buf = buf[self.max_seq_len :]
                buf_labels = buf_labels[self.max_seq_len :]
                self._pack_state["current_seq"] = buf
                self._pack_state["current_labels"] = buf_labels
                yield seq, lab, None
            self._pack_state["current_seq"] = buf
            self._pack_state["current_labels"] = buf_labels

        if buf and not self.drop_remainder:
            pad_start = None
            if len(buf) < self.max_seq_len:
                pad_start = len(buf)
                pad = self.max_seq_len - len(buf)
                buf.extend([self.eod_token_id] * pad)
                buf_labels.extend([-100] * pad)
            self._pack_state["current_seq"] = []
            self._pack_state["current_labels"] = []
            yield buf[: self.max_seq_len], buf_labels[: self.max_seq_len], pad_start

        if self._doc_state and self._doc_state.get("exhausted"):
            self._reset_stream_state()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._doc_state is None or self._doc_state.get("exhausted"):
            self._reset_stream_state()
            self._init_doc_state()
        if self.pack:
            for seq, lab, boundaries, pad_start in self._pack_sequences_with_boundaries():
                input_ids = torch.tensor(seq, dtype=torch.long)
                labels = torch.tensor(lab, dtype=torch.long)
                if pad_start is not None and pad_start < labels.numel():
                    labels[pad_start:] = -100  # safety

                sample: Dict[str, Any] = {"input_ids": input_ids, "labels": labels}
                if self.emit_cu_seqlens:
                    sample["cu_seqlens"] = torch.tensor(boundaries, dtype=torch.int32)
                    sample["max_seqlen"] = self.max_seq_len
                yield sample
        else:
            for seq, lab, pad_start in self._simple_sequences():
                input_ids = torch.tensor(seq, dtype=torch.long)
                labels = torch.tensor(lab, dtype=torch.long)
                if pad_start is not None:
                    labels[pad_start:] = -100  # safety
                yield {"input_ids": input_ids, "labels": labels}


def collate_onyx(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Stack fixed tensors; keep cu_seqlens ragged-safe (list of tensors).
    out: Dict[str, Any] = {}
    out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
    out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)

    if "cu_seqlens" in batch[0]:
        bnds = [b["cu_seqlens"] for b in batch]
        max_len = max(x.numel() for x in bnds)
        cu = torch.full((len(batch), max_len), -1, dtype=torch.int32)
        ns = torch.zeros((len(batch),), dtype=torch.int32)
        for i, x in enumerate(bnds):
            cu[i, : x.numel()] = x
            ns[i] = x.numel()
        out["cu_seqlens"] = cu
        out["num_segs"] = ns
        out["max_seqlen"] = torch.tensor(batch[0].get("max_seqlen", 0), dtype=torch.int32)
    return out


# =============================================================================
# LR
# =============================================================================

def get_lr(step: int, config: TrainingConfig, total_steps: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)

    decay_steps = max(1, total_steps - config.warmup_steps)
    progress = (step - config.warmup_steps) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    return config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        scale = pg.get("lr_scale", 1.0)
        pg["lr"] = lr * scale


def _tree_map_tensors(obj: Any, fn):
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _tree_map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tree_map_tensors(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_tree_map_tensors(v, fn) for v in obj)
    return obj


def _assert_fp32_everywhere(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        if param.is_floating_point() and param.dtype != torch.float32:
            raise RuntimeError(f"Non-fp32 parameter: {name} ({param.dtype})")

    for name, buf in module.named_buffers():
        if buf is None:
            continue
        if buf.is_floating_point() and buf.dtype != torch.float32:
            raise RuntimeError(f"Non-fp32 buffer: {name} ({buf.dtype})")


def _assert_fp32_memory_states(obj: Any, path: str = "memory_states") -> None:
    if torch.is_tensor(obj):
        if obj.is_floating_point() and obj.dtype != torch.float32:
            raise RuntimeError(f"Non-fp32 tensor at {path}: {obj.dtype}")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            _assert_fp32_memory_states(value, f"{path}.{key}")
        return
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            _assert_fp32_memory_states(value, f"{path}[{idx}]")
        return
    if isinstance(obj, tuple):
        for idx, value in enumerate(obj):
            _assert_fp32_memory_states(value, f"{path}[{idx}]")
        return


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    if tensor.numel() == 0:
        return {"mean": float("nan"), "max": float("nan"), "min": float("nan")}
    data = tensor.detach().float()
    finite = torch.isfinite(data)
    if finite.any():
        data = data[finite]
        return {
            "mean": float(data.mean().item()),
            "max": float(data.max().item()),
            "min": float(data.min().item()),
        }
    return {"mean": float("nan"), "max": float("nan"), "min": float("nan")}


def _tensor_finite_report(tensor: torch.Tensor) -> Dict[str, Any]:
    x = tensor.detach()
    nan = torch.isnan(x).sum().item()
    inf = torch.isinf(x).sum().item()
    total = x.numel()
    finite_mask = torch.isfinite(x)
    finite_count = finite_mask.sum().item()
    rep = {
        "nan": int(nan),
        "inf": int(inf),
        "finite": int(finite_count),
        "total": int(total),
    }
    if finite_count > 0:
        xf = x[finite_mask].float()
        rep.update(mean=float(xf.mean().item()), max=float(xf.max().item()), min=float(xf.min().item()))
    else:
        rep.update(mean=float("nan"), max=float("nan"), min=float("nan"))
    return rep


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self._interrupt_requested = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.resume_vocab_size: Optional[int] = None
        self.init_vocab_size: Optional[int] = None
        self._pending_rng_state: Optional[Dict[str, Any]] = None
        self._rng_restored_once = False
        self._prefetched_batch: Optional[Dict[str, Any]] = None
        self.last_memory_states: Optional[Any] = None
        self._resume_dataset_state_loaded = False
        self._resume_memory_states_loaded = False
        self._logger_configured = False
        self._checked_token_range = False
        self.skipped_steps = 0

        # Initialize monitors to None (will be set in setup() if enabled)
        self.diversity_monitor: Optional[Any] = None
        self.memory_monitor: Optional[Any] = None

    def _log(self, msg: str, *, force: bool = False) -> None:
        if not force and getattr(self.config, "quiet", False):
            return
        log(msg)

    def _handle_signal(self, signum, frame):
        self._log(f"\n[SIGNAL] Received {signum}, will save and exit after current step...", force=True)
        self._interrupt_requested = True

    def _has_nonfinite_grads(self) -> bool:
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                self._log(f"[WARN] Non-finite grad in {name}; skipping optimizer step.", force=True)
                return True
        return False

    def _nan_tripwire(
        self,
        where: str,
        *,
        loss: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        grad_norm: Optional[float] = None,
        memory_states: Optional[Any] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        if memory_states is not None:
            self.last_memory_states = memory_states

        lrs = []
        if hasattr(self, "optimizer"):
            lrs = [pg.get("lr") for pg in self.optimizer.param_groups]
        lr_str = ", ".join([f"{lr:.2e}" for lr in lrs if isinstance(lr, (float, int))]) if lrs else "n/a"
        grad_norm_str = f"{grad_norm:.4e}" if isinstance(grad_norm, (float, int)) else "n/a"

        self._log(
            f"[NaN] {where} non-finite detected | step={self.global_step} | lr={lr_str} | grad_norm={grad_norm_str}",
            force=True,
        )

        if loss is not None and torch.is_tensor(loss):
            loss_val = float(loss.detach().float().cpu().item())
            self._log(f"[NaN] loss={loss_val}", force=True)

        if logits is not None and torch.is_tensor(logits):
            rep = _tensor_finite_report(logits)
            self._log(
                "[NaN] logits report: "
                f"nan={rep['nan']} inf={rep['inf']} finite={rep['finite']}/{rep['total']} "
                f"mean={rep['mean']:.4e} max={rep['max']:.4e} min={rep['min']:.4e}",
                force=True,
            )

        if labels is not None and torch.is_tensor(labels):
            try:
                flat = labels[:, 1:].contiguous().view(-1)
                valid = flat != -100
                valid_count = int(valid.sum().item())
                if valid.any():
                    lab_min = int(flat[valid].min().item())
                    lab_max = int(flat[valid].max().item())
                else:
                    lab_min = -100
                    lab_max = -100
                self._log(
                    f"[NaN] labels: valid={valid_count} min_valid={lab_min} max_valid={lab_max}",
                    force=True,
                )
            except Exception as exc:
                self._log(f"[NaN] labels report failed: {exc}", force=True)

        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        mixers = getattr(model, "mixers", None)
        if mixers is not None:
            for idx, mixer in enumerate(mixers):
                matrix = getattr(mixer, "matrix", None)
                if matrix is None:
                    continue
                rep = _tensor_finite_report(matrix)
                msg = (
                    f"[NaN] mhc_mixer[{idx}] matrix report: "
                    f"nan={rep['nan']} inf={rep['inf']} finite={rep['finite']}/{rep['total']} "
                    f"mean={rep['mean']:.4e} max={rep['max']:.4e} min={rep['min']:.4e}"
                )
                try:
                    with torch.no_grad():
                        proj = mixer._mix_matrix()
                    proj_rep = _tensor_finite_report(proj)
                    msg += (
                        " | proj report: "
                        f"nan={proj_rep['nan']} inf={proj_rep['inf']} finite={proj_rep['finite']}/{proj_rep['total']} "
                        f"mean={proj_rep['mean']:.4e} max={proj_rep['max']:.4e} min={proj_rep['min']:.4e}"
                    )
                except Exception as exc:
                    msg += f" | proj_err={exc}"
                used = getattr(mixer, "last_P_used", None)
                if used is not None:
                    used_rep = _tensor_finite_report(used)
                    msg += (
                        " | used report: "
                        f"nan={used_rep['nan']} inf={used_rep['inf']} finite={used_rep['finite']}/{used_rep['total']} "
                        f"mean={used_rep['mean']:.4e} max={used_rep['max']:.4e} min={used_rep['min']:.4e}"
                    )
                else:
                    used_stats = getattr(mixer, "last_P_used_stats", None)
                    if used_stats is not None:
                        msg += (
                            " | used report: "
                            f"nan={used_stats['nan']} inf={used_stats['inf']} "
                            f"finite={used_stats['finite']}/{used_stats['total']} "
                            f"mean={used_stats['mean']:.4e} max={used_stats['max']:.4e} min={used_stats['min']:.4e}"
                        )
                self._log(msg, force=True)

        try:
            self.save_checkpoint(f"nan_step_{self.global_step}")
            self._log(f"[NaN] Saved checkpoint: nan_step_{self.global_step}", force=True)
        except Exception as exc:
            self._log(f"[NaN] Failed to save checkpoint: {exc}", force=True)

        raise RuntimeError(f"Non-finite detected at {where} (step {self.global_step}).")

    def _find_nonfinite_grads(self, max_items: int = 8) -> List[str]:
        bad: List[str] = []
        for name, p in self.model.named_parameters():
            g = p.grad
            if g is None:
                continue
            try:
                finite = bool(torch.isfinite(g).all().item())
            except Exception:
                finite = False
            if not finite:
                bad.append(name)
                if len(bad) >= max_items:
                    break
        return bad

    def _safe_clip_grad_norm(self, max_norm: float, eps: float = 1e-6) -> float:
        """
        Safer alternative to torch.nn.utils.clip_grad_norm_.

        On MPS, clip_grad_norm_ can sometimes produce NaN norms; this implementation:
        - Computes the norm robustly by accumulating scalar sums on CPU
        - Never scales gradients if the computed norm is non-finite
        """
        if max_norm <= 0:
            return 0.0

        total_sq = 0.0
        found = False
        for p in self.model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if g.numel() == 0:
                continue
            found = True
            if not torch.isfinite(g).all():
                return float("nan")
            gg = g.float()
            total_sq += float((gg * gg).sum().item())

        if not found:
            return 0.0

        total_norm = math.sqrt(total_sq)
        if not math.isfinite(total_norm) or total_norm == 0.0:
            return total_norm

        clip_coef = float(max_norm) / (total_norm + float(eps))
        if clip_coef >= 1.0:
            return total_norm

        for p in self.model.parameters():
            if p.grad is None:
                continue
            p.grad.mul_(clip_coef)
        return total_norm

    def _extract_vocab_size_from_ckpt(self, ckpt: Dict[str, Any]) -> Optional[int]:
        msd = ckpt.get("model_state_dict") or ckpt.get("model") or None
        if not isinstance(msd, dict):
            return None
        for k in ("embed.weight", "lm_head.weight"):
            if k in msd and hasattr(msd[k], "shape"):
                return int(msd[k].shape[0])
        return None

    def _resize_vocab_tensor(self, state_dict: Dict[str, torch.Tensor], key: str, target: torch.Tensor) -> None:
        if key not in state_dict:
            return
        t = state_dict[key]
        if t.shape == target.shape:
            return
        # This helper is only intended to handle vocab row-count changes. If the hidden
        # dimension (d_model) differs, the checkpoint is from a different-sized model.
        if (
            t.ndim == 2
            and target.ndim == 2
            and t.shape[0] == target.shape[0]
            and t.shape[1] != target.shape[1]
            and key in ("embed.weight", "lm_head.weight")
        ):
            raise ValueError(
                f"Cannot load {key}: checkpoint d_model={int(t.shape[1])} vs model d_model={int(target.shape[1])}. "
                "To warm-start a larger model, first grow the checkpoint with "
                "`python tools/grow_model_ckpt.py --in_ckpt <small.pt> --out_ckpt <grown.pt> "
                "--old_config configs/<small>.json --new_config configs/<big>.json` "
                "and use the grown checkpoint as --init_checkpoint."
            )
        if t.ndim != target.ndim or t.shape[1:] != target.shape[1:]:
            raise ValueError(f"Cannot resize {key}: ckpt {tuple(t.shape)} vs target {tuple(target.shape)}")
        new_t = torch.zeros_like(target)
        rows = min(t.shape[0], target.shape[0])
        new_t[:rows] = t[:rows]
        state_dict[key] = new_t
        self._log(f"Resized {key}: {tuple(t.shape)} -> {tuple(target.shape)}", force=True)

    def _get_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _set_rng_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        py_state = state.get("python")
        if py_state is not None:
            random.setstate(py_state)
        torch_state = state.get("torch")
        if torch_state is not None:
            if not isinstance(torch_state, torch.Tensor):
                torch_state = torch.tensor(torch_state, dtype=torch.uint8)
            if torch_state.dtype != torch.uint8 or torch_state.device.type != "cpu":
                torch_state = torch_state.to(device="cpu", dtype=torch.uint8)
            torch.set_rng_state(torch_state)
        if "cuda" in state:
            if torch.cuda.is_available():
                cuda_state = state["cuda"]
                if isinstance(cuda_state, (list, tuple)):
                    fixed_states = []
                    for s in cuda_state:
                        if not isinstance(s, torch.Tensor):
                            s = torch.tensor(s, dtype=torch.uint8)
                        if s.dtype != torch.uint8 or s.device.type != "cpu":
                            s = s.to(device="cpu", dtype=torch.uint8)
                        fixed_states.append(s)
                    torch.cuda.set_rng_state_all(fixed_states)
                else:
                    if not isinstance(cuda_state, torch.Tensor):
                        cuda_state = torch.tensor(cuda_state, dtype=torch.uint8)
                    if cuda_state.dtype != torch.uint8 or cuda_state.device.type != "cpu":
                        cuda_state = cuda_state.to(device="cpu", dtype=torch.uint8)
                    torch.cuda.set_rng_state(cuda_state)
            else:
                self._log("[WARN] Checkpoint has CUDA RNG state, but CUDA is not available.", force=True)

    def setup(self):
        cfg = self.config
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
            self._log(f"Using CUDA: {torch.cuda.get_device_name()}", force=True)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_type = "mps"
            self._log("Using MPS (Apple Silicon)", force=True)
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"
            self._log("Using CPU", force=True)
            cfg.use_amp = False

        if cfg.amp_dtype == "bfloat16":
            self.autocast_dtype = torch.bfloat16
        elif cfg.amp_dtype == "float16":
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = torch.float32

        # Default: no AMP on MPS/CPU unless explicitly requested
        if self.device_type in ("mps", "cpu") and cfg.amp_flag is None:
            cfg.use_amp = False

        self.use_autocast = cfg.use_amp and self.device_type in ("cuda", "mps")
        if cfg.use_amp:
            self._log(f"Autocast dtype: {self.autocast_dtype} | AMP: {cfg.use_amp}")
        else:
            self._log("Using fp32 (AMP disabled)")

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")

        self._log(f"Loading tokenizer: {cfg.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Peek vocab sizes from resume/init checkpoints so we never accidentally shrink
        resume_meta = None
        init_meta = None
        if cfg.resume:
            resume_meta = torch.load(cfg.resume, map_location="cpu", weights_only=False)
            self.resume_vocab_size = self._extract_vocab_size_from_ckpt(resume_meta)
        if cfg.init_checkpoint:
            init_meta = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=False)
            self.init_vocab_size = self._extract_vocab_size_from_ckpt(init_meta)

        # Load model config
        if cfg.model_config_path:
            self._log(f"Loading model config from: {cfg.model_config_path}")
            with open(cfg.model_config_path, "r") as f:
                json_cfg = json.load(f)
            arch = json_cfg.get("architecture", json_cfg)
            if not isinstance(arch, dict):
                raise ValueError("model_config_path must contain a JSON object or an 'architecture' dict.")
            arch = dict(arch)
            if "rope_theta" in arch and "rope_base" not in arch:
                # Back-compat: some configs use rope_theta.
                arch["rope_base"] = arch.pop("rope_theta")
            model_config = OnyxConfig(**arch)
        else:
            self._log("Creating Onyx model with default config...")
            model_config = OnyxConfig()

        # Runtime overrides (device/TrainingConfig-specific)
        model_config.max_seq_len = cfg.max_seq_len
        model_config.train_seq_len = cfg.max_seq_len
        model_config.memory_reg_weight = cfg.memory_reg_weight
        model_config.label_smoothing = cfg.label_smoothing
        model_config.entropy_reg_weight = cfg.entropy_reg_weight
        model_config.feedback_strength = cfg.feedback_strength
        model_config.use_flash_attention = (self.device_type == "cuda")
        model_config.gradient_checkpointing = cfg.gradient_checkpointing
        model_config.mhc_debug_finite_checks = cfg.mhc_debug_finite_checks
        model_config.mhc_debug_every = cfg.mhc_debug_every

        # Align vocab size (never shrink below any checkpoint vocab if provided)
        tok_vs = len(self.tokenizer)
        ckpt_vs = max([v for v in [self.resume_vocab_size, self.init_vocab_size] if v is not None], default=0)
        target_vs = max(tok_vs, ckpt_vs, model_config.vocab_size)
        if target_vs != model_config.vocab_size:
            self._log(f"Adjusting model vocab_size from {model_config.vocab_size} to {target_vs} to match tokenizer/checkpoint.")
            model_config.vocab_size = target_vs

        dtype_mode = "fp32 (AMP disabled)" if not cfg.use_amp else f"amp({self.autocast_dtype})"
        self._log(f"[Config] device_type={self.device_type} | dtype_mode={dtype_mode}")
        self._log(
            "[Config] d_model={d_model} n_layers={n_layers} n_heads={n_heads} "
            "n_kv_heads={n_kv_heads} d_ff={d_ff}".format(
                d_model=model_config.d_model,
                n_layers=model_config.n_layers,
                n_heads=model_config.n_heads,
                n_kv_heads=model_config.n_kv_heads,
                d_ff=model_config.d_ff,
            )
        )
        self._log(
            "[Toggles] tie_embeddings={tie} use_hope_attention={hope} "
            "use_cms_ffn={cms} cms_num_levels={cms_levels} "
            "use_short_conv={short_conv} memory_chunk_size={mem_chunk} "
            "gradient_checkpointing={grad_ckpt}".format(
                tie=model_config.tie_embeddings,
                hope=model_config.use_hope_attention,
                cms=model_config.use_cms_ffn,
                cms_levels=model_config.cms_num_levels,
                short_conv=model_config.use_short_conv,
                mem_chunk=model_config.memory_chunk_size,
                grad_ckpt=model_config.gradient_checkpointing,
            )
        )
        self._log(
            "[Vocab] tokenizer_len={tok} ckpt_vocab={ckpt} model_vocab={model}".format(
                tok=tok_vs,
                ckpt=ckpt_vs,
                model=model_config.vocab_size,
            )
        )
        self._log(
            "[Seq] cfg.max_seq_len={cfg_len} model.max_seq_len={model_len}".format(
                cfg_len=cfg.max_seq_len,
                model_len=model_config.max_seq_len,
            )
        )

        if cfg.experimental_mhc:
            self._log(
                "[Experimental] Using OnyxMHC mhc_n={n} mhc_mode={mode} sinkhorn={sinkhorn} iters={iters}".format(
                    n=cfg.mhc_n,
                    mode=cfg.mhc_mode,
                    sinkhorn=cfg.mhc_sinkhorn,
                    iters=cfg.mhc_sinkhorn_iters,
                )
            )
            self.model = OnyxMHC(
                model_config,
                mhc_n=cfg.mhc_n,
                mhc_mode=cfg.mhc_mode,
                mhc_sinkhorn=cfg.mhc_sinkhorn,
                mhc_sinkhorn_iters=cfg.mhc_sinkhorn_iters,
            ).to(self.device)
        else:
            self.model = Onyx(model_config).to(self.device)
        if not cfg.use_amp:
            enforce_fp32_everywhere(self.model)
        try:
            tied = self.model.embed.weight.data_ptr() == self.model.lm_head.weight.data_ptr()
        except Exception:
            tied = False
        self._log(
            "[Tied] embed.shape={embed_shape} lm_head.shape={lm_shape} tied={tied}".format(
                embed_shape=tuple(self.model.embed.weight.shape),
                lm_shape=tuple(self.model.lm_head.weight.shape),
                tied=tied,
            )
        )

        # Load weights from init_checkpoint (weights only)
        if cfg.init_checkpoint:
            self._log(f"Loading model weights from init_checkpoint: {cfg.init_checkpoint}")
            sd = init_meta.get("model_state_dict") if isinstance(init_meta, dict) and "model_state_dict" in init_meta else init_meta
            if not isinstance(sd, dict):
                raise ValueError("init_checkpoint did not contain a state_dict-like mapping")

            # Resize vocab-dependent tensors if needed
            if hasattr(self.model, "embed"):
                self._resize_vocab_tensor(sd, "embed.weight", self.model.embed.weight)
            if hasattr(self.model, "lm_head"):
                self._resize_vocab_tensor(sd, "lm_head.weight", self.model.lm_head.weight)

            incompatible = self.model.load_state_dict(sd, strict=cfg.init_strict)
            if hasattr(incompatible, "missing_keys") and hasattr(incompatible, "unexpected_keys"):
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    self._log(
                        "[Init] Incompatible keys: "
                        f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
                        force=True,
                    )
            self._log("Init weights loaded.")
            if not cfg.use_amp:
                enforce_fp32_everywhere(self.model)

        # Dataset + loader
        self._log(f"Loading dataset from: {cfg.data_glob}")
        self.dataset = StreamingPackedDataset(
            data_glob=cfg.data_glob,
            tokenizer=self.tokenizer,
            max_seq_len=cfg.max_seq_len,
            pack=cfg.pack_sequences,
            seed=cfg.seed,
            drop_remainder=cfg.drop_remainder,
            emit_cu_seqlens=True,
            shuffle_buffer_docs=cfg.shuffle_buffer_docs,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            num_workers=0,
            pin_memory=(self.device_type == "cuda"),
            collate_fn=collate_onyx,
        )
        self.data_iter = iter(self.dataloader)

        # Optimizer
        param_groups = get_param_groups(
            self.model,
            weight_decay=cfg.weight_decay,
            memory_lr_scale=cfg.memory_lr_scale,
        )

        if cfg.use_m3_optimizer:
            self._log("Using M3 optimizer")
            self.optimizer = M3Optimizer(
                param_groups,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                beta_slow=cfg.m3_beta_slow,
                slow_freq=cfg.m3_slow_freq,
                slow_weight=cfg.m3_slow_weight,
            )
        else:
            self._log("Using AdamW optimizer")
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=cfg.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=cfg.weight_decay,
            )

        tokens_per_batch = cfg.batch_size * cfg.max_seq_len
        self.accumulation_steps = max(1, cfg.tokens_per_step // max(1, tokens_per_batch))
        self._log(f"Gradient accumulation: {self.accumulation_steps} steps")
        self._log(f"Tokens per optimizer step: {self.accumulation_steps * tokens_per_batch:,}")

        self.scaler = None
        if cfg.use_amp and cfg.amp_dtype == "float16":
            if self.device_type == "cuda":
                self.scaler = torch.amp.GradScaler("cuda")
            elif self.device_type == "mps":
                try:
                    # MPS fp16 can overflow easily; start with a smaller scale than CUDA defaults.
                    self.scaler = torch.amp.GradScaler("mps", init_scale=2.0**10, growth_interval=2000)
                    self._log("Using GradScaler on MPS.")
                except Exception as e:
                    self._log(f"GradScaler not available on MPS: {e}", force=True)

        self.global_step = 0
        self.tokens_seen = 0
        self.best_loss = float("inf")
        self.current_epoch = 0

        os.makedirs(cfg.save_dir, exist_ok=True)

        # Resume full state if requested
        if cfg.resume:
            self.load_checkpoint(cfg.resume)

        if cfg.wandb_project and WANDB_AVAILABLE:
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config={**vars(cfg), "device": str(self.device)},
            )

        if not cfg.use_amp:
            _assert_fp32_everywhere(self.model)
            init_states = self.model.init_memory_states(cfg.batch_size, self.device, torch.float32)
            _assert_fp32_memory_states(init_states)
            if self._resume_memory_states_loaded and self.last_memory_states is not None:
                _assert_fp32_memory_states(self.last_memory_states)

        if cfg.use_torch_compile and self.device_type == "cuda" and hasattr(torch, "compile"):
            self._log(f"Compiling model with mode: {cfg.compile_mode}")
            self.model = torch.compile(self.model, mode=cfg.compile_mode)

        # Initialize monitoring
        monitor_freq = cfg.monitor_every if cfg.monitor_every > 0 else cfg.log_every
        if cfg.monitor_diversity:
            from onyx.monitoring import DiversityMonitor
            self.diversity_monitor = DiversityMonitor(
                vocab_size=model_config.vocab_size,
                monitor_every=monitor_freq,
                alert_top10_mass=cfg.alert_top10_mass,
                alert_entropy_ratio=cfg.alert_entropy_ratio,
                alert_effective_vocab=cfg.alert_effective_vocab,
                tokenizer=self.tokenizer,
            )
            self._log(f"Diversity monitoring enabled (every {monitor_freq} steps)")
        else:
            self.diversity_monitor = None

        if cfg.monitor_memory_states:
            from onyx.monitoring import MemoryMonitor
            self.memory_monitor = MemoryMonitor(
                num_layers=model_config.n_layers,
                monitor_every=monitor_freq,
                alert_norm_threshold=cfg.alert_memory_norm,
            )
            self._log(f"Memory state monitoring enabled (every {monitor_freq} steps)")
        else:
            self.memory_monitor = None

        self._log("\nSetup complete!\n")

    def get_batch(self) -> Optional[Dict[str, Any]]:
        if self._prefetched_batch is not None:
            batch = self._prefetched_batch
            self._prefetched_batch = None
        else:
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataloader)
                try:
                    batch = next(self.data_iter)
                except StopIteration:
                    return None

        out: Dict[str, Any] = {}
        out["input_ids"] = batch["input_ids"].to(self.device)
        out["labels"] = batch["labels"].to(self.device)
        if "cu_seqlens" in batch:
            out["cu_seqlens"] = batch["cu_seqlens"].to(self.device)
            out["num_segs"] = batch["num_segs"].to(self.device)
            out["max_seqlen"] = batch["max_seqlen"]
        return out

    def train_step(self, memory_states=None):
        cfg = self.config
        self.model.train()
        last_logits = None
        mps_debug = (
            cfg.mps_debug_memory
            and self.device_type == "mps"
            and (cfg.mps_debug_steps <= 0 or self.global_step < cfg.mps_debug_steps)
        )

        if memory_states is None:
            memory_states = self.model.init_memory_states(
                cfg.batch_size,
                self.device,
                self.autocast_dtype if self.use_autocast else torch.float32,
            )

        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_tokens = 0

        for accum_idx in range(self.accumulation_steps):
            batch = self.get_batch()
            if batch is None:
                break

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            total_tokens += input_ids.numel()
            if not self._checked_token_range:
                min_id = int(input_ids.min().item())
                max_id = int(input_ids.max().item())
                vocab_size = int(getattr(self.model.config, "vocab_size", 0) or 0)
                if vocab_size > 0 and (min_id < 0 or max_id >= vocab_size):
                    raise RuntimeError(f"Token IDs out of range: min={min_id} max={max_id} vocab={vocab_size}")
                self._checked_token_range = True

            cu_seqlens = None
            max_seqlen = None
            packed_cu_seqlens = None
            packed_num_segs = None
            if "cu_seqlens" in batch:
                packed_cu_seqlens = batch["cu_seqlens"]
                packed_num_segs = batch.get("num_segs")
            if self.device_type == "cuda" and packed_cu_seqlens is not None:
                # Build combined cu_seqlens across batch (flattened)
                B, S = input_ids.shape
                combined = [0]
                offset = 0
                for b in range(B):
                    n = int(packed_num_segs[b].item()) if packed_num_segs is not None else packed_cu_seqlens.size(1)
                    sample_cu = packed_cu_seqlens[b, :n]
                    for i in range(1, sample_cu.numel()):
                        combined.append(offset + int(sample_cu[i].item()))
                    offset += S
                cu_seqlens = torch.tensor(combined, dtype=torch.int32, device=self.device)
                max_seqlen = int(batch["max_seqlen"].item()) if torch.is_tensor(batch["max_seqlen"]) else S

            memory_states = self.model.detach_memory_states(memory_states)

            if cfg.mhc_debug_finite_checks:
                model_for_step = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
                try:
                    model_for_step.global_step = self.global_step
                    model_for_step.global_micro_step = accum_idx
                except Exception:
                    pass

            if self.use_autocast and self.device_type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        memory_states=memory_states,
                        update_memories=True,
                        return_memory_reg_loss=True,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        packed_cu_seqlens=packed_cu_seqlens,
                        packed_num_segs=packed_num_segs,
                    )
            elif self.use_autocast and self.device_type == "mps":
                with torch.autocast(device_type="mps", dtype=self.autocast_dtype):
                    out = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        memory_states=memory_states,
                        update_memories=True,
                        return_memory_reg_loss=True,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        packed_cu_seqlens=packed_cu_seqlens,
                        packed_num_segs=packed_num_segs,
                    )
            else:
                out = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    memory_states=memory_states,
                    update_memories=True,
                    return_memory_reg_loss=True,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    packed_cu_seqlens=packed_cu_seqlens,
                    packed_num_segs=packed_num_segs,
                )

            last_logits = out.get("logits")
            memory_states = out["memory_states"]
            loss = out.get("loss")
            if mps_debug:
                self._log_mps_mem("after_forward", micro_step=accum_idx, micro_total=self.accumulation_steps)
            if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
                self._nan_tripwire(
                    "loss",
                    loss=loss,
                    logits=last_logits,
                    memory_states=memory_states,
                    labels=labels,
                )

            loss = out["loss"] / self.accumulation_steps
            mem_reg = out.get("memory_reg_loss", 0.0)
            if isinstance(mem_reg, torch.Tensor):
                mem_reg = mem_reg / self.accumulation_steps

            if isinstance(mem_reg, torch.Tensor) and not torch.isfinite(mem_reg).all():
                self._log("[WARN] memory_reg_loss is non-finite; dropping it for this step.", force=True)
                mem_reg = torch.zeros_like(loss)

            combined_loss = loss + mem_reg if isinstance(mem_reg, torch.Tensor) else loss
            if isinstance(combined_loss, torch.Tensor) and not torch.isfinite(combined_loss).all():
                self._log("[WARN] combined_loss is non-finite; falling back to loss only for this step.", force=True)
                combined_loss = loss

            if self.scaler is not None:
                self.scaler.scale(combined_loss).backward()
            else:
                combined_loss.backward()
            if mps_debug:
                self._log_mps_mem("after_backward", micro_step=accum_idx, micro_total=self.accumulation_steps)

            total_loss += float(loss.item()) * self.accumulation_steps

        if total_tokens == 0:
            return None, memory_states

        self.last_memory_states = memory_states

        # === Monitoring ===
        # Token diversity monitoring
        if self.diversity_monitor and last_logits is not None:
            div_metrics = self.diversity_monitor.compute_metrics(last_logits, self.global_step)
            if div_metrics:
                # Log to wandb if available
                if cfg.wandb_project and WANDB_AVAILABLE:
                    wandb.log({
                        "diversity/top10_mass": div_metrics["top10_mass"],
                        "diversity/top50_mass": div_metrics["top50_mass"],
                        "diversity/top100_mass": div_metrics["top100_mass"],
                        "diversity/entropy": div_metrics["entropy"],
                        "diversity/entropy_ratio": div_metrics["entropy_ratio"],
                        "diversity/effective_vocab": div_metrics["effective_vocab"],
                    }, step=self.global_step)

                # Check for alerts
                warnings = self.diversity_monitor.check_alerts(div_metrics, self.global_step)
                for warning in warnings:
                    self._log(warning, force=True)

        # Memory state monitoring (optional)
        if self.memory_monitor and memory_states is not None:
            mem_metrics = self.memory_monitor.compute_metrics(memory_states, self.global_step)
            if mem_metrics:
                # Log to wandb if available
                if cfg.wandb_project and WANDB_AVAILABLE:
                    for layer_idx, norm in enumerate(mem_metrics["norms"]):
                        wandb.log({f"memory/layer_{layer_idx}_norm": norm}, step=self.global_step)
                    for layer_idx, update_mag in enumerate(mem_metrics["update_mags"]):
                        if update_mag > 0:  # Skip first measurement (no previous state)
                            wandb.log({f"memory/layer_{layer_idx}_update_mag": update_mag}, step=self.global_step)

                # Check for alerts
                mem_warnings = self.memory_monitor.check_alerts(mem_metrics, self.global_step)
                for warning in mem_warnings:
                    self._log(warning, force=True)

        if not hasattr(self, "_cms_manager"):
            self._cms_manager = None
            model_cfg = getattr(self.model, "config", None)
            if model_cfg is not None and getattr(model_cfg, "use_cms_ffn", False):
                self._cms_manager = CMSFrequencyManager(self.model, model_cfg)

        if self._cms_manager is not None:
            self._cms_manager.mask_gradients(self.global_step)

        if self._has_nonfinite_grads():
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.update()
            self.skipped_steps += 1
            return {
                "loss": total_loss / max(1, self.accumulation_steps),
                "grad_norm": float("nan"),
                "tokens": total_tokens,
                "skipped": 1,
            }, memory_states

        if cfg.gradient_clip > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.device_type == "mps":
                grad_norm_val = float(self._safe_clip_grad_norm(cfg.gradient_clip))
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                grad_norm_val = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
            if not math.isfinite(grad_norm_val):
                bad = self._find_nonfinite_grads()
                if bad:
                    self._log(f"[NaN] Non-finite grads in: {', '.join(bad)}", force=True)
                # If we're using GradScaler, allow it to handle transient fp16 overflows by
                # skipping the optimizer step and reducing the scale, instead of hard-failing.
                if self.scaler is None:
                    self._nan_tripwire(
                        "grad_norm",
                        grad_norm=grad_norm_val,
                        logits=last_logits,
                        memory_states=memory_states,
                    )
        else:
            grad_norm_val = 0.0

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if mps_debug:
            self._log_mps_mem("after_step", micro_step=0, micro_total=1)

        self.global_step += 1
        self.tokens_seen += total_tokens

        return {
            "loss": total_loss / max(1, self.accumulation_steps),
            "grad_norm": grad_norm_val,
            "tokens": total_tokens,
            "skipped": 0,
        }, memory_states

    def _maybe_report_memory(self) -> None:
        cfg = self.config
        if cfg.mem_report_every <= 0:
            return
        if self.global_step % cfg.mem_report_every != 0:
            return

        rss_gb = _get_rss_gb()
        mps_alloc_gb, mps_driver_gb = _get_mps_mem_gb()
        msg_parts = [f"[Mem] rss_gb={rss_gb:.2f}"]
        if mps_alloc_gb is not None:
            msg_parts.append(f"mps_alloc_gb={mps_alloc_gb:.2f}")
        if mps_driver_gb is not None:
            msg_parts.append(f"mps_driver_gb={mps_driver_gb:.2f}")
        self._log(" | ".join(msg_parts), force=True)

        if cfg.wandb_project and WANDB_AVAILABLE:
            payload = {"train/mem_rss_gb": rss_gb, "train/step": self.global_step}
            if mps_alloc_gb is not None:
                payload["train/mps_alloc_gb"] = mps_alloc_gb
            if mps_driver_gb is not None:
                payload["train/mps_driver_gb"] = mps_driver_gb
            wandb.log(payload, step=self.global_step)

        over = []
        if cfg.max_rss_gb and rss_gb > cfg.max_rss_gb:
            over.append(f"rss_gb={rss_gb:.2f} > {cfg.max_rss_gb:.2f}")
        if cfg.max_mps_alloc_gb and mps_alloc_gb is not None and mps_alloc_gb > cfg.max_mps_alloc_gb:
            over.append(f"mps_alloc_gb={mps_alloc_gb:.2f} > {cfg.max_mps_alloc_gb:.2f}")
        if over:
            self._log(f"[WARN] Memory threshold exceeded: {', '.join(over)}", force=True)
            if cfg.auto_interrupt_on_mem:
                self._interrupt_requested = True

    def _log_mps_mem(self, msg: str, *, micro_step: Optional[int] = None, micro_total: Optional[int] = None) -> None:
        if self.device_type != "mps" or not hasattr(torch, "mps"):
            return
        try:
            cur = torch.mps.current_allocated_memory() / 1e9
            drv = torch.mps.driver_allocated_memory() / 1e9
        except Exception:
            return
        step = self.global_step
        micro = ""
        if micro_step is not None and micro_total is not None:
            micro = f" micro={micro_step + 1}/{micro_total}"
        self._log(f"[MPS] step={step}{micro} {msg} cur_gb={cur:.2f} drv_gb={drv:.2f}", force=True)

    def _maybe_cleanup(self) -> None:
        cfg = self.config
        if self.device_type == "mps" and cfg.mps_empty_cache_every > 0 and self.global_step % cfg.mps_empty_cache_every == 0:
            try:
                torch.mps.empty_cache()
            except Exception as e:
                self._log(f"[WARN] torch.mps.empty_cache failed: {e}", force=True)
        if cfg.gc_collect_every > 0 and self.global_step % cfg.gc_collect_every == 0:
            gc.collect()

    def train(self):
        if not self._logger_configured:
            _configure_logger(self.config.log_file, self.config.tee, self.config.flush_logs)
            self._logger_configured = True

        if not hasattr(self, "model"):
            self.setup()

        cfg = self.config
        bench_mode = cfg.bench_steps and cfg.bench_steps > 0
        if not cfg.log_file and cfg.log_every < 10 and sys.platform == "darwin":
            self._log("Tip: use --log_file to prevent Terminal scrollback memory growth on macOS.", force=True)

        if cfg.max_steps is not None:
            total_steps = int(cfg.max_steps)
        elif cfg.train_tokens_target is not None:
            # IMPORTANT: `tokens_seen` is incremented by the *actual* tokens processed per optimizer step
            # (= accumulation_steps * batch_size * max_seq_len). If `cfg.tokens_per_step` is smaller
            # than that (e.g. user sets 256 while batch_size*max_seq_len is 8192), accumulation_steps
            # clamps to 1 and the true tokens/step becomes the batch tokens, not `cfg.tokens_per_step`.
            effective_tps = int(getattr(self, 'accumulation_steps', 1)) * int(cfg.batch_size) * int(cfg.max_seq_len)
            if effective_tps <= 0:
                effective_tps = max(1, int(cfg.tokens_per_step))
            if int(cfg.tokens_per_step) < effective_tps:
                self._log(f"[WARN] tokens_per_step ({cfg.tokens_per_step}) < effective tokens/step ({effective_tps}); using {effective_tps} for --train_tokens_target scheduling.", force=True)
            total_steps = int(cfg.train_tokens_target // max(1, effective_tps))
        else:
            # If neither is provided, run "epochs" based on a conservative estimate:
            # we approximate steps per epoch by counting batches for one pass.
            self._log("No --max_steps or --train_tokens_target provided; estimating steps_per_epoch by one streaming pass...")
            est_steps = 0
            # NOTE: this is a streaming pass; it will take time but not load all docs into RAM.
            for _ in self.dataloader:
                est_steps += 1
            est_steps = max(1, est_steps // max(1, self.accumulation_steps))
            total_steps = est_steps * max(1, int(cfg.num_epochs))
            # reset iterator
            self.data_iter = iter(self.dataloader)
            self._log(f"Estimated steps_per_epoch={est_steps:,} -> total_steps={total_steps:,}")

        if bench_mode:
            total_steps = self.global_step + int(cfg.bench_steps)

        if cfg.warmup_ratio is not None:
            if cfg.warmup_ratio <= 0.0 or cfg.warmup_ratio >= 1.0:
                self._log("[WARN] warmup_ratio must be between 0 and 1; ignoring.", force=True)
            else:
                cfg.warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
                if cfg.warmup_steps >= total_steps:
                    cfg.warmup_steps = max(1, total_steps - 1)

        if cfg.resume:
            resume_lr = get_lr(self.global_step, cfg, total_steps)
            set_lr(self.optimizer, resume_lr)

        self._log("=" * 70)
        self._log("Starting training")
        self._log(f"  Total steps: {total_steps:,}")
        if cfg.warmup_ratio is not None and 0.0 < cfg.warmup_ratio < 1.0:
            self._log(f"  Warmup steps: {cfg.warmup_steps} (ratio {cfg.warmup_ratio:.3f})")
        self._log(f"  Save every steps: {cfg.save_every_steps}")
        self._log(f"  keep_last_n: {cfg.keep_last_n} | keep_every_steps: {cfg.keep_every_steps}")
        self._log("=" * 70)

        if self._pending_rng_state is not None:
            self._set_rng_state(self._pending_rng_state)
            self._pending_rng_state = None
            self._rng_restored_once = True

        if cfg.resume:
            opt_lrs = [pg.get("lr", 0.0) for pg in self.optimizer.param_groups]
            sched_lr = get_lr(self.global_step, cfg, total_steps)
            mem_restored = self._resume_memory_states_loaded
            self._log(
                "[Resume] Integrity report: "
                f"step={self.global_step}, tokens_seen={self.tokens_seen:,}, "
                f"opt_lrs={[f'{lr:.2e}' for lr in opt_lrs]}, "
                f"sched_lr={sched_lr:.2e}, "
                f"memory_states_restored={mem_restored}, "
                f"rng_restored={self._rng_restored_once}, "
                f"dataset_state_loaded={self._resume_dataset_state_loaded}"
            )

        steps_per_epoch = None
        if cfg.num_epochs and total_steps and cfg.num_epochs > 0:
            steps_per_epoch = max(1, int(total_steps // max(1, cfg.num_epochs)))

        start_time = time.time()
        log_loss = 0.0
        log_tokens = 0
        log_steps = 0
        memory_states = None
        if cfg.persist_memory_across_steps:
            if self.last_memory_states is not None:
                memory_states = self.last_memory_states
            else:
                memory_states = self.model.init_memory_states(
                    cfg.batch_size,
                    self.device,
                    self.autocast_dtype if self.use_autocast else torch.float32,
                )

        while self.global_step < total_steps:
            if self._interrupt_requested:
                self._log("\n[INTERRUPT] Saving checkpoint before exit...", force=True)
                self.save_checkpoint(f"interrupt_step_{self.global_step}")
                self._log("Checkpoint saved. Exiting.", force=True)
                return

            if steps_per_epoch and self.global_step > 0 and self.global_step % steps_per_epoch == 0:
                self.current_epoch += 1
                if cfg.reset_memory_on_epoch:
                    memory_states = None

            if cfg.reset_memory_every_steps > 0 and self.global_step > 0 and self.global_step % cfg.reset_memory_every_steps == 0:
                memory_states = None

            lr = get_lr(self.global_step, cfg, total_steps)
            set_lr(self.optimizer, lr)

            metrics, memory_states = self.train_step(
                memory_states=memory_states if cfg.persist_memory_across_steps else None
            )
            if metrics is None:
                continue

            log_loss += metrics["loss"]
            log_tokens += metrics["tokens"]
            log_steps += 1

            if self.global_step % cfg.log_every == 0:
                elapsed = time.time() - start_time
                avg_loss = log_loss / max(1, log_steps)
                tokps = log_tokens / max(1e-8, elapsed)
                scale = None
                if self.scaler is not None:
                    try:
                        scale = float(self.scaler.get_scale())
                    except Exception:
                        scale = None

                msg = f"Step {self.global_step:6d} | Loss {avg_loss:.4f} | LR {lr:.2e} | Tok/s {tokps:.0f}"
                if scale is not None:
                    msg += f" | Scale {scale:.1f}"
                if self.skipped_steps:
                    msg += f" | Skipped {self.skipped_steps}"
                self._log(msg)

                if cfg.wandb_project and WANDB_AVAILABLE:
                    payload = {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/grad_norm": metrics["grad_norm"],
                        "train/tokps": tokps,
                        "train/tokens_seen": self.tokens_seen,
                        "train/step": self.global_step,
                        "train/skip_steps": self.skipped_steps,
                    }
                    if scale is not None:
                        payload["train/grad_scale"] = scale
                    wandb.log(payload)

                log_loss = 0.0
                log_tokens = 0
                log_steps = 0
                start_time = time.time()

            self._maybe_report_memory()
            self._maybe_cleanup()

            if cfg.save_every_steps > 0 and (self.global_step % cfg.save_every_steps == 0) and not (bench_mode and cfg.disable_saves_during_bench):
                self.save_checkpoint(f"step_{self.global_step}")

            if bench_mode and (self.global_step >= total_steps):
                self.save_checkpoint(f"bench_step_{self.global_step}")
                self._log("Benchmark complete. Exiting.", force=True)
                return

        self.save_checkpoint(f"final_step_{self.global_step}")
        self._log("\nTraining complete!", force=True)
        self._log(f"  Total steps: {self.global_step:,}", force=True)
        self._log(f"  Total tokens: {self.tokens_seen:,}", force=True)

    def save_checkpoint(self, name: str):
        cfg = self.config
        os.makedirs(cfg.save_dir, exist_ok=True)

        if cfg.stage_name:
            # Prefix stage name for progressive sequence schedules.
            name = f"{cfg.stage_name}_{name}"
        path = os.path.join(cfg.save_dir, f"checkpoint_{name}.pt")
        tmp_path = path + ".tmp"

        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_loss": self.best_loss,
            "current_epoch": getattr(self, "current_epoch", 0),
            "config": vars(cfg),
        }
        ckpt["rng_state"] = self._get_rng_state()
        if self.last_memory_states is not None:
            ckpt["memory_states"] = _tree_map_tensors(
                self.last_memory_states,
                lambda t: t.detach().to("cpu"),
            )
        if hasattr(self, "dataset") and hasattr(self.dataset, "state_dict"):
            ckpt["dataset_state"] = self.dataset.state_dict(mode=cfg.dataset_state_mode)
        if self.scaler is not None:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)  # atomic
        self._log(f"Saved checkpoint: {path}", force=True)

        self._cleanup_checkpoints()

    def load_checkpoint(self, path: str):
        self._log(f"Resuming from: {path}", force=True)
        map_location = getattr(self, "device", torch.device("cpu"))
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        sd = ckpt["model_state_dict"]
        # Resize vocab tensors if needed
        if hasattr(model, "embed"):
            self._resize_vocab_tensor(sd, "embed.weight", model.embed.weight)
        if hasattr(model, "lm_head"):
            self._resize_vocab_tensor(sd, "lm_head.weight", model.lm_head.weight)

        model.load_state_dict(sd, strict=True)
        if not self.config.use_amp:
            enforce_fp32_everywhere(model)
            _assert_fp32_everywhere(model)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.global_step = int(ckpt.get("global_step", 0))
        self.tokens_seen = int(ckpt.get("tokens_seen", 0))
        self.best_loss = float(ckpt.get("best_loss", float("inf")))
        self.current_epoch = int(ckpt.get("current_epoch", 0))

        self._pending_rng_state = ckpt.get("rng_state")
        mem_states = ckpt.get("memory_states")
        if mem_states is not None:
            def _move_state(t: torch.Tensor) -> torch.Tensor:
                if not self.config.use_amp and t.is_floating_point():
                    return t.to(self.device, dtype=torch.float32)
                return t.to(self.device)

            self.last_memory_states = _tree_map_tensors(mem_states, _move_state)
            self._resume_memory_states_loaded = True
            if not self.config.use_amp:
                _assert_fp32_memory_states(self.last_memory_states)
        else:
            self.last_memory_states = None
            self._resume_memory_states_loaded = False

        self._resume_dataset_state_loaded = False
        if hasattr(self, "dataset") and "dataset_state" in ckpt:
            try:
                self.dataset.load_state_dict(ckpt["dataset_state"])
                if hasattr(self, "dataloader"):
                    self.data_iter = iter(self.dataloader)
                self._resume_dataset_state_loaded = bool(
                    getattr(self.dataset, "_doc_state", None) or getattr(self.dataset, "_pack_state", None)
                )
                doc_state = getattr(self.dataset, "_doc_state", None) or {}
                pack_state = getattr(self.dataset, "_pack_state", None) or {}
                file_pos = doc_state.get("file_pos", 0)
                pack_seq_len = len(pack_state.get("current_seq", []))
                pack_label_len = len(pack_state.get("current_labels", []))
                pack_bound_len = len(pack_state.get("current_boundaries", []))
                first_batch_tokens = None
                if hasattr(self, "data_iter"):
                    try:
                        self._prefetched_batch = next(self.data_iter)
                        if self._prefetched_batch is not None and "input_ids" in self._prefetched_batch:
                            first_batch_tokens = int(self._prefetched_batch["input_ids"].numel())
                    except StopIteration:
                        self._prefetched_batch = None
                    except Exception as e:
                        self._log(f"[WARN] Resume batch peek failed: {e}", force=True)
                        self._prefetched_batch = None
                self._log(
                    "[Resume] Sanity: "
                    f"first_batch_tokens={first_batch_tokens}, "
                    f"file_pos={file_pos}, "
                    f"pack_seq={pack_seq_len}, "
                    f"pack_labels={pack_label_len}, "
                    f"pack_bounds={pack_bound_len}"
                )
            except Exception as e:
                self._log(f"[WARN] Failed to load dataset state: {e}", force=True)

        self._log(f"  Resumed at step {self.global_step}, tokens_seen={self.tokens_seen:,}")

    def _cleanup_checkpoints(self):
        cfg = self.config
        if cfg.keep_last_n <= 0:
            return

        # Only manage step checkpoints; keep "final"/"epoch"/"interrupt" etc.
        ckpt_dir = Path(cfg.save_dir)
        step_ckpts = list(ckpt_dir.glob("checkpoint_step_*.pt"))
        step_ckpts += list(ckpt_dir.glob("checkpoint_*_step_*.pt"))
        step_ckpts = sorted(set(step_ckpts))
        if len(step_ckpts) <= cfg.keep_last_n:
            return

        def _step_num(p: Path) -> Optional[int]:
            # checkpoint_<prefix>_step_12345.pt
            m = re.search(r"_step_(\d+)$", p.stem)
            return int(m.group(1)) if m else None

        keep: set[Path] = set()

        # Keep milestone steps if requested
        if cfg.keep_every_steps and cfg.keep_every_steps > 0:
            for p in step_ckpts:
                n = _step_num(p)
                if n is not None and n % cfg.keep_every_steps == 0:
                    keep.add(p)

        # Keep newest N by mtime
        # Keep newest N by step number (deterministic; avoids coarse mtime flakiness)
        newest = sorted(step_ckpts, key=lambda p: (_step_num(p) if _step_num(p) is not None else -1), reverse=True)[: cfg.keep_last_n]
        keep.update(newest)

        # Delete others
        for p in step_ckpts:
            if p not in keep:
                try:
                    p.unlink()
                except Exception as e:
                    self._log(f"[WARN] Could not delete old checkpoint {p}: {e}", force=True)


# =============================================================================
# Main
# =============================================================================

def _build_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    amp_default = TrainingConfig.__dataclass_fields__["use_amp"].default
    use_amp = args.amp if args.amp is not None else amp_default

    cfg = TrainingConfig(
        data_glob=args.data_glob,
        tokenizer_name=args.tokenizer,
        model_config_path=args.model_config,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        tokens_per_step=args.tokens_per_step,
        pack_sequences=args.pack_sequences,
        drop_remainder=args.drop_remainder,
        shuffle_buffer_docs=args.shuffle_buffer_docs,
        num_epochs=args.num_epochs,
        train_tokens_target=args.train_tokens_target,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        memory_lr_scale=args.memory_lr_scale,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        label_smoothing=args.label_smoothing,
        entropy_reg_weight=args.entropy_reg_weight,
        feedback_strength=args.feedback_strength,
        use_m3_optimizer=(not args.use_adamw),
        use_torch_compile=args.compile,
        compile_mode=args.compile_mode,
        use_amp=use_amp,
        amp_dtype=args.amp_dtype,
        amp_flag=args.amp,
        gradient_checkpointing=args.gradient_checkpointing,
        experimental_mhc=args.experimental_mhc,
        mhc_n=args.mhc_n,
        mhc_sinkhorn=args.mhc_sinkhorn,
        mhc_sinkhorn_iters=args.mhc_sinkhorn_iters,
        mhc_mode=args.mhc_mode,
        mhc_debug_finite_checks=args.mhc_debug_finite_checks,
        mhc_debug_every=args.mhc_debug_every,
        persist_memory_across_steps=args.persist_memory_across_steps,
        reset_memory_every_steps=args.reset_memory_every_steps,
        reset_memory_on_epoch=args.reset_memory_on_epoch,
        bench_steps=args.bench_steps,
        disable_saves_during_bench=args.disable_saves_during_bench,
        save_dir=args.save_dir,
        save_every_steps=args.save_every_steps,
        keep_last_n=args.keep_last_n,
        keep_every_steps=args.keep_every_steps,
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
        init_strict=args.init_strict,
        log_every=args.log_every,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        monitor_diversity=args.monitor_diversity,
        monitor_memory_states=args.monitor_memory_states,
        monitor_every=args.monitor_every,
        alert_top10_mass=args.alert_top10_mass,
        alert_entropy_ratio=args.alert_entropy_ratio,
        alert_effective_vocab=args.alert_effective_vocab,
        alert_memory_norm=args.alert_memory_norm,
        dry_run=args.dry_run,
        seed=args.seed,
        log_file=args.log_file,
        tee=args.tee,
        flush_logs=args.flush_logs,
        mem_report_every=args.mem_report_every,
        max_rss_gb=args.max_rss_gb,
        max_mps_alloc_gb=args.max_mps_alloc_gb,
        auto_interrupt_on_mem=args.auto_interrupt_on_mem,
        mps_empty_cache_every=args.mps_empty_cache_every,
        gc_collect_every=args.gc_collect_every,
        dataset_state_mode=args.dataset_state_mode,
        quiet=args.quiet,
        mps_debug_memory=args.mps_debug_memory,
        mps_debug_steps=args.mps_debug_steps,
    )

    if cfg.experimental_mhc:
        base = os.path.basename(cfg.save_dir.rstrip(os.sep))
        if "mhc" not in base.lower():
            cfg.save_dir = os.path.join(cfg.save_dir, "mhc")

    if args.dry_run:
        cfg.max_steps = 3
        cfg.log_every = 1
        cfg.save_every_steps = 0
        cfg.use_torch_compile = False
        cfg.wandb_project = None
        log("\nDRY RUN MODE (3 steps)\n")

    return cfg


def _load_schedule(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("schedule_json must be a non-empty list of stage configs.")
    return data


def _find_latest_checkpoint(ckpt_dir: str, pattern: str = "checkpoint_*_step_*.pt") -> Optional[str]:
    paths = list(Path(ckpt_dir).glob(pattern))
    if not paths:
        return None

    def _step_num(p: Path) -> int:
        m = re.search(r"_step_(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    paths.sort(key=lambda p: (_step_num(p), p.stat().st_mtime), reverse=True)
    return str(paths[0])


def _apply_stage_overrides(cfg: TrainingConfig, stage: Dict[str, Any]) -> None:
    for key, value in stage.items():
        if key in ("name", "resume"):
            continue
        if key in TrainingConfig.__dataclass_fields__:
            setattr(cfg, key, value)
        else:
            log(f"[WARN] Unknown stage override ignored: {key}")


def _run_schedule(args: argparse.Namespace) -> None:
    stages = _load_schedule(args.schedule_json)
    base_cfg = _build_config_from_args(args)
    last_stage_name: Optional[str] = None

    for idx, stage in enumerate(stages):
        cfg = copy.deepcopy(base_cfg)
        _apply_stage_overrides(cfg, stage)
        stage_name = stage.get("name") or f"stage{idx}_len{cfg.max_seq_len}"
        cfg.stage_name = stage_name

        if "resume" in stage:
            cfg.resume = stage["resume"]
        elif idx > 0 and args.resume_stage_from_last:
            pattern = f"checkpoint_{last_stage_name}_step_*.pt" if last_stage_name else "checkpoint_*_step_*.pt"
            cfg.resume = _find_latest_checkpoint(cfg.save_dir, pattern=pattern)
            if cfg.resume is None and last_stage_name:
                cfg.resume = _find_latest_checkpoint(cfg.save_dir)

        print(f"\n=== Running {stage_name} ===")
        if cfg.resume:
            print(f"Resume: {cfg.resume}")
        Trainer(cfg).train()
        last_stage_name = stage_name

def main():
    p = argparse.ArgumentParser(
        description="Train Onyx Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Recommended:\n"
            "  python -m onyx.train ... --log_file /path/train.log --tee "
            "--mem_report_every 50 --mps_empty_cache_every 50\n"
            "Tip: On macOS, Terminal/iTerm scrollback can grow memory quickly. "
            "Use --log_file (and --tee if you want console output) or limit scrollback."
        ),
    )

    p.add_argument("--data_glob", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="NousResearch/Hermes-2-Pro-Llama-3-8B")
    p.add_argument("--model_config", type=str, default=None)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--tokens_per_step", type=int, default=8192)
    p.add_argument("--no_pack_sequences", action="store_false", dest="pack_sequences")
    p.add_argument("--drop_remainder", action="store_true")
    p.add_argument("--shuffle_buffer_docs", type=int, default=2048)

    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--train_tokens_target", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)

    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--memory_lr_scale", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--warmup_ratio", type=float, default=None, help="Override warmup_steps as ratio of total_steps.")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--entropy_reg_weight", type=float, default=0.0)
    p.add_argument("--feedback_strength", type=float, default=1.0)

    p.add_argument("--use_adamw", action="store_true")

    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", type=str, default="reduce-overhead")

    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--amp_dtype", type=str, choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--persist_memory_across_steps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reset_memory_every_steps", type=int, default=0)
    p.add_argument("--reset_memory_on_epoch", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--experimental_mhc", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mhc_n", type=int, default=2)
    p.add_argument("--mhc_sinkhorn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mhc_sinkhorn_iters", type=int, default=10)
    p.add_argument("--mhc_mode", type=str, choices=["mhc", "hc"], default="mhc")
    p.add_argument("--mhc_debug_finite_checks", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mhc_debug_every", type=int, default=0)

    p.add_argument("--bench_steps", type=int, default=0)
    p.add_argument("--disable_saves_during_bench", action="store_true")

    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--save_every_steps", type=int, default=0)
    p.add_argument("--keep_last_n", type=int, default=5)
    p.add_argument("--keep_every_steps", type=int, default=0)
    p.add_argument("--dataset_state_mode", type=str, choices=["full", "light"], default="light")

    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--init_checkpoint", type=str, default=None)
    p.add_argument(
        "--init_strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If false, allow partial init_checkpoint loads when architecture changes (warm-start).",
    )

    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--log_file", type=str, default="")
    p.add_argument("--tee", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--flush_logs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    # Monitoring
    p.add_argument("--monitor_diversity", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--monitor_memory_states", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--monitor_every", type=int, default=0, help="0 = use log_every")
    p.add_argument("--alert_top10_mass", type=float, default=0.7)
    p.add_argument("--alert_entropy_ratio", type=float, default=0.3)
    p.add_argument("--alert_effective_vocab", type=int, default=100)
    p.add_argument("--alert_memory_norm", type=float, default=100.0)

    p.add_argument("--mem_report_every", type=int, default=50)
    p.add_argument("--max_rss_gb", type=float, default=0.0)
    p.add_argument("--max_mps_alloc_gb", type=float, default=0.0)
    p.add_argument("--auto_interrupt_on_mem", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mps_empty_cache_every", type=int, default=50)
    p.add_argument("--gc_collect_every", type=int, default=200)
    p.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mps_debug_memory", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mps_debug_steps", type=int, default=3)

    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--schedule_json", type=str, default=None, help="JSON list of stage overrides for progressive seq-len training.")
    p.add_argument("--resume_stage_from_last", action=argparse.BooleanOptionalAction, default=True, help="Auto-resume each stage from latest checkpoint.")

    args = p.parse_args()

    if args.schedule_json:
        _run_schedule(args)
        return

    cfg = _build_config_from_args(args)
    _configure_logger(cfg.log_file, cfg.tee, cfg.flush_logs)
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
