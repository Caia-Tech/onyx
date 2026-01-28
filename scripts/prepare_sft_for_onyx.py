#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, TextIO


def _coerce_str(x: Any) -> Optional[str]:
    if isinstance(x, str):
        return x
    return None


def _convert_record(obj: Any) -> Optional[Dict[str, str]]:
    if not isinstance(obj, dict):
        return None
    user = _coerce_str(obj.get("input"))
    assistant = _coerce_str(obj.get("output"))
    if not user or not assistant:
        return None
    return {"user": user, "assistant": assistant}


def _open_text(path: Path, mode: str) -> TextIO:
    return path.open(mode, encoding="utf-8", newline="\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert input/output SFT JSONL into Onyx train.py chat-shaped JSONL.")
    p.add_argument("--in_jsonl", required=True, help="Input JSONL path (expects fields: input, output).")
    p.add_argument("--out_jsonl", required=True, help="Output JSONL path (writes {user, assistant} per line).")
    p.add_argument("--max_records", type=int, default=0, help="If >0, stop after writing this many records.")
    p.add_argument("--log_every", type=int, default=200000, help="Progress log cadence (lines read).")
    args = p.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    lines = 0
    written = 0
    bad_json = 0
    skipped = 0

    with _open_text(in_path, "r") as fin, _open_text(out_path, "w") as fout:
        for line in fin:
            lines += 1
            s = line.strip()
            if not s:
                skipped += 1
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            rec = _convert_record(obj)
            if rec is None:
                skipped += 1
                continue

            fout.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

            if args.max_records > 0 and written >= args.max_records:
                break

            if args.log_every > 0 and (lines % args.log_every) == 0:
                dt = max(1e-6, time.time() - t0)
                r = written / dt
                print(
                    f"[prep] lines={lines:,} written={written:,} skipped={skipped:,} bad_json={bad_json:,} rate={r:,.1f} rec/s",
                    file=sys.stderr,
                )

    dt = max(1e-6, time.time() - t0)
    r = written / dt
    print(
        f"[prep] done: lines={lines:,} written={written:,} skipped={skipped:,} bad_json={bad_json:,} secs={dt:.1f} rate={r:,.1f} rec/s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

