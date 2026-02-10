#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download nuprl/AgentPack and export claude.jsonl")
    ap.add_argument("--output", required=True, help="Output jsonl path, for example data/claude.jsonl")
    ap.add_argument("--split", default="train", help="HF split")
    ap.add_argument("--data-dir", default="train", help="HF data_dir")
    ap.add_argument("--limit", type=int, default=0, help="0 means all rows")
    ap.add_argument("--progress-every", type=int, default=1000)
    args = ap.parse_args()

    ensure_parent(args.output)

    ds = load_dataset("nuprl/AgentPack", split=args.split, data_dir=args.data_dir, streaming=True)

    started = time.time()
    written = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            out = {
                "event_id": row.get("event_id"),
                "agent": row.get("agent"),
                "repo": row.get("repo"),
                "sha": row.get("sha"),
                "description": row.get("description") or row.get("message") or row.get("pr_title") or row.get("commit_message") or "",
                "patch": row.get("patch") or "",
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

            if written % args.progress_every == 0:
                elapsed = max(1e-9, time.time() - started)
                rate = written / elapsed
                print(f"written={written} ({rate:.1f} rows/s)", flush=True)

            if args.limit and written >= args.limit:
                break

    elapsed = max(1e-9, time.time() - started)
    print(f"done: {written} rows -> {args.output} ({written / elapsed:.1f} rows/s)")


if __name__ == "__main__":
    main()
