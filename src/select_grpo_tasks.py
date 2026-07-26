"""Mechanically select GRPO tasks from screening results.

Hand-copying "promising" tasks into the config leaves no selection criterion
in code (why these 8 becomes a mere assertion) and invites transcription
errors. Instead, select by explicit rules from the screening summary.json,
deterministically reproducing the same 8.

Rules:
    1. Compute each task's success rate success/n, excluding "_overall"
    2. Sort descending by (success rate, success count, -task_id)
       — ties in rate go to more successes; further ties to ascending
         task_id (lower id first = deterministic tiebreak)
    3. Drop tasks below min_success (default 1, i.e. never succeeded)
    4. Take the top top_n (default 8)

Usage:
    python -m src.select_grpo_tasks --summary /content/screen_data/summary.json
    python -m src.select_grpo_tasks --summary s.json --emit_yaml  # emit the task_ids block
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_tasks(
    summary: dict[str, dict],
    top_n: int = 8,
    min_success: int = 1,
) -> list[str]:
    """Select and return task keys from the summary.json dict (deterministic)."""
    rows = []
    for key, stat in summary.items():
        if key == "_overall":
            continue
        n = int(stat.get("n", 0))
        s = int(stat.get("success", 0))
        if n == 0 or s < min_success:
            continue
        tid = int(key.split(":")[-1])
        rows.append((s / n, s, -tid, key))
    rows.sort(reverse=True)  # descending by success rate → success count → -task_id
    return [key for _, _, _, key in rows[:top_n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--top_n", type=int, default=8)
    ap.add_argument("--min_success", type=int, default=1)
    ap.add_argument("--emit_yaml", action="store_true", help="emit as a task_ids: YAML block")
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text())
    selected = select_tasks(summary, top_n=args.top_n, min_success=args.min_success)

    if args.emit_yaml:
        print("task_ids:")
        for k in selected:
            print(f"  - {k}")
    else:
        for k in selected:
            stat = summary[k]
            print(f"{k}\t{stat['success']}/{stat['n']}")
        print(f"# selected {len(selected)} / {len(summary) - ('_overall' in summary)} candidates")


if __name__ == "__main__":
    main()
