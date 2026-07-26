"""Plot GRPO learning curves (success / staged reward vs iteration).

metrics.jsonl can record duplicate iters under resume operation, so the last
row per iter is used. Success gets a Wilson 95% CI.

Usage:
    python -m src.plot_grpo_curve --metrics /path/to/metrics.jsonl \
        --out assets/rl_curve_grpo.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_metrics(path: Path) -> list[dict]:
    rows: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows[int(d["iter"])] = d  # for duplicate iters, keep the last row (the redo after resume)
    return [rows[k] for k in sorted(rows)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("assets/rl_curve_grpo.png"))
    ap.add_argument("--title", default="GRPO on LIBERO-Plus (SmolVLA, staged reward)")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_metrics(args.metrics)
    iters = [r["iter"] for r in rows]
    succ = [r["success_rate"] for r in rows]
    rew = [r["mean_reward"] for r in rows]
    lo, hi = [], []
    for r in rows:
        n = int(r.get("n_episodes", 0))
        k = round(r["success_rate"] * n)
        a, b = wilson_ci(k, n)
        lo.append(a)
        hi.append(b)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ax.fill_between(iters, lo, hi, color="#4C78A8", alpha=0.15, linewidth=0)
    ax.plot(iters, succ, color="#4C78A8", linewidth=2, marker="o", markersize=4,
            label="success rate (SDE rollout)")
    ax.plot(iters, rew, color="#E45756", linewidth=2, marker="o", markersize=4,
            label="mean staged reward")
    ax.set_xlabel("GRPO iteration")
    ax.set_ylabel("rate / reward")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(iters[:: max(1, len(iters) // 16)])
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(args.title, fontsize=11)
    ax.text(0.99, 0.02,
            "shaded: Wilson 95% CI (n per iter = group_size × tasks)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#666666")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"saved -> {args.out} ({len(rows)} iterations)")


if __name__ == "__main__":
    main()
