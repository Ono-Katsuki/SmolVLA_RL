"""Every between-arm contrast this project can report, with its interval.

The figure exists to make one thing visible: the only interval that excludes zero
is the one we withdrew. Full-sample eval A is shown because withdrawing a result
means showing what was withdrawn, not deleting it.

    python paper/scripts/plot_eval_intervals.py --out paper/figures/eval_intervals.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("paper/data"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures/eval_intervals.png"))
    a = ap.parse_args()

    A = json.loads((a.data / "eval_A_summary.json").read_text())
    W = json.loads((a.data / "eval_A_wave0_reanalysis.json").read_text())["wave0_paired"]
    P = json.loads((a.data / "eval_unseen_prereg_result.json").read_text())

    rows = []  # (label, point pp, lo, hi, withdrawn, note)
    for arm, name in (("rs_sft", "RS-SFT"), ("dpo", "flow-DPO"), ("grpo", "GRPO")):
        d = A["paired_vs_sft"][arm]
        lo, hi = [v * 100 for v in d["per_task_diff_ci95_taskboot"]]
        pt = (A["per_task_successes_out_of_15"][arm]["79"] * 0 + (lo + hi) / 2)
        # point estimate = mean per-task difference, recomputed from the counts
        s = A["per_task_successes_out_of_15"]
        ts = A["task_ids_effective"]
        pt = sum((s[arm][str(t)] - s["sft"][str(t)]) / 15 for t in ts) / len(ts) * 100
        rows.append((f"{name} vs SFT", pt, lo, hi, True,
                     "105 eps/arm, withdrawn"))
    w = W["rs_sft_vs_sft"]
    rows.append(("RS-SFT vs SFT", (W["rs_sft"]["rate"] - W["sft"]["rate"]) * 100,
                 w["diff_ci95_pp"][0], w["diff_ci95_pp"][1], False,
                 "wave 0 only, 56 eps/arm"))
    pr = P["primary"]
    rows.append(("GRPO vs SFT", pr["point_estimate_pp"], *pr["ci95_task_bootstrap_pp"], False,
                 "preregistered, 48 unseen tasks"))

    fig, ax = plt.subplots(figsize=(8.0, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ys = list(range(len(rows)))[::-1]
    ax.axvline(0, color=INK2, linewidth=1.0, zorder=1)
    for y, (label, pt, lo, hi, withdrawn, note) in zip(ys, rows):
        c = ORANGE if withdrawn else BLUE
        ax.plot([lo, hi], [y, y], color=c, linewidth=2.0, solid_capstyle="round", zorder=3)
        ax.plot([pt], [y], marker="o", markersize=8, color=c, zorder=4,
                markerfacecolor=SURFACE if withdrawn else c,
                markeredgecolor=c, markeredgewidth=2.0)
        ax.text(hi + 1.2, y, f"{pt:+.1f} [{lo:+.1f}, {hi:+.1f}]",
                va="center", fontsize=8.6, color=INK2)
        ax.text(-0.012, y, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9.5,
                color=INK if not withdrawn else INK2)
        ax.text(-0.012, y - 0.30, note,
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=7.6, color=INK2)

    ax.set_yticks([]); ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.set_xlabel("mean per-task success-rate difference vs. SFT  (percentage points, "
                  "task-clustered 95% CI)", color=INK2, fontsize=9)
    fig.suptitle("Every between-arm contrast, and what survives the audit",
                 color=INK, fontsize=12, x=0.012, ha="left", y=0.975)
    fig.text(0.012, 0.900,
             "The only interval that excludes zero is the one we withdrew.",
             color=INK2, fontsize=9.5, ha="left")
    fig.text(0.012, 0.030,
             "Withdrawn (hollow, orange): eval A ran on the training initial states and its pairing\n"
             "breaks for 7 of 15 episodes per task, so the full sample is not a valid paired contrast.\n"
             "Reported (solid, blue): the wave-0 subset and the preregistered unseen-task test.",
             color=INK2, fontsize=7.6, ha="left", linespacing=1.5)
    fig.subplots_adjust(top=0.840, bottom=0.335, left=0.215, right=0.795)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
