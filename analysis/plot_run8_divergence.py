"""Three-panel figure: what 20 GRPO iterations actually moved.

The point is the DIVERGENCE, not any one panel. The policy moved a great deal,
the reward it is optimising moved slightly, and the task success rate it is
supposed to stand in for did not move. Read together with kl_coef = 0, that is a
recognisable failure mode -- drift towards the shaped reward without task
improvement -- and it is the reason the next run is a question about the KL
anchor and not only about step count.

    python paper/scripts/plot_run8_divergence.py --metrics <run8/metrics.jsonl> --out <fig.png>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical slots 1-3 (light surface #fcfcfb); see the palette reference.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SURFACE = "#fcfcfb"


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """slope, its standard error, and t. OLS on purpose: Newey-West at n=20 is
    not trustworthy, and Durbin-Watson showed no autocorrelation worth correcting
    (success 2.12, reward 1.65), so the plain fit is the honest one."""
    A = np.vstack([x, np.ones_like(x)]).T
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ b
    sxx = ((x - x.mean()) ** 2).sum()
    se = np.sqrt((resid ** 2).sum() / (len(x) - 2) / sxx)
    return b[0], se, b[0] / se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("paper/figures/run8_divergence.png"))
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.metrics.read_text().splitlines() if l.strip()]
    it = np.array([r["iter"] for r in rows], dtype=float)

    panels = [
        ("success_rate", BLUE, "Task success rate", "fraction of 64 episodes", "{:+.2f}"),
        ("mean_reward", ORANGE, "Mean staged reward  (the quantity being optimised)", "reward", "{:+.2f}"),
        ("drift", AQUA, "Policy drift from the SFT initialization", "L2 distance", "{:+.2f}"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.0), sharex=True,
                             facecolor=SURFACE, gridspec_kw={"hspace": 0.32})
    for ax, (key, color, title, ylab, _) in zip(axes, panels):
        y = np.array([r[key] for r in rows], dtype=float)
        slope, se, t = ols(it, y)

        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)

        ax.plot(it, y, color=color, linewidth=2.0, marker="o", markersize=4.5,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.2,
                zorder=3)
        ax.plot(it, slope * it + (y.mean() - slope * it.mean()), color=color,
                linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.85, zorder=2)

        verdict = "significant" if abs(t) > 2.101 else "not significant"
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=7)
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        ax.text(0.985, 0.055, f"slope {slope:+.4f}/iter   t = {t:+.2f}   {verdict}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.2, color=INK2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.85))

    axes[-1].set_xlabel("GRPO iteration  (64 episodes each)", color=INK2, fontsize=9)
    axes[-1].set_xticks(range(0, len(it), 2))
    fig.suptitle("20 iterations of Flow-SDE GRPO on SmolVLA, run8 (kl_coef = 0)",
                 color=INK, fontsize=12, x=0.012, ha="left", y=0.983)
    fig.text(0.012, 0.947,
             "The policy moved a great deal, the reward moved slightly, the success rate did not move.",
             color=INK2, fontsize=9.5, ha="left")
    fig.text(0.012, 0.008,
             "One run, no anchored-KL control. Trends are exploratory OLS fits, not\n"
             "preregistered; two of the three would not survive a multiplicity correction.\n"
             "Per-iteration binomial standard error at p~0.30, n=64 is +/-5.7 pp --- the size\n"
             "of the entire change in the top panel.",
             color=INK2, fontsize=7.6, ha="left", linespacing=1.55)
    fig.subplots_adjust(top=0.900, bottom=0.145, left=0.105, right=0.975)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
