"""Compute the preregistered analysis for the unseen-task study from raw result.json.

    python analysis/analyze_unseen_prereg.py <sft/result.json> <grpo/result.json>

Both are produced by src.eval_heldout and live under the Drive output dir. This
recomputes everything the paper reports, including the preregistered SECONDARY
endpoint (exact McNemar on episode-paired discordance), which needs the
per-episode records rather than per-task counts.

Prespecified: primary = mean per-task success-rate difference (grpo - sft) with a
95% task-clustered bootstrap CI, 20000 resamples, seed 12345. Secondary = exact
McNemar. Tasks that errored for BOTH arms are dropped; an error on one arm only is
an imbalance to repair, not to drop, and this script refuses to proceed on one.
"""
from __future__ import annotations

import json
import sys
from math import comb

import numpy as np


def load(path: str) -> dict:
    r = json.loads(open(path).read())
    return {t["task_id"]: t for t in r["per_task"]}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    sft, grpo = load(sys.argv[1]), load(sys.argv[2])
    assert set(sft) == set(grpo), "arms evaluated different task lists -- VOID"

    err_s = {t for t, v in sft.items() if v.get("error")}
    err_g = {t for t, v in grpo.items() if v.get("error")}
    if err_s ^ err_g:
        raise SystemExit(f"IMBALANCE: errored on one arm only: {sorted(err_s ^ err_g)}")
    keep = sorted(set(sft) - err_s)

    d = np.array([(grpo[t]["k"] - sft[t]["k"]) / sft[t]["n"] for t in keep])
    rng = np.random.default_rng(12345)
    bs = np.array([rng.choice(d, d.size, replace=True).mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])

    # secondary: pair episodes by (task, init_state); the runner logs init_state
    b = c = 0
    for t in keep:
        gs = {e["init_state"]: e["success"] for e in sft[t]["episodes"]}
        gg = {e["init_state"]: e["success"] for e in grpo[t]["episodes"]}
        assert gs.keys() == gg.keys(), f"task {t}: initial states differ between arms"
        for i in gs:
            b += gg[i] and not gs[i]      # grpo wins
            c += gs[i] and not gg[i]      # sft wins

    ns = sum(sft[t]["k"] for t in keep); ng = sum(grpo[t]["k"] for t in keep)
    n = sum(sft[t]["n"] for t in keep)
    print(f"effective tasks {len(keep)}   dropped (errored both arms) {sorted(err_s)}")
    print(f"sft  {ns}/{n} = {ns/n:.4f}\ngrpo {ng}/{n} = {ng/n:.4f}")
    print(f"\nPRIMARY   mean per-task diff {d.mean()*100:+.2f} pp   "
          f"95% task-clustered CI [{lo*100:+.2f}, {hi*100:+.2f}] pp   "
          f"includes zero: {lo <= 0 <= hi}")
    print(f"SECONDARY exact McNemar   discordant grpo-wins={b} sft-wins={c}   "
          f"p = {mcnemar_exact(b, c):.4f}")


if __name__ == "__main__":
    main()
