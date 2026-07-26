"""Produce a diagnostic summary from collect's episodes.jsonl.

Disentangles "is the reward bad / which stage fails / is there a GRPO signal":
  1. Stage-attainment funnel: P(reached), P(grasp|reached), P(lift|grasp), P(success|lift)
  2. Reward separation: success/failure mean rewards, gap, pairwise order agreement
     (pairwise order accuracy: fraction of success episodes scoring above
      failures. ~1.0 is ideal; below 0.7 the reward ordering does not match
      the end goal)
  3. GRPO signal: within-task reward std (tasks with small std produce no advantage)
  4. Chunk-weighting bias: mean chunk counts of successes/failures (long failures overweight negative advantage)

Usage:
    python -m src.analyze_episodes --episodes /content/rollout_data/episodes.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


W_SUCCESS = 0.25   # kept in sync with privileged.py. r_progress = staged_reward − W_SUCCESS×success


def _rate(num: int, den: int) -> str:
    return f"{num}/{den}={num / den:.2f}" if den else "n/a"


def _depth(e: dict) -> int:
    """Funnel attainment depth (0=before, 1=reached, 2=grasp_attempted, 3=lifted)."""
    return 3 if e.get("lifted") else 2 if e.get("grasped") else 1 if e.get("reached") else 0


def _progress(e: dict) -> float:
    """Dense reward with the success bonus removed.

    pairwise_order_accuracy includes +W_SUCCESS×success, so even 1.0 proves
    nothing beyond "successes were scored as successes" (external review
    comment). What GRPO actually needs is to rank failures against each other
    usefully even in groups with zero successes, so we measure agreement
    between funnel depth and reward ordering on failures only, bonus removed.
    """
    return e.get("staged_reward", 0.0) - W_SUCCESS * bool(e.get("success"))


def _failure_order_consistency(eps: list[dict]) -> float | None:
    """Fraction of failure pairs in the same set (differing depth) where the deeper one had higher r_progress.

    Note: this is a sanity check, not evidence of reward validity (external
    review comment). By weight design the per-depth r_progress ranges do not
    overlap (depth1 0.20-0.35 / depth2 0.40-0.55 / depth3 0.60-0.75;
    within-depth variation is only approach≤0.15), so if the monotone gating
    is healthy this is structurally 1.0. Anything below 1.0 means an
    inconsistency between the log and the reward code (e.g. a missed flag) —
    it is kept solely to detect that. Whether the reward ordering leads to
    future success can only be established via correlation with external
    quantities (e.g. max_lift_dz, not currently logged) or cross-checkpoint
    comparison."""
    fails = [e for e in eps if not e.get("success")]
    pairs = [
        (a, b) for a in fails for b in fails
        if _depth(a) > _depth(b) and a.get("task_key") == b.get("task_key")
    ]
    if not pairs:
        return None
    return sum(_progress(a) > _progress(b) for a, b in pairs) / len(pairs)


def _group_by_task(eps: list[dict]) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_task[e.get("task_key", "?")].append(e)
    return by_task


def analyze(rows: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_key", "?")].append(r)

    out: dict = {"per_task": {}, "n_episodes": len(rows)}
    for task, eps in sorted(by_task.items()):
        n = len(eps)
        reached = [e for e in eps if e.get("reached")]
        grasped = [e for e in eps if e.get("grasped")]
        lifted = [e for e in eps if e.get("lifted")]
        succ = [e for e in eps if e.get("success")]
        fail = [e for e in eps if not e.get("success")]
        rw = lambda es: sum(e.get("staged_reward", 0.0) for e in es) / len(es) if es else None  # noqa: E731
        rewards = [e.get("staged_reward", 0.0) for e in eps]
        mean_r = sum(rewards) / n if n else 0.0
        std_r = (sum((x - mean_r) ** 2 for x in rewards) / n) ** 0.5 if n else 0.0
        pairs = [(s.get("staged_reward", 0.0), f.get("staged_reward", 0.0)) for s in succ for f in fail]
        order_acc = sum(a > b for a, b in pairs) / len(pairs) if pairs else None
        out["per_task"][task] = {
            "n": n,
            "success_rate": round(len(succ) / n, 3) if n else 0.0,
            "funnel": {
                "P(reached)": _rate(len(reached), n),
                "P(grasp|reached)": _rate(sum(1 for e in reached if e.get("grasped")), len(reached)),
                "P(lift|grasp)": _rate(sum(1 for e in grasped if e.get("lifted")), len(grasped)),
                "P(success|lift)": _rate(sum(1 for e in lifted if e.get("success")), len(lifted)),
            },
            "reward_success_mean": round(rw(succ), 3) if succ else None,
            "reward_failure_mean": round(rw(fail), 3) if fail else None,
            "reward_gap": round(rw(succ) - rw(fail), 3) if succ and fail else None,
            "pairwise_order_accuracy": round(order_acc, 3) if order_acc is not None else None,
            "group_reward_std": round(std_r, 3),
            # Confirms a GRPO signal exists: identical rewards for everyone means zero advantage and no learning signal
            "n_unique_rewards": len({round(x, 4) for x in rewards}),
            "failure_order_consistency": (
                round(foa, 3) if (foa := _failure_order_consistency(eps)) is not None else None
            ),
            "mean_chunks_success": round(sum(e.get("n_chunks", 0) for e in succ) / len(succ), 1) if succ else None,
            "mean_chunks_failure": round(sum(e.get("n_chunks", 0) for e in fail) / len(fail), 1) if fail else None,
            "probe_ready_rate": _rate(sum(1 for e in eps if e.get("probe_ready")), n),
        }

    # Overall
    succ_all = [r for r in rows if r.get("success")]
    fail_all = [r for r in rows if not r.get("success")]
    pairs = [
        (s.get("staged_reward", 0.0), f.get("staged_reward", 0.0))
        for s in succ_all for f in fail_all
        if s.get("task_key") == f.get("task_key")   # compare within the same task only (avoids difficulty confounds)
    ]
    reached_all = [r for r in rows if r.get("reached")]
    grasped_all = [r for r in rows if r.get("grasped")]
    # Pairwise accuracy is only defined for tasks with both successes and failures.
    # State the number of tasks where it was definable, to avoid falsely reporting "1.0 on all tasks"
    tasks_with_pairs = sum(
        1 for t in out["per_task"].values() if t["pairwise_order_accuracy"] is not None
    )
    out["overall"] = {
        "success_rate": round(len(succ_all) / len(rows), 3) if rows else 0.0,
        "within_task_pairwise_order_accuracy": round(
            sum(a > b for a, b in pairs) / len(pairs), 3
        ) if pairs else None,
        "tasks_with_defined_pairwise_accuracy": f"{tasks_with_pairs}/{len(out['per_task'])}",
        "failure_order_consistency": (
            round(foa, 3) if (foa := _failure_order_consistency(rows)) is not None else None
        ),
        "funnel": {
            "P(reached)": _rate(len(reached_all), len(rows)),
            "P(grasp_attempted|reached)": _rate(
                sum(1 for e in reached_all if e.get("grasped")), len(reached_all)
            ),
            "P(lift|grasp_attempted)": _rate(
                sum(1 for e in grasped_all if e.get("lifted")), len(grasped_all)
            ),
            # effective_lift = lifted∪success. Because lift detection has false
            # negatives (success but lifted=False), the raw P(lift|grasp)
            # underestimates the progress conversion rate. For cross-iteration
            # comparison, primarily use this same-definition metric
            "P(lift_or_success|grasp_attempted)": _rate(
                sum(1 for e in grasped_all if e.get("lifted") or e.get("success")),
                len(grasped_all),
            ),
            "P(lifted|success)": _rate(
                sum(1 for e in succ_all if e.get("lifted")), len(succ_all)
            ),
        },
        # Unconditional progress rate. The conditional
        # P(lift_or_success|grasp_attempted) drops successes without the grasp
        # flag (observed on task79) from the denominator, so this is the
        # primary metric for cross-iteration comparison
        "P(lift_or_success)": _rate(
            sum(1 for e in rows if e.get("lifted") or e.get("success")), len(rows)
        ),
        # Two-stage detector coverage: audit both the grasp and lift detectors on successful trajectories
        "P(grasp_attempted|success)": _rate(
            sum(1 for e in succ_all if e.get("grasped")), len(succ_all)
        ),
        "success_without_grasp_flag": sum(1 for e in succ_all if not e.get("grasped")),
        # Success without the lifted flag = lift-detection false negative (z
        # threshold or a task that needs no lifting). If frequent, the funnel
        # interpretation (especially P(success|lift)) needs a caveat
        "success_without_lift_flag": sum(1 for e in succ_all if not e.get("lifted")),
        "mean_chunks_success": round(
            sum(r.get("n_chunks", 0) for r in succ_all) / len(succ_all), 1
        ) if succ_all else None,
        "mean_chunks_failure": round(
            sum(r.get("n_chunks", 0) for r in fail_all) / len(fail_all), 1
        ) if fail_all else None,
    }

    # For GRPO training logs (with an iter field), also emit per-iter trends:
    # shows which funnel stage learning is moving and whether reward separation improves
    if any("iter" in r for r in rows):
        by_iter: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            by_iter[int(r.get("iter", -1))].append(r)
        trend = {}
        for it, eps in sorted(by_iter.items()):
            n = len(eps)
            lifted = [e for e in eps if e.get("lifted")]
            reached = [e for e in eps if e.get("reached")]
            grasped = [e for e in eps if e.get("grasped")]
            # grasp rate ↑ with lift rate flat = close-command hacking; lift ↑
            # and success ↑ = the fix serves the objective; reward ↑ with lift
            # flat = mis-optimization of the dense reward
            trend[it] = {
                "n": n,
                "success_rate": round(sum(bool(e.get("success")) for e in eps) / n, 3),
                "reward_mean": round(sum(e.get("staged_reward", 0.0) for e in eps) / n, 3),
                "reward_progress_mean": round(sum(_progress(e) for e in eps) / n, 3),
                "P(reached)": round(len(reached) / n, 2),
                "P(grasp_attempted)": round(len(grasped) / n, 2),
                "P(lifted)": round(len(lifted) / n, 2),
                "P(lift_or_success)": round(
                    sum(1 for e in eps if e.get("lifted") or e.get("success")) / n, 2
                ),
                "P(success|lift)": round(
                    sum(1 for e in lifted if e.get("success")) / len(lifted), 2
                ) if lifted else None,
                # fraction of task groups where all rewards are identical (advantage exactly zero)
                "zero_advantage_group_fraction": round(
                    sum(
                        1 for tg in _group_by_task(eps).values()
                        if len({round(e.get("staged_reward", 0.0), 6) for e in tg}) == 1
                    ) / len(_group_by_task(eps)), 2
                ) if eps else None,
                "failure_order_consistency": (
                    round(foa, 3) if (foa := _failure_order_consistency(eps)) is not None else None
                ),
            }
        out["per_iter"] = trend
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.episodes.read_text().splitlines() if l.strip()]
    report = analyze(rows)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
