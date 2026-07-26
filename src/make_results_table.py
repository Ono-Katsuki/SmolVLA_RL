"""Collect the 4-arm evaluation result.json files into a Markdown comparison table.

Reads the <eval_root>/<method>/<mode>/result.json files left by
scripts/eval_4arm.sh and emits a table of success_rate and CI (task
bootstrap) per method × mode. ci95_task is shown alongside the point estimate to
convey uncertainty, but it is each method's **marginal** interval and not a test
between methods. The table carries a note that method comparison uses the paired
contrast (task-clustered CI on per-task differences + exact McNemar).

Usage:
    python -m src.make_results_table --eval_root /content/eval_out --out table.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD_ORDER = ["sft", "rs_sft", "dpo", "grpo"]
METHOD_LABEL = {"sft": "SFT (baseline)", "rs_sft": "RS-SFT", "dpo": "flow-DPO", "grpo": "GRPO (ours)"}
MODE_ORDER = ["in_dist", "heldout"]
MODE_LABEL = {"in_dist": "in-dist", "heldout": "held-out (camera)"}


def _fmt(res: dict) -> str:
    """Format success_rate and the task-bootstrap CI as 'xx.x% [lo–hi]'."""
    if not res:
        return "—"
    p = res.get("success_rate")
    ci = res.get("ci95_task") or res.get("ci95_episode")
    n = res.get("n_total")
    nt = res.get("n_tasks")
    s = f"{p * 100:.1f}%"
    if ci:
        s += f" [{ci[0] * 100:.0f}–{ci[1] * 100:.0f}]"
    if n:
        s += f" (n={n}"
        if nt:
            s += f", {nt}task"
        s += ")"
    return s


def load(eval_root: Path) -> dict:
    out: dict = {}
    for method in METHOD_ORDER:
        for mode in MODE_ORDER:
            rf = eval_root / method / mode / "result.json"
            if rf.is_file():
                out[(method, mode)] = json.loads(rf.read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = load(args.eval_root)
    methods = [m for m in METHOD_ORDER if any((m, mo) in data for mo in MODE_ORDER)]

    lines = []
    lines.append("| Method | " + " | ".join(MODE_LABEL[m] for m in MODE_ORDER) + " |")
    lines.append("|---|" + "---|" * len(MODE_ORDER))
    for method in methods:
        row = [METHOD_LABEL.get(method, method)]
        for mode in MODE_ORDER:
            row.append(_fmt(data.get((method, mode), {})))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Success rate (task-bootstrap 95% CI, accounting for task correlation). in-dist and held-out are")
    lines.append("drawn from different task pools, so the gap between them confounds viewpoint sensitivity with task difficulty")
    lines.append("(note this is not pure viewpoint generalization).")
    lines.append("")
    lines.append("**The CIs in this table are not a test between methods.** They are each method's")
    lines.append("marginal interval: non-overlapping intervals do imply a significant difference, but")
    lines.append("overlapping ones do NOT imply the absence of one (reading non-significance off an")
    lines.append("overlap is a mistake). Judge between methods with the paired contrast the paper uses:")
    lines.append("the task-clustered bootstrap CI on per-task differences, paired on matching")
    lines.append("(task_id, init_state), together with the exact McNemar test on discordant pairs")
    lines.append("(implemented in analysis/analyze_unseen_prereg.py).")
    table = "\n".join(lines)

    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n")
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
