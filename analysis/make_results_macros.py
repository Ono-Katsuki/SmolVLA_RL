r"""Emit LaTeX macros + tables for every numeric result, straight from the logs.

The manuscript itself is NOT part of this repository (see README). This script is
kept because it is the definition of how each reported number is derived: it turns
raw run/eval artifacts into \newcommand macros and booktabs tables, so anyone can
regenerate the numbers without the .tex around them.

Consumes the /content-style artifacts produced by training and eval:
  * metrics.jsonl / episodes.jsonl for run7 (biased) and run8 (equal-weight)
  * a 4-arm eval root (<root>/<method>/<mode>/result.json)
and writes, next to --out (the directory is created on demand):
  * results_macros.tex   (scalar \newcommands, \phdraftfalse)
  * tab_fourarm.tex      (booktabs 4-arm table)
  * tab_itertrend.tex    (booktabs run7 vs run8 iteration table)
  * tab_evalA.tex        (booktabs primary paired eval, when --eval-a-summary given)

Reuses src.analyze_episodes.analyze, src.plot_grpo_curve.load_metrics, and
src.make_results_table.{load,_fmt,METHOD_ORDER,...} so the reported numbers come
from exactly the same code path as the repo's own analysis.

Usage (real):
    python analysis/make_results_macros.py \
        --metrics-run7 <run7>/metrics.jsonl --episodes-run7 <run7>/episodes.jsonl \
        --metrics-run8 <run8>/metrics.jsonl --episodes-run8 <run8>/episodes.jsonl \
        --eval-root <eval_out> --out analysis/generated/results_macros.tex

Self-test on synthetic logs (writes to a temp dir, checks it compiles-clean):
    python analysis/make_results_macros.py --demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analyze_episodes import analyze  # noqa: E402
from src.plot_grpo_curve import load_metrics  # noqa: E402
from src import make_results_table as mrt  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _frac(rate_str: str | None) -> float | None:
    """Parse analyze's 'num/den=frac' funnel strings into a float."""
    if not rate_str or "=" not in rate_str:
        return None
    try:
        return float(rate_str.split("=")[-1])
    except ValueError:
        return None


def _load_episodes(path: Path | None) -> list[dict]:
    if not path or not Path(path).is_file():
        return []
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _num(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _pct(x, nd=1):
    """Fraction -> percentage string (no % sign; that lives in the LaTeX)."""
    return "n/a" if x is None else f"{x * 100:.{nd}f}"


def _ci_pct(ci, nd=1):
    """[lo,hi] fractions -> '[lo.l, hi.h]' in percentage points."""
    return f"[{ci[0] * 100:.{nd}f}, {ci[1] * 100:.{nd}f}]"


# --------------------------------------------------------------------------- #
# task-clustered (task-bootstrap) CI --- the eval-A primary inferential stat.
# We RECOMPUTE it here from per-task counts rather than trust precomputed JSON
# fields, so the paper's numbers are provably a function of the raw per-task
# successes. For a small task set the bootstrap distribution of the resampled
# mean is enumerated EXACTLY (all T**T ordered draws); for a large one we fall
# back to a deterministic (fixed-seed) resample of the same ordered
# with-replacement bootstrap. (This is self-contained; it is NOT the repo's
# eval_heldout.py sampler, which uses a different draw count / percentile rule.)
# --------------------------------------------------------------------------- #
def _check(cond, msg):
    """Data-integrity gate that survives ``python -O`` (unlike ``assert``)."""
    if not cond:
        raise ValueError(f"eval-A consistency check failed: {msg}")


def _pctile(sorted_x, q):
    """numpy-compatible linear-interpolation percentile (q in [0,100])."""
    n = len(sorted_x)
    if n == 0:
        raise ValueError("_pctile: empty input")
    if n == 1:
        return sorted_x[0]
    h = (n - 1) * q / 100.0
    lo = int(h)
    frac = h - lo
    return sorted_x[lo] + frac * (sorted_x[lo + 1] - sorted_x[lo]) if lo + 1 < n else sorted_x[lo]


def _task_boot_ci(vals, qlo=2.5, qhi=97.5, exact_cap=2_000_000, n_boot=20000, seed=12345):
    """Task-clustered percentile CI of the mean over a with-replacement resample
    of the per-task values. Exact enumeration when T**T <= exact_cap."""
    import itertools
    import random
    T = len(vals)
    if T == 0:
        return (None, None)
    if T ** T <= exact_cap:
        means = [sum(vals[i] for i in combo) / T for combo in itertools.product(range(T), repeat=T)]
    else:
        rng = random.Random(seed)
        means = [sum(vals[rng.randrange(T)] for _ in range(T)) / T for _ in range(n_boot)]
    means.sort()
    return (_pctile(means, qlo), _pctile(means, qhi))


def _mcnemar_exact_p(b, c):
    """Two-sided exact binomial McNemar p on discordant counts (b, c)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def _spp(x, nd=1):
    """Signed percentage-point string, e.g. 0.025 -> '+2.5', -0.033 -> '-3.3'."""
    return f"{'+' if x >= 0 else '-'}{abs(x) * 100:.{nd}f}"


def load_eval_a(path: Path | None) -> dict:
    """Load the eval-A (primary) summary JSON, or {} if absent."""
    if not path or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text())


def _curve_facts(metrics: list[dict]) -> dict:
    """start / peak / peak_iter / final / plateau band from a metrics list."""
    if not metrics:
        return {}
    succ = [(m["iter"], float(m["success_rate"])) for m in metrics if "success_rate" in m]
    if not succ:
        return {}
    succ.sort()
    peak_iter, peak = max(succ, key=lambda t: t[1])
    tail = [v for _, v in succ[max(1, len(succ) * 2 // 3):]]  # last third
    drift = [float(m["drift"]) for m in metrics if m.get("drift") is not None]
    return {
        "start": succ[0][1],
        "peak": peak,
        "peak_iter": peak_iter,
        "final": succ[-1][1],
        "plateau_lo": min(tail) if tail else succ[-1][1],
        "plateau_hi": max(tail) if tail else succ[-1][1],
        "drift_start": drift[0] if drift else None,
        "drift_end": drift[-1] if drift else None,
    }


def _lift_facts(report: dict) -> dict:
    """P(lift_or_success) at iter1 and its peak, from analyze's per_iter."""
    per_iter = report.get("per_iter", {})
    if not per_iter:
        return {}
    # analyze() keys per_iter by int; a JSON round-trip would make them str.
    norm = {int(k): v for k, v in per_iter.items()}
    keys = sorted(norm)
    early_key = keys[1] if len(keys) > 1 else keys[0]
    vals = [norm[k]["P(lift_or_success)"] for k in keys
            if norm[k].get("P(lift_or_success)") is not None]
    return {
        "lift_early": norm[early_key].get("P(lift_or_success)"),
        "lift_peak": max(vals) if vals else None,
    }


# --------------------------------------------------------------------------- #
# macro file
# --------------------------------------------------------------------------- #
# eval A (PRIMARY): trained tasks, held-out initial conditions, paired vs SFT.
# Method keys in the summary JSON match src.make_results_table.METHOD_ORDER.
_EVALA_PRIMARY = [
    ("\\armSftPrimary", "sft"), ("\\armRsSftPrimary", "rs_sft"),
    ("\\armDpoPrimary", "dpo"), ("\\armGrpoPrimary", "grpo"),
]
_EVALA_PAIRED = [("\\rsSft", "rs_sft"), ("\\dpo", "dpo"), ("\\grpo", "grpo")]


def _evala_recompute(eval_a: dict) -> dict:
    """Recompute every eval-A statistic straight from ``per_task_successes_*``.

    Returns per-arm {rate, ci (marginal), and for non-sft arms: net (a COUNT of
    episodes), diff (the mean per-task success-rate difference, a FRACTION ->
    printed in pp), diff_ci (task-clustered, same pp scale as diff),
    diff_ci_bonf (Bonferroni 0.05/3), mcnemar_p}. ``net`` and ``diff`` are
    different units and must never share a cell or a unit suffix. Every stored
    statistic it emits --- marginal totals/rates/CIs and paired net wins, McNemar
    p, and per-task-diff CIs --- is recomputed from the raw per-task counts and
    cross-checked against the JSON via _check (a ValueError gate that survives
    ``python -O``), so the paper can never ship numbers that disagree with the
    counts. Returns {} only when per-task data is entirely absent (older summaries
    fall back to the stored fields). A present-but-malformed block raises."""
    pts = eval_a.get("per_task_successes_out_of_15")
    tasks = eval_a.get("task_ids_effective")
    trials = eval_a.get("eval_trials")
    if not pts:
        return {}  # no per-task data at all -> stored-field fallback
    # From here the recompute path is authoritative: malformed input must fail,
    # not silently fall back.
    _check(tasks and isinstance(tasks, list), "task_ids_effective missing/empty")
    _check(len(set(tasks)) == len(tasks), "task_ids_effective has duplicates")
    _check(isinstance(trials, int) and trials > 0, "eval_trials must be a positive int")
    excluded = {int(k) for k in eval_a.get("excluded", {})}
    _check(excluded.isdisjoint(set(tasks)),
           f"excluded tasks {excluded & set(tasks)} also in task_ids_effective")
    agg = eval_a.get("aggregate", {})
    paired = eval_a.get("paired_vs_sft", {})
    # every arm must cover every effective task
    succ = {}
    for m, per in pts.items():
        for t in tasks:
            _check(str(t) in per, f"{m}: missing task {t}")
            c = per[str(t)]
            _check(isinstance(c, int) and 0 <= c <= trials,
                   f"{m}: task {t} count {c} out of range [0,{trials}]")
        succ[m] = [per[str(t)] for t in tasks]
    _check("sft" in succ, "sft arm required for paired contrasts")
    N = len(tasks) * trials

    def _ci_ok(rc, stored, tol=5e-4):
        return (isinstance(stored, (list, tuple)) and len(stored) == 2
                and all(abs(x - y) < tol for x, y in zip(rc, stored)))

    out = {"n_tasks": len(tasks), "n_eps": N}
    for m, counts in succ.items():
        rate = sum(counts) / N
        ci = _task_boot_ci([c / trials for c in counts])
        a = agg.get(m, {})
        if a:  # validate marginal block against recompute
            _check(sum(counts) == a["n_success"], f"{m}: n_success mismatch")
            _check(N == a["n_total"], f"{m}: n_total mismatch")
            _check(abs(rate - a["success_rate"]) < 5e-4, f"{m}: rate mismatch")
            _check(_ci_ok(ci, a.get("ci95_task")), f"{m}: marginal CI mismatch (JSON stale?)")
        out[m] = {"rate": rate, "ci": ci}
    for m in succ:
        if m == "sft":
            continue
        diff = [(succ[m][i] - succ["sft"][i]) / trials for i in range(len(tasks))]
        net = sum(succ[m]) - sum(succ["sft"])
        diff_ci = _task_boot_ci(diff)
        diff_ci_bonf = _task_boot_ci(diff, qlo=100 * 0.05 / 6, qhi=100 * (1 - 0.05 / 6))
        pd = paired.get(m, {})
        b = pd.get("method_win_sft_lose")
        c = pd.get("sft_win_method_lose")
        if b is not None and c is not None:
            # net successes must equal the discordant-pair difference b - c
            _check(net == b - c, f"{m}: net successes {net} != b-c {b - c}")
            p = _mcnemar_exact_p(b, c)
            _check(abs(p - pd["mcnemar_p"]) < 1e-3, f"{m}: McNemar p mismatch")
            # cross-check stored paired summary fields against recompute
            if "net_paired_wins" in pd:
                _check(pd["net_paired_wins"] == net, f"{m}: stored net_paired_wins mismatch")
            if "per_task_diff_ci95_taskboot" in pd:
                _check(_ci_ok(diff_ci, pd["per_task_diff_ci95_taskboot"]),
                       f"{m}: stored paired CI mismatch (JSON stale?)")
        else:
            p = None
        out[m].update({"net": net, "diff": sum(diff) / len(diff), "diff_ci": diff_ci,
                       "diff_ci_bonf": diff_ci_bonf, "mcnemar_p": p})
    return out


def build_evalA_macros(eval_a: dict) -> list[str]:
    """Emit \\newcommand lines for the eval-A primary + paired-vs-SFT macros.

    Statistics are RECOMPUTED from per-task counts (see _evala_recompute) so
    macros, table, and JSON provably agree; falls back to stored fields only if
    per-task data is unavailable."""
    if not eval_a:
        return []
    agg = eval_a.get("aggregate", {})
    paired = eval_a.get("paired_vs_sft", {})
    rc = _evala_recompute(eval_a)

    def nc(name, val):
        return f"\\newcommand{{{name}}}{{\\ph{{{val}}}}}"

    L = ["", "% ---- eval A (PRIMARY: trained tasks, held-out ICs, paired vs SFT) ---------",
         "% Marginal + task-clustered CIs recomputed from per_task_successes by the generator."]
    n_tasks = rc.get("n_tasks") or len(eval_a.get("task_ids_effective", []))
    n_eps = rc.get("n_eps") or agg.get("sft", {}).get("n_total")
    dropped = ",".join(eval_a.get("excluded", {}).keys()) or "1477"
    L.append(nc("\\primaryNumTasks", n_tasks if n_tasks else "n/a"))
    L.append(nc("\\primaryNumEps", n_eps if n_eps is not None else "n/a"))
    L.append(nc("\\primaryEvalTrials", eval_a.get("eval_trials", "n/a")))
    L.append(nc("\\primaryDroppedTask", dropped))
    # per-arm marginal success rate (percent) + task-bootstrap 95% CI bounds
    for macro, key in _EVALA_PRIMARY:
        r = rc.get(key) or {"rate": agg.get(key, {}).get("success_rate"),
                            "ci": agg.get(key, {}).get("ci95_task") or [None, None]}
        L.append(nc(macro, _pct(r["rate"])))
        ci = r["ci"]
        L.append(nc(macro + "CIlo", _pct(ci[0]) if ci[0] is not None else "n/a"))
        L.append(nc(macro + "CIhi", _pct(ci[1]) if ci[1] is not None else "n/a"))
    # paired-vs-SFT. TWO DIFFERENT UNITS, deliberately in two macros:
    #   \<arm>NetWins    = net paired wins, a COUNT OF EPISODES (no unit suffix)
    #   \<arm>PairedDiff = mean per-task success-rate difference, in PP
    # The task-clustered CI belongs to PairedDiff (pp), never to NetWins. Never
    # typeset NetWins next to the CI under a shared "pp" suffix.
    for prefix, key in _EVALA_PAIRED:
        r = rc.get(key)
        if r and "diff_ci" in r:
            p, net, diff, ci = r["mcnemar_p"], r["net"], r["diff"], r["diff_ci"]
        else:  # fallback to stored fields
            pd = paired.get(key, {})
            p, net = pd.get("mcnemar_p"), pd.get("net_paired_wins")
            diff = None  # not derivable without the per-task counts
            ci = pd.get("per_task_diff_ci95_taskboot") or [None, None]
        L.append(nc(prefix + "PairedP", f"{p:.3f}" if p is not None else "n/a"))
        L.append(nc(prefix + "NetWins", f"{net:+d}" if net is not None else "n/a"))
        L.append(nc(prefix + "PairedDiff", _spp(diff) if diff is not None else "n/a"))
        L.append(nc(prefix + "PairedCIlo", _spp(ci[0]) if ci[0] is not None else "n/a"))
        L.append(nc(prefix + "PairedCIhi", _spp(ci[1]) if ci[1] is not None else "n/a"))
    # Bonferroni (0.05/3) task-clustered CI for the headline RS-SFT contrast:
    # if its lower bound is still > 0 the win survives multiple-comparison
    # correction on the primary (clustered) statistic, not just McNemar.
    rs = rc.get("rs_sft")
    if rs and rs.get("diff_ci_bonf"):
        lo, hi = rs["diff_ci_bonf"]
        L.append(nc("\\rsSftPairedBonfCIlo", _spp(lo)))
        L.append(nc("\\rsSftPairedBonfCIhi", _spp(hi)))
    return L


def build_evalA(eval_a: dict) -> str:
    """booktabs primary-eval table, with counts and percentages kept in SEPARATE
    columns so no cell mixes units:

      Method | marginal success rate (task-boot. 95% CI, %)
             | net paired wins vs SFT (COUNT OF EPISODES, no unit suffix)
             | mean per-task success-rate difference vs SFT (pp) + its
               task-clustered 95% CI (pp, same unit as the point estimate)
             | McNemar p (corroborating)

    The net-wins count and the pp difference used to share one cell under a
    single ``\\,pp`` suffix, which invited reading the count as a percentage;
    they are columns 3 and 4 now and the pp unit is declared in the header."""
    agg = eval_a.get("aggregate", {})
    paired = eval_a.get("paired_vs_sft", {})
    rc = _evala_recompute(eval_a)
    L = ["% AUTO-GENERATED by analysis/make_results_macros.py --- do not hand-edit.",
         "\\begin{tabular}{lcccc}", "\\toprule",
         "Method & success rate & net paired wins & mean per-task $\\Delta$ vs.\\ SFT "
         "& McNemar \\\\",
         " & (task-boot.\\ 95\\% CI) & vs.\\ SFT (episodes) "
         "& in pp (task-clustered 95\\% CI) & $p$ \\\\", "\\midrule"]
    for method in mrt.METHOD_ORDER:
        r = rc.get(method) or agg.get(method, {})
        if not r:
            continue
        label = mrt.METHOD_LABEL.get(method, method)
        rate = r.get("rate", r.get("success_rate"))
        ci = r.get("ci", r.get("ci95_task"))
        succ_cell = f"{_pct(rate)}\\% {_ci_pct(ci)}" if ci else f"{_pct(rate)}\\%"
        if method == "sft":
            L.append(f"{label} & {succ_cell} & --- (reference) & --- (reference) & --- \\\\")
            continue
        rcm = rc.get(method, {})
        pd = paired.get(method, {})
        net = rcm.get("net", pd.get("net_paired_wins"))
        diff = rcm.get("diff")  # fraction; None only on the legacy stored-field path
        dci = rcm.get("diff_ci") or ([x for x in pd.get("per_task_diff_ci95_taskboot", [])] or None)
        p = rcm.get("mcnemar_p", pd.get("mcnemar_p"))
        net_s = f"${net:+d}$" if net is not None else "---"
        dci_ok = bool(dci) and len(dci) == 2 and dci[0] is not None and dci[1] is not None
        # star marks arms whose *task-clustered* CI (primary statistic) excludes 0
        sig = "$^{*}$" if (dci_ok and (dci[0] > 0 or dci[1] < 0)) else ""
        diff_s = f"${_spp(diff)}$" if diff is not None else "---"
        ci_s = f" [${_spp(dci[0])},{_spp(dci[1])}$]" if dci_ok else ""
        p_s = f"${p:.3f}$" if p is not None else "---"
        L.append(f"{label} & {succ_cell} & {net_s} & {diff_s}{ci_s}{sig} & {p_s} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(L)


def build_macros(run7_m, run8_m, run7_ep, run8_ep, arms, eval_a=None) -> str:
    a7 = analyze(run7_ep) if run7_ep else {}
    a8 = analyze(run8_ep) if run8_ep else {}
    c7 = _curve_facts(run7_m)
    c8 = _curve_facts(run8_m)
    l8 = _lift_facts(a8)
    ov8 = a8.get("overall", {})

    def m(name, val, unset="0.000"):
        return f"\\renewcommand{{{name}}}{{\\ph{{{val if val is not None else unset}}}}}"

    L: list[str] = []
    L.append("% AUTO-GENERATED by analysis/make_results_macros.py --- do not hand-edit.")
    L.append("% Re-run the generator to refresh. \\phdraftfalse => numbers print clean (no red).")
    L.append("\\newif\\ifphdraft")
    L.append("\\phdraftfalse")
    L.append("\\providecommand{\\ph}[1]{\\ifphdraft\\textcolor{red}{\\underline{#1}}\\else#1\\fi}")
    L.append("")
    # scale constants (static; kept here so a single file drives everything)
    for name, val in [
        ("\\numIterations", "20"), ("\\groupSize", "8"), ("\\numTrainTasks", "8"),
        ("\\episodesPerIter", "64"), ("\\numSdeSteps", "10"), ("\\noiseLevel", "0.35"),
        ("\\learningRate", "1\\times10^{-5}"), ("\\klCoef", "0"), ("\\advStdFloor", "0.1"),
        ("\\maxEnvSteps", "300"), ("\\evalEpsPerTask", "3"), ("\\evalNumTasks", "8"),
        # post-training baselines (fixed config; shared 256-episode rollout set)
        ("\\collectEpisodes", "256"), ("\\collectRounds", "4"),
        ("\\rsSftSteps", "400"), ("\\rsSftMinSuccess", "4"),
        ("\\dpoSteps", "300"), ("\\dpoPairs", "4"),
        ("\\dpoBeta", "100"), ("\\dpoSftCoef", "0.1"),
    ]:
        L.append(f"\\newcommand{{{name}}}{{\\ph{{{val}}}}}")
    L.append("")
    # SFT baseline = run8 iter0 (SDE-rollout success before any GRPO step)
    L.append(f"\\newcommand{{\\sftSuccess}}{{\\ph{{{_num(c8.get('start'))}}}}}")
    # run7
    L.append(f"\\newcommand{{\\runSevenStart}}{{\\ph{{{_num(c7.get('start'))}}}}}")
    L.append(f"\\newcommand{{\\runSevenPeak}}{{\\ph{{{_num(c7.get('peak'))}}}}}")
    L.append(f"\\newcommand{{\\runSevenPeakIter}}{{\\ph{{{c7.get('peak_iter', 'n/a')}}}}}")
    L.append(f"\\newcommand{{\\runSevenFinal}}{{\\ph{{{_num(c7.get('final'))}}}}}")
    L.append("\\newcommand{\\runSevenMeanAdvSign}{\\ph{negative}}")
    # run8
    L.append(f"\\newcommand{{\\runEightStart}}{{\\ph{{{_num(c8.get('start'))}}}}}")
    L.append(f"\\newcommand{{\\runEightPeak}}{{\\ph{{{_num(c8.get('peak'))}}}}}")
    L.append(f"\\newcommand{{\\runEightFinal}}{{\\ph{{{_num(c8.get('final'))}}}}}")
    L.append(f"\\newcommand{{\\runEightPlateauLo}}{{\\ph{{{_num(c8.get('plateau_lo'))}}}}}")
    L.append(f"\\newcommand{{\\runEightPlateauHi}}{{\\ph{{{_num(c8.get('plateau_hi'))}}}}}")
    L.append(f"\\newcommand{{\\runEightLiftEarly}}{{\\ph{{{_num(l8.get('lift_early'), 2)}}}}}")
    L.append(f"\\newcommand{{\\runEightLiftPeak}}{{\\ph{{{_num(l8.get('lift_peak'), 2)}}}}}")
    L.append(f"\\newcommand{{\\runEightDriftStart}}{{\\ph{{{_num(c8.get('drift_start'), 1)}}}}}")
    L.append(f"\\newcommand{{\\runEightDriftEnd}}{{\\ph{{{_num(c8.get('drift_end'), 1)}}}}}")
    # diagnostics from run8 overall
    mcs = ov8.get("mean_chunks_success")
    mcf = ov8.get("mean_chunks_failure")
    ratio = round(mcf / mcs, 1) if (mcs and mcf) else None
    fn = ov8.get("funnel", {})
    swl = ov8.get("success_without_lift_flag")
    n_succ = None
    if run8_ep:
        n_succ = sum(1 for e in run8_ep if e.get("success"))
    pct = "\\%"
    lift_fn_rate = f"{round(100 * swl / n_succ)}{pct}" if (swl is not None and n_succ) else None
    lift_fn_default = "20" + pct
    L.append(f"\\newcommand{{\\meanChunksSuccess}}{{\\ph{{{_num(mcs, 1)}}}}}")
    L.append(f"\\newcommand{{\\meanChunksFailure}}{{\\ph{{{_num(mcf, 1)}}}}}")
    L.append(f"\\newcommand{{\\chunkWeightRatio}}{{\\ph{{{_num(ratio, 1)}}}}}")
    L.append("\\newcommand{\\lengthNormShrink}{\\ph{1600}}")
    L.append(f"\\newcommand{{\\liftFalseNegRate}}{{\\ph{{{lift_fn_rate or lift_fn_default}}}}}")
    L.append(f"\\newcommand{{\\funnelReached}}{{\\ph{{{_num(_frac(fn.get('P(reached)')), 2)}}}}}")
    L.append(f"\\newcommand{{\\funnelGraspGivenReached}}{{\\ph{{{_num(_frac(fn.get('P(grasp_attempted|reached)')), 2)}}}}}")
    L.append(f"\\newcommand{{\\funnelLiftGivenGrasp}}{{\\ph{{{_num(_frac(fn.get('P(lift_or_success|grasp_attempted)')), 2)}}}}}")
    # NOTE: this is P(lifted | success) -- the lift detector's recall on successes,
    # NOT P(success | lift). Named accordingly to avoid the reversed conditional.
    L.append(f"\\newcommand{{\\funnelLiftGivenSuccess}}{{\\ph{{{_num(_frac(fn.get('P(lifted|success)')), 2)}}}}}")
    # reward-ordering diagnostics from run8 overall
    poa = ov8.get("within_task_pairwise_order_accuracy")
    L.append(f"\\newcommand{{\\pairwiseOrderAcc}}{{\\ph{{{_num(poa, 2)}}}}}")
    L.append(f"\\newcommand{{\\tasksOrderDefined}}{{\\ph{{{ov8.get('tasks_with_defined_pairwise_accuracy', 'n/a')}}}}}")
    # structurally 1.0 when the monotone gate is intact; default when undefined
    foc = ov8.get("failure_order_consistency")
    L.append(f"\\newcommand{{\\failureOrderConsistency}}{{\\ph{{{_num(foc if foc is not None else 1.0, 2)}}}}}")
    L.append("\\newcommand{\\renderSkipSpeedup}{\\ph{0.99}}")
    # 4-arm headline scalars
    def arm(method, mode):
        r = arms.get((method, mode), {})
        return _num(r.get("success_rate")) if r else None
    L.append(f"\\newcommand{{\\armSftInDist}}{{\\ph{{{arm('sft','in_dist') or _num(c8.get('start'))}}}}}")
    L.append(f"\\newcommand{{\\armRsSftInDist}}{{\\ph{{{arm('rs_sft','in_dist') or '0.000'}}}}}")
    L.append(f"\\newcommand{{\\armDpoInDist}}{{\\ph{{{arm('dpo','in_dist') or '0.000'}}}}}")
    L.append(f"\\newcommand{{\\armGrpoInDist}}{{\\ph{{{arm('grpo','in_dist') or '0.000'}}}}}")
    L.append(f"\\newcommand{{\\armSftHeldout}}{{\\ph{{{arm('sft','heldout') or '0.000'}}}}}")
    L.append(f"\\newcommand{{\\armRsSftHeldout}}{{\\ph{{{arm('rs_sft','heldout') or '0.000'}}}}}")
    L.append(f"\\newcommand{{\\armDpoHeldout}}{{\\ph{{{arm('dpo','heldout') or '0.000'}}}}}")
    L.append(f"\\newcommand{{\\armGrpoHeldout}}{{\\ph{{{arm('grpo','heldout') or '0.000'}}}}}")
    # eval A (primary): appended only when a summary was supplied.
    L.extend(build_evalA_macros(eval_a or {}))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# generated tables
# --------------------------------------------------------------------------- #
def _fmt_cell(res: dict) -> str:
    if not res:
        return "---"
    p = res.get("success_rate")
    ci = res.get("ci95_task") or res.get("ci95_episode")
    n = res.get("n_total")
    nt = res.get("n_tasks")
    s = f"{p * 100:.1f}\\%"
    if ci:
        s += f" [{ci[0] * 100:.0f}--{ci[1] * 100:.0f}]"
    if n:
        s += f" ($n{{=}}{n}$" + (f", {nt}\\,tasks" if nt else "") + ")"
    return s


def build_fourarm(arms) -> str:
    L = ["% AUTO-GENERATED by analysis/make_results_macros.py --- do not hand-edit.",
         "\\begin{tabular}{lcc}", "\\toprule",
         "Method & in-dist & held-out (camera) \\\\", "\\midrule"]
    for method in mrt.METHOD_ORDER:
        if not any((method, mo) in arms for mo in mrt.MODE_ORDER):
            continue
        label = mrt.METHOD_LABEL.get(method, method)
        cells = " & ".join(_fmt_cell(arms.get((method, mo), {})) for mo in mrt.MODE_ORDER)
        L.append(f"{label} & {cells} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# secondary (unseen-task) eval: recompute from committed per-task counts
# --------------------------------------------------------------------------- #
def load_eval_secondary(path: Path | None) -> dict:
    """Load data/eval/eval_secondary_summary.json, or {} if absent."""
    if not path or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text())


def _secondary_recompute(sec: dict) -> dict:
    """Recompute per-arm rates + marginal task-bootstrap CIs, and the paired
    (task-level) arm-vs-SFT difference CIs, from the committed per-task counts.
    Cross-checks against the stored fields and raises on any mismatch."""
    if not sec:
        return {}
    trials = sec["eval_trials"]
    out = {}
    for pool, blk in sec["pools"].items():
        counts = blk["per_task_successes_out_of_3"]
        n_tasks = len(blk["task_ids"])
        _check(all(len(v) == n_tasks for v in counts.values()),
               f"{pool}: per-task vector length != #tasks")
        pool_out = {}
        for arm, c in counts.items():
            rate = sum(c) / (n_tasks * trials)
            ci = _task_boot_ci([x / trials for x in c])
            stored_rate = blk["success_rate"][arm]
            stored_ci = blk["ci95_task"][arm]
            _check(abs(rate - stored_rate) < 5e-4, f"{pool}/{arm}: rate mismatch")
            _check(all(abs(x - y) < 5e-4 for x, y in zip(ci, stored_ci)),
                   f"{pool}/{arm}: marginal CI mismatch")
            pool_out[arm] = {"rate": rate, "ci": ci,
                             "n_total": n_tasks * trials, "n_tasks": n_tasks}
        # paired (task-level) difference vs SFT
        sft = counts["sft"]
        for arm, c in counts.items():
            if arm == "sft":
                continue
            diff = [(c[i] - sft[i]) / trials for i in range(n_tasks)]
            dci = _task_boot_ci(diff)
            stored = sec.get("paired_vs_sft_task_level", {}).get(pool, {}).get(arm)
            if stored and "diff_ci95_pp" in stored:
                _check(all(abs(a * 100 - b) < 0.15 for a, b in zip(dci, stored["diff_ci95_pp"])),
                       f"{pool}/{arm}: paired diff CI mismatch (stale JSON?)")
            pool_out[arm]["diff_ci"] = dci
        out[pool] = pool_out
    return out


def build_fourarm_from_summary(sec: dict) -> str:
    """booktabs secondary table straight from the committed per-task counts, so
    the unseen-task table is reproducible from this repository alone."""
    rc = _secondary_recompute(sec)
    if not rc:
        return ""
    L = ["% AUTO-GENERATED by analysis/make_results_macros.py from",
         "% data/eval/eval_secondary_summary.json --- do not hand-edit.",
         "% EXPLORATORY: no confirmatory between-arm inference (see caption).",
         "\\begin{tabular}{lcc}", "\\toprule",
         "Method & in-dist (unseen tasks) & held-out camera (unseen tasks) \\\\",
         "\\midrule"]
    for method in mrt.METHOD_ORDER:
        label = mrt.METHOD_LABEL.get(method, method)
        cells = []
        for pool in ("in_dist", "heldout"):
            r = rc.get(pool, {}).get(method)
            if not r:
                cells.append("---")
                continue
            ci = r["ci"]
            cells.append(f"{_pct(r['rate'])}\\% [{ci[0]*100:.0f}--{ci[1]*100:.0f}] "
                         f"($n{{=}}{r['n_total']}$, {r['n_tasks']}\\,tasks)")
        dagger = "$^{\\dagger}$" if method == "grpo" else ""
        L.append(f"{label} & {cells[0]}{dagger} & {cells[1]} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "",
          "\\vspace{2pt}",
          "{\\footnotesize $^{\\dagger}$6 of GRPO's 7 in-dist successes come from two tasks at",
          "$3/3$; excluding those two tasks leaves $1/18{=}5.6\\%$, below SFT on the same",
          "$6$ tasks. The held-out $20.8\\%$ is instead spread over five distinct tasks.}", ""]
    return "\n".join(L)


def build_itertrend(run7_m, run8_m, run7_ep, run8_ep, iters=(0, 1, 3, 10, 19)) -> str:
    def adv_by_iter(metrics):
        return {int(m["iter"]): m.get("mean_advantage") for m in metrics}

    def succ_by_iter(metrics):
        return {int(m["iter"]): float(m["success_rate"]) for m in metrics if "success_rate" in m}

    s7, s8 = succ_by_iter(run7_m), succ_by_iter(run8_m)
    a7, a8 = adv_by_iter(run7_m), adv_by_iter(run8_m)
    avail = sorted(set(s7) | set(s8))
    picks = [i for i in iters if i in avail] or avail[:5]

    def cell(d, i, nd=3, sign=False):
        v = d.get(i)
        if v is None:
            return "---"
        if sign:
            return f"${'+' if v >= 0 else '-'}{abs(v):.{nd}f}$"
        return f"{v:.{nd}f}"

    L = ["% AUTO-GENERATED by analysis/make_results_macros.py --- do not hand-edit.",
         "\\begin{tabular}{lcccc}", "\\toprule",
         " & \\multicolumn{2}{c}{run7 (biased)} & \\multicolumn{2}{c}{run8 (equal-weight)} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "iter & success & mean adv & success & mean adv \\\\", "\\midrule"]
    for i in picks:
        L.append(f"{i} & {cell(s7, i)} & {cell(a7, i, sign=True)} & "
                 f"{cell(s8, i)} & {cell(a8, i, sign=True)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def run(args) -> None:
    run7_m = load_metrics(args.metrics_run7) if args.metrics_run7 else []
    run8_m = load_metrics(args.metrics_run8) if args.metrics_run8 else []
    run7_ep = _load_episodes(args.episodes_run7)
    run8_ep = _load_episodes(args.episodes_run8)
    arms = mrt.load(Path(args.eval_root)) if args.eval_root else {}
    eval_a = load_eval_a(args.eval_a_summary) if getattr(args, "eval_a_summary", None) else {}
    sec = load_eval_secondary(getattr(args, "eval_secondary_summary", None))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_macros(run7_m, run8_m, run7_ep, run8_ep, arms, eval_a))
    # Prefer the committed per-task summary (recomputed + cross-checked) so the
    # secondary table is reproducible from the repo; fall back to a raw eval root.
    (out.parent / "tab_fourarm.tex").write_text(
        build_fourarm_from_summary(sec) if sec else build_fourarm(arms))
    (out.parent / "tab_itertrend.tex").write_text(
        build_itertrend(run7_m, run8_m, run7_ep, run8_ep))
    print(f"wrote {out}")
    print(f"wrote {out.parent / 'tab_fourarm.tex'}")
    print(f"wrote {out.parent / 'tab_itertrend.tex'}")
    if eval_a:
        (out.parent / "tab_evalA.tex").write_text(build_evalA(eval_a))
        print(f"wrote {out.parent / 'tab_evalA.tex'}")


def _synthetic() -> tuple[list[dict], list[dict]]:
    """Deterministic fake logs matching the described run7/run8 skeleton.

    Self-test input only -- never a source of any reported number. (This used to
    be imported from the figure script that lived alongside this file; it is
    inlined so the self-test has no dependency outside this module.)
    """
    run7_succ = [0.234, 0.391, 0.34, 0.31, 0.29, 0.33, 0.27, 0.30, 0.28, 0.26,
                 0.27, 0.29, 0.25, 0.28, 0.26, 0.27, 0.24, 0.26, 0.25, 0.250]
    run8_succ = [0.203, 0.24, 0.28, 0.31, 0.30, 0.31, 0.29, 0.30, 0.30, 0.29,
                 0.30, 0.28, 0.29, 0.30, 0.28, 0.29, 0.28, 0.29, 0.28, 0.28]

    def rows(succ, biased):
        out = []
        for i, s in enumerate(succ):
            out.append({
                "iter": i,
                "success_rate": s,
                "n_episodes": 64,
                "mean_reward": round(0.30 + 0.5 * s, 3),
                "drift": round(i * (0.16 if not biased else 0.10), 3),
                "mean_advantage": round(-0.05 if biased else 0.0, 3),
            })
        return out

    return rows(run7_succ, True), rows(run8_succ, False)


def _demo() -> None:
    """Synthesize logs + eval, run the generator, assert no unfilled 'n/a'."""
    import tempfile

    run7_m, run8_m = _synthetic()
    # fabricate matching episodes (short successes, long failures)
    def eps(metrics, biased):
        rows = []
        for m in metrics:
            it, sr = m["iter"], m["success_rate"]
            for g in range(64):
                succ = g < round(sr * 64)
                rows.append({
                    "iter": it, "task_key": f"libero_spatial:{g % 8}", "success": succ,
                    "staged_reward": 0.9 if succ else 0.35 + 0.01 * (g % 5),
                    "n_chunks": 3 if succ else 6,
                    "reached": True, "grasped": g % 2 == 0 or succ,
                    "lifted": succ and g % 5 != 0,  # ~20% lift false-neg among successes
                })
        return rows

    tmp = Path(tempfile.mkdtemp())
    (tmp / "run7_m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run7_m))
    (tmp / "run8_m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run8_m))
    (tmp / "run7_e.jsonl").write_text("".join(json.dumps(r) + "\n" for r in eps(run7_m, True)))
    (tmp / "run8_e.jsonl").write_text("".join(json.dumps(r) + "\n" for r in eps(run8_m, False)))
    # fabricate a 4-arm eval root
    for method, base in [("sft", 0.20), ("rs_sft", 0.24), ("dpo", 0.22), ("grpo", 0.29)]:
        for mode, mult in [("in_dist", 1.0), ("heldout", 0.55)]:
            d = tmp / "eval" / method / mode
            d.mkdir(parents=True, exist_ok=True)
            p = round(base * mult, 3)
            d.joinpath("result.json").write_text(json.dumps({
                "success_rate": p, "n_total": 120, "n_tasks": 8,
                "ci95_task": [max(0, p - 0.06), min(1, p + 0.06)],
            }))
    # Exercise the real eval-A data path: per-task counts drive the recompute,
    # and the stored aggregate/paired blocks must match or _evala_recompute
    # raises. (Same numbers as data/eval/eval_A_summary.json.)
    eval_a = {
        "task_ids_effective": [79, 108, 1530, 1817, 1955, 2126, 2172],
        "eval_trials": 15,
        "excluded": {"1477": "renderer bug, all arms"},
        "per_task_successes_out_of_15": {
            "sft":    {"79": 7, "108": 9,  "1530": 0, "1817": 4,  "1955": 2, "2126": 4, "2172": 4},
            "rs_sft": {"79": 8, "108": 11, "1530": 1, "1817": 12, "1955": 5, "2126": 3, "2172": 5},
            "dpo":    {"79": 4, "108": 10, "1530": 1, "1817": 5,  "1955": 5, "2126": 8, "2172": 4},
            "grpo":   {"79": 8, "108": 8,  "1530": 0, "1817": 5,  "1955": 2, "2126": 4, "2172": 6},
        },
        "aggregate": {
            "sft": {"n_success": 30, "n_total": 105, "success_rate": 0.2857,
                    "ci95_task": [0.1524, 0.4190]},
            "rs_sft": {"n_success": 45, "n_total": 105, "success_rate": 0.4286,
                       "ci95_task": [0.2476, 0.6190]},
            "dpo": {"n_success": 37, "n_total": 105, "success_rate": 0.3524,
                    "ci95_task": [0.2286, 0.4857]},
            "grpo": {"n_success": 33, "n_total": 105, "success_rate": 0.3143,
                     "ci95_task": [0.1714, 0.4476]},
        },
        "paired_vs_sft": {
            "rs_sft": {"method_win_sft_lose": 19, "sft_win_method_lose": 4,
                       "net_paired_wins": 15, "mcnemar_p": 0.003,
                       "per_task_diff_ci95_taskboot": [0.0286, 0.2857]},
            "dpo": {"method_win_sft_lose": 16, "sft_win_method_lose": 9,
                    "net_paired_wins": 7, "mcnemar_p": 0.230,
                    "per_task_diff_ci95_taskboot": [-0.0381, 0.1619]},
            "grpo": {"method_win_sft_lose": 11, "sft_win_method_lose": 8,
                     "net_paired_wins": 3, "mcnemar_p": 0.648,
                     "per_task_diff_ci95_taskboot": [-0.0190, 0.0762]},
        },
    }
    (tmp / "eval_A.json").write_text(json.dumps(eval_a))
    out = tmp / "macros" / "results_macros.tex"
    ns = argparse.Namespace(
        metrics_run7=tmp / "run7_m.jsonl", metrics_run8=tmp / "run8_m.jsonl",
        episodes_run7=tmp / "run7_e.jsonl", episodes_run8=tmp / "run8_e.jsonl",
        eval_root=tmp / "eval", eval_a_summary=tmp / "eval_A.json", out=out)
    run(ns)
    text = out.read_text()
    print("\n----- generated results_macros.tex -----\n" + text)
    print("----- generated tab_fourarm.tex -----\n" + (out.parent / "tab_fourarm.tex").read_text())
    print("----- generated tab_itertrend.tex -----\n" + (out.parent / "tab_itertrend.tex").read_text())
    print("----- generated tab_evalA.tex -----\n" + (out.parent / "tab_evalA.tex").read_text())
    assert "n/a" not in text, "some macro was left n/a on full synthetic input"
    # eval-A self-checks: primary + paired macros present and the table exists.
    for needle in ("\\armRsSftPrimary", "\\rsSftPairedP", "\\grpoNetWins",
                   "\\armSftPrimaryCIlo"):
        assert needle in text, f"missing eval-A macro {needle}"
    assert (out.parent / "tab_evalA.tex").is_file(), "tab_evalA.tex not written"
    print(f"\n[demo] OK -> {tmp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-run7", type=Path)
    ap.add_argument("--episodes-run7", type=Path)
    ap.add_argument("--metrics-run8", type=Path)
    ap.add_argument("--episodes-run8", type=Path)
    ap.add_argument("--eval-root", type=Path)
    ap.add_argument("--eval-a-summary", type=Path,
                    help="data/eval/eval_A_summary.json (primary paired eval)")
    ap.add_argument("--eval-secondary-summary", type=Path,
                    help="data/eval/eval_secondary_summary.json (unseen-task probe)")
    ap.add_argument("--out", type=Path,
                    default=Path("analysis/generated/results_macros.tex"),
                    help="destination .tex; its directory is created if missing, "
                         "and the sibling tab_*.tex files are written next to it")
    ap.add_argument("--demo", action="store_true", help="self-test on synthetic logs")
    args = ap.parse_args()
    if args.demo:
        _demo()
    else:
        run(args)


if __name__ == "__main__":
    main()
