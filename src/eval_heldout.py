"""LIBERO / LIBERO-Plus evaluation runner.

For each (suite, mode), evaluate the selected tasks **one task at a time**
for eval_trials episodes and collect per-task outcomes. mode:
    in_dist  ... sampled from tasks in perturbation categories other than camera_pose
    heldout  ... sampled only from camera_pose (viewpoint) perturbation tasks

Statistical caveats (always account for these when claiming results):
  * Episodes within a task share difficulty and perturbation, and where
    len(init_states) == 1 they share the initial state too. That does not by
    itself make them statistically dependent: conditional on a fixed initial
    state, independently seeded rollouts are independent draws (corrected
    2026-07-29 -- this docstring used to claim dependence, and that was wrong).
    What it does mean is that ci95_episode is an interval for success *at the
    states actually visited*, not for success on the task, because it samples no
    initial-state variability at all. The **task bootstrap interval (ci95_task)**,
    which resamples tasks, is reported alongside; use it to judge superiority
    between methods or vs. upstream.
  * in_dist and heldout are **sampled from different task pools**, so their
    success-rate difference confounds "viewpoint sensitivity" with "task
    difficulty difference". Claiming pure viewpoint generalization would
    require pairing the same base task with/without the viewpoint
    perturbation (not implemented: for now read it as "performance on a
    distribution that includes viewpoint perturbations").
  * The "heldout" mode refers **only to the task pool of a perturbation category
    (camera_pose)**, never to initial states. The initial state is selected by
    neither the seed nor the mode: LiberoEnv.reset() overwrites it with
    set_init_state(init_states[init_state_id % N]), where init_state_id starts
    from episode_index and advances by n_envs on every reset.
    **As long as you evaluate an already-trained task, no choice of seed or
    offset makes the initial conditions held out** (see the --init_state_offset
    help). To rule out contamination by construction, evaluate untrained tasks.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path


HELDOUT_CATEGORIES = {"Camera Viewpoints"}


def _load_classification(libero_root: Path) -> dict:
    path = libero_root / "libero" / "libero" / "benchmark" / "task_classification.json"
    with path.open() as f:
        return json.load(f)


def _resolve_task_ids(suite: str, mode: str, classification: dict) -> list[int]:
    """Return zero-based LeRobot task IDs for the requested category split."""
    entries = classification.get(suite, [])
    candidates: list[int] = []
    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        in_heldout = entry.get("category") in HELDOUT_CATEGORIES
        if (mode == "heldout") == in_heldout:
            # task_classification.json is one-based; LeRobot's task_ids are zero-based.
            candidates.append(int(entry["id"]) - 1)
    return sorted(set(candidates))


def _camera_name_mapping(policy_path: Path) -> str | None:
    """Return JSON mapping the env's cameras to the policy's visual feature names."""
    cfg_path = policy_path / "config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    visuals = sorted(
        k.removeprefix("observation.images.")
        for k, v in cfg.get("input_features", {}).items()
        if v.get("type") == "VISUAL" and "empty_camera" not in k
    )
    if len(visuals) < 2:
        return None
    return json.dumps(
        {"agentview_image": visuals[0], "robot0_eye_in_hand_image": visuals[1]},
        separators=(",", ":"),
    )


def _eval_once(
    checkpoint: Path,
    suite: str,
    task_id: int,
    n_episodes: int,
    output_dir: Path,
    batch_size: int,
) -> dict:
    """Evaluate one task with lerobot-eval and return the parsed eval_info.json."""
    policy_path = checkpoint / "pretrained_model"
    if not policy_path.is_dir():
        policy_path = checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, min(batch_size, n_episodes))
    use_async = "true" if batch_size > 1 else "false"
    cmd = [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        "--env.type=libero_plus",
        f"--env.task={suite}",
        f"--env.task_ids={json.dumps([task_id], separators=(',', ':'))}",
        f"--eval.batch_size={batch_size}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.use_async_envs={use_async}",
        f"--output_dir={output_dir}",
    ]
    cam_map = _camera_name_mapping(policy_path)
    if cam_map:
        cmd.insert(5, f"--env.camera_name_mapping={cam_map}")
    subprocess.run(cmd, check=True)
    return json.loads((output_dir / "eval_info.json").read_text())


# ------------------------------------------------------------
# In-process engine: load the policy once per method and run all tasks.
# The CLI engine launches lerobot-eval per (method × mode × task), paying the
# policy load + env build (~2 min) every time (~2 h of pure overhead over 64
# invocations). This engine counts success checks directly, so it also does
# not depend on the eval_info.json schema.
# Validation procedure: run the same (checkpoint, task) once on both engines
# and confirm the success rates agree within binomial fluctuation before
# switching to inproc.
# ------------------------------------------------------------
class _InprocEvaluator:
    """Runs deterministic evaluation (policy.select_action) in our own spawn envs.

    Uses the same transform chain as lerobot-eval (preprocess_observation →
    attach task → env_preprocessor → preprocessor → select_action →
    postprocessor → env_postprocessor). No SDE (normal inference).
    """

    def __init__(self, checkpoint: Path):
        import torch

        import lerobot.policies  # noqa: F401  (draccus registry)
        from lerobot.configs.policies import PreTrainedConfig

        from .grpo.train_grpo import (
            _build_processors,
            _camera_name_mapping as _cam_map_dict,
            _load_policy,
            _resolve_pretrained_dir,
        )

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_dir = _resolve_pretrained_dir(str(checkpoint))
        self.cam_map = _cam_map_dict(PreTrainedConfig.from_pretrained(str(self.pretrained_dir)))

        from lerobot.envs.configs import LiberoPlusEnv

        env_cfg = LiberoPlusEnv(task="libero_spatial", task_ids=[0], camera_name_mapping=self.cam_map)
        self.policy_cfg, self.policy = _load_policy(self.pretrained_dir, env_cfg, self.device)
        self.processors = _build_processors(self.policy_cfg, env_cfg, self.pretrained_dir, self.device)
        self.policy.eval()

    def eval_task(
        self, suite: str, task_id: int, n_episodes: int, batch_size: int, seed: int,
        init_state_offset: int = 0,
    ) -> tuple[int, int]:
        """Evaluate one task for n_episodes and return (n_total, n_success).

        Episode seeds are assigned deterministically from `seed`. Note, however,
        that in LIBERO-Plus the seed does NOT select the initial state (it only
        drives the action-sampling RNG). What determines the initial state is
        init_state_offset: the indices actually used are
        offset .. offset+n_episodes-1. Passing the same offset to every method
        therefore gives paired samples whose initial states match exactly.

        **Do NOT try to build a held-out evaluation on an already-trained task
        with this offset.** This docstring used to say that "32 or above gives
        initial states neither rollout collection ({0..31}) nor GRPO training
        ({0..7}) ever touched". The analysis of 2026-07-25 showed that is wrong.
        A terminated episode runs extra resets, so collection does not touch
        {0..31} but a scattered, outcome-dependent set. That analysis estimated
        it reaches as high as 79 and, after wrapping via % N, contaminates 32 of
        50 leaving no free window of width 15; those specific figures rest on a
        reset model that can err in both directions and are not established
        (2026-07-29 correction -- see src/init_state_coverage.py). The part that
        decides this docstring either way is weaker and does hold: the visited
        set cannot be reconstructed reliably, so no offset can be shown to be
        held out. Evaluating an untrained task cannot suffer this contamination
        in principle, so the offset can stay at 0 (see the --init_state_offset
        help as well).
        """
        import numpy as np

        from lerobot.envs import preprocess_observation
        from lerobot.utils.constants import ACTION

        from .grpo.rollout import _successes_from_info
        from .grpo.spawn_env import make_spawn_vec_env

        torch = self.torch
        n_done = n_success = 0
        wave_idx = 0
        episodes: list[dict] = []   # per-episode record (seed/init_state, success)
        B0 = min(batch_size, n_episodes)
        n_init_task: int | None = None   # len(init_states); read on wave 0
        # Rebuild the venv for every wave. Rationale (2026-07-25 audit):
        #   * In LIBERO-Plus the initial state is decided not by the seed but by
        #     episode_index and the number of resets (see the docstring of
        #     spawn_env.make_spawn_vec_env).
        #   * A terminated episode causes AT LEAST TWO resets (LiberoEnv.step's
        #     internal reset + the vector env's autoreset; more if the slot,
        #     which keeps being stepped, terminates again), so if the venv is
        #     reused the second wave's initial state ends up depending on "this
        #     arm's own outcome pattern in the first wave", which breaks pairing.
        #     In practice 7 of the 15 trials in eval A had become unpaired this way.
        # Rebuilding each wave with base = init_state_offset + wave*B0 makes the
        # initial state independent of the arm's success pattern, so it matches
        # exactly across every method.
        while n_done < n_episodes:
            base = init_state_offset + wave_idx * B0
            _, venv = make_spawn_vec_env(
                suite, task_id, self.cam_map, n_envs=B0, episode_index_base=base
            )
            try:
                try:
                    env_max = venv.call("_max_episode_steps")[0]
                except Exception:  # noqa: BLE001
                    env_max = 300
                limit = int(env_max or 300)
                if wave_idx == 0:
                    # Always read N. Even at offset=0, if n_episodes > N the window
                    # wraps around and evaluates the same initial state twice.
                    # Those repeats are not statistically dependent, but they add
                    # no initial-state variability, so n counts as distinct
                    # initial conditions episodes that are not (corrected
                    # 2026-07-29). This hole exists whether or not an offset is set.
                    try:
                        n_init = int(venv.call("_init_states")[0].shape[0])
                    except Exception as e:  # noqa: BLE001
                        if init_state_offset:
                            raise SystemExit(
                                f"{suite}:{task_id}: cannot read _init_states ({e}); refusing "
                                "to run an offset protocol blind."
                            ) from e
                        print(
                            f"[warn] {suite}:{task_id}: cannot read _init_states ({e}); "
                            f"cannot verify that {n_episodes} episodes fit without wrapping.",
                            flush=True,
                        )
                        n_init = None
                    if n_init is not None and init_state_offset + n_episodes > n_init:
                        raise SystemExit(
                            f"{suite}:{task_id} has only {n_init} init states; "
                            f"offset {init_state_offset}+{n_episodes} would wrap (% N) and "
                            "re-evaluate initial states already used in this same run"
                            + (" (and fall back onto training states)" if init_state_offset else "")
                            + ". Lower --eval_trials or pick a different task."
                        )
                    n_init_task = n_init
                B = venv.num_envs
                seeds = [seed + wave_idx * 1000 + i for i in range(B)]
                torch.manual_seed(seed + wave_idx)  # reproducibility of the FM initial noise
                obs, _ = venv.reset(seed=seeds)
                self.policy.reset()
                done = np.zeros(B, dtype=bool)
                success = np.zeros(B, dtype=bool)
                step = 0
                while not done.all() and step < limit:
                    observation = preprocess_observation(obs)
                    try:
                        observation["task"] = list(venv.call("task_description"))
                    except Exception:  # noqa: BLE001
                        observation["task"] = [""] * B
                    observation = self.processors["env_preprocessor"](observation)
                    observation = self.processors["preprocessor"](observation)
                    with torch.no_grad():
                        action = self.policy.select_action(observation)
                    action = self.processors["postprocessor"](action)
                    transition = self.processors["env_postprocessor"]({ACTION: action})
                    obs, _r, term, trunc, info = venv.step(transition[ACTION].to("cpu").numpy())
                    step += 1
                    just = (np.asarray(term) | np.asarray(trunc)).reshape(-1) & ~done
                    if just.any():
                        succ = _successes_from_info(info, B)
                        for i in np.flatnonzero(just):
                            success[i] = succ[i]
                        done |= just
                take = min(B, n_episodes - n_done)  # remainder wave: keep only the first `take` envs
                for i in range(take):
                    episodes.append({
                        "seed": int(seeds[i]),
                        # The guard above guarantees offset+n_episodes <= N, so
                        # base+i cannot wrap around. n_init is recorded too so this
                        # can be audited after the fact.
                        "init_state": int(base + i),   # index of the initial state actually used
                        "n_init": n_init_task,
                        "success": bool(success[i]),
                    })
                n_done += take
                n_success += int(success[:take].sum())
                wave_idx += 1
            finally:
                venv.close()
        return n_done, n_success, episodes


def _extract_outcomes(raw: dict, budget_n: int) -> tuple[int, int]:
    """Return (n_total, n_success) from eval_info.json.

    Measured per-episode outcomes take priority (a single-task invocation, so
    their count is exactly this task's trial count). Only when no outcome list
    exists do we approximate with pc_success × budget_n (budget_n = the
    n_episodes spent on this task = the correct total; previously
    len(task_ids)×n_episodes inflated the total and made the CI overconfident:
    caught by Codex).
    """
    overall = raw.get("overall") or raw.get("aggregated") or {}
    for src in (overall.get("successes"), raw.get("successes")):
        if isinstance(src, list) and src:
            return len(src), sum(bool(x) for x in src)
    per_ep = raw.get("per_episode")
    if isinstance(per_ep, list) and per_ep:
        succ = [bool(e.get("success", e.get("is_success"))) for e in per_ep]
        return len(succ), sum(succ)
    pc = overall.get("pc_success")
    if pc is not None:
        return budget_n, round(float(pc) * budget_n / 100)
    raise RuntimeError(
        "eval_info.json has no per-episode success/failure, so the number of trials "
        "cannot be determined (check the lerobot version / output schema)"
    )


def _wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson 95% interval for a binomial success rate (assumes episode independence; ignores task correlation = optimistic)."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def _task_bootstrap_ci(
    per_task: list[dict], seed: int, iters: int = 10000
) -> list[float]:
    """95% bootstrap interval resampling tasks.

    Each bootstrap sample computes "resample tasks with replacement → sum
    their (k,n) → rate", then takes the 2.5/97.5 percentiles. It captures
    between-task variance, correcting the overconfidence (Wilson's optimism
    bias) of settings that evaluate few tasks with many episodes.
    """
    tasks = [(t["k"], t["n"]) for t in per_task if t["n"] > 0]
    if len(tasks) < 2:  # a single task cannot be bootstrapped → return the episode Wilson interval
        k = sum(t[0] for t in tasks)
        n = sum(t[1] for t in tasks)
        return _wilson_ci(k, n)
    rng = random.Random(seed)
    m = len(tasks)
    rates = []
    for _ in range(iters):
        kk = nn = 0
        for _ in range(m):
            k, n = tasks[rng.randrange(m)]
            kk += k
            nn += n
        rates.append(kk / nn if nn else 0.0)
    rates.sort()
    lo = rates[int(0.025 * iters)]
    hi = rates[int(0.975 * iters)]
    return [round(lo, 4), round(hi, 4)]


def run(args: argparse.Namespace) -> None:
    # Guard against a silent no-op: the cli engine does not implement
    # init_state_offset. (There is precedent — the render-skip wrapper once
    # silently degraded into a no-op — so always fail hard here.)
    if getattr(args, "init_state_offset", 0) and getattr(args, "engine", "cli") != "inproc":
        raise SystemExit(
            f"--init_state_offset={args.init_state_offset} requires --engine inproc; "
            f"the {args.engine!r} engine ignores it and would silently evaluate the "
            "TRAINING initial states."
        )
    if getattr(args, "task_ids", None):
        # Explicit task list (e.g. evaluating the 8 tasks used for GRPO training).
        # Bypasses the split lottery and evaluates in the given order.
        # NOTE: choosing evaluation seeds disjoint from training does NOT produce
        # held-out initial conditions. In LIBERO-Plus the initial state is fixed by
        # init_state_id, not by the seed, so the moment you name an already-trained
        # task this becomes a paired comparison ON THE TRAINING initial conditions,
        # not a generalization measurement (this is exactly what eval A was, and it
        # is biased in favour of RS-SFT / flow-DPO, which are built from the
        # collected rollouts). To measure generalization, pass untrained tasks.
        selected = [int(x) for x in args.task_ids.split(",") if x.strip()]
    else:
        classification = _load_classification(args.libero_root)
        task_ids = _resolve_task_ids(args.suite, args.mode, classification)
        if not task_ids:
            raise SystemExit(f"no tasks matched suite={args.suite} mode={args.mode}")
        # Feasibility filter. In LIBERO-Plus, N=len(init_states) is binary — either
        # 1 or 50 — and N=1 tasks make up about 19% of the in_dist pool. Requesting
        # 3 episodes hits the wave-0 guard on an N=1 task and halts the entire run
        # (this killed the first run of 2026-07-25 outright). N is a benchmark-specific
        # constant unrelated to the results, so narrowing the pool by it cannot make
        # the selection depend on observations. Apply it BEFORE sample.
        if args.min_init_states:
            from lerobot.envs.libero import _get_suite, get_task_init_states

            # The first argument of get_task_init_states is the suite OBJECT,
            # not the suite name.
            suite_obj = _get_suite(args.suite)
            keep, dropped = [], []
            for tid in task_ids:
                try:
                    n_init = int(len(get_task_init_states(suite_obj, tid, is_libero_plus=True)))
                except Exception as e:  # noqa: BLE001
                    raise SystemExit(
                        f"cannot read init_states for {args.suite}:{tid} ({e}); refusing to "
                        "build a sampling pool we cannot verify."
                    ) from e
                (keep if n_init >= args.min_init_states else dropped).append(tid)
            print(
                f"[pool] min_init_states={args.min_init_states}: keeping {len(keep)} of "
                f"{len(task_ids)} tasks ({len(dropped)} dropped for too few initial states)",
                flush=True,
            )
            task_ids = keep

        # Building a preregistered "novel" task set requires removing from the
        # population any task whose results we have already seen. Otherwise the
        # selection could depend on past observations (garden of forking paths).
        # Apply the exclusion BEFORE sample.
        excluded = {int(x) for x in (args.exclude_task_ids or "").split(",") if x.strip()}
        if excluded:
            pool = [t for t in task_ids if t not in excluded]
            hit = len(task_ids) - len(pool)
            if hit != len(excluded):
                print(
                    f"[warn] --exclude_task_ids listed {len(excluded)} ids but only {hit} "
                    f"were present in suite={args.suite} mode={args.mode}",
                    flush=True,
                )
            if len(pool) < args.tasks_per_suite:
                raise SystemExit(
                    f"after excluding {hit} tasks only {len(pool)} remain, fewer than the "
                    f"requested {args.tasks_per_suite}; the preregistered design cannot be met."
                )
            task_ids = pool
        random.seed(args.seed)
        selected = random.sample(task_ids, min(args.tasks_per_suite, len(task_ids)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_task: list[dict] = []
    # Resilience to VM reclamation: write task_<tid>.json incrementally per task and
    # skip any that already exist. Put output_dir on Drive and completed tasks
    # survive a VM reclaim, so re-running the same command resumes where it left off
    # (the policy load is lazy as well).
    evaluator = None
    for tid in selected:
        task_file = args.output_dir / f"task_{int(tid):05d}.json"
        if task_file.exists():
            cached = json.loads(task_file.read_text())
            if cached.get("error"):
                # Do not cache a failed task forever (the denominator would shrink
                # silently). Setting cached=None here used to make the cached.get()
                # below raise AttributeError, so on resume the whole run died at the
                # first errored task.
                print(f"[eval] {args.suite}:{tid} cached ERROR -> re-running")
                cached = None
                continue_to_rerun = True
            else:
                continue_to_rerun = False
            # Never silently reuse a cache produced under different settings. Old
            # runs did not record init_state, so always reject them on a run that
            # uses an offset.
            want = int(getattr(args, "init_state_offset", 0))
            eps = (cached or {}).get("episodes") or []
            got = eps[0].get("init_state") if eps else None
            if not continue_to_rerun and eps and got is not None and int(got) != want:
                raise SystemExit(
                    f"{task_file} was produced with init_state offset {got}, but this run "
                    f"asks for {want}. Use a fresh --output_dir instead of mixing protocols."
                )
            if want and eps and got is None:
                raise SystemExit(
                    f"{task_file} predates init-state logging (old offset-0 protocol). "
                    f"Use a fresh --output_dir for the offset-{want} run."
                )
            if cached is not None:
                per_task.append(cached)
                print(f"[eval] {args.suite}:{tid} cached (skip)")
                continue
            # cached is None => an errored cache. Do NOT continue here; fall through
            # so the task is actually re-run. (This used to `continue`, printing
            # "re-running" while in fact skipping the task entirely and dropping it
            # from the denominator.)
        eps: list = []
        if args.engine == "inproc":
            if evaluator is None:
                evaluator = _InprocEvaluator(args.checkpoint)
            # Per-task fault tolerance: a rare exception (e.g. in worker rendering)
            # must not stop the whole run. Retry once with a fresh venv; if it still
            # fails, record the error, skip, and move on to the next task. An errored
            # task has n=0, so it cannot contaminate success_rate (it never enters
            # the denominator).
            try:
                n, k, eps = evaluator.eval_task(
                    args.suite, tid, args.eval_trials, args.eval_batch_size,
                    seed=args.seed + tid, init_state_offset=args.init_state_offset,
                )
            except Exception as e1:  # noqa: BLE001
                print(f"[eval] {args.suite}:{tid} error, retry once: {str(e1)[:120]}")
                try:
                    n, k, eps = evaluator.eval_task(
                        args.suite, tid, args.eval_trials, args.eval_batch_size,
                        seed=args.seed + tid, init_state_offset=args.init_state_offset,
                    )
                except Exception as e2:  # noqa: BLE001
                    print(f"[eval] {args.suite}:{tid} FAILED twice, skip: {str(e2)[:120]}")
                    entry = {"task_id": tid, "n": 0, "k": 0, "rate": 0.0,
                             "episodes": [], "error": str(e2)[:200]}
                    task_file.write_text(json.dumps(entry))
                    per_task.append(entry)
                    continue
        else:
            raw = _eval_once(
                args.checkpoint,
                args.suite,
                tid,
                args.eval_trials,
                args.output_dir / f"task_{tid}" / "lerobot_eval",
                batch_size=args.eval_batch_size,
            )
            n, k = _extract_outcomes(raw, args.eval_trials)
        # episodes: [{seed, init_state, n_init, success}]. The pairing key is
        # init_state, NOT seed (the seed only drives action sampling). Because the
        # venv is rebuilt per wave, init_state is a function of (offset, wave, batch)
        # alone and is therefore identical across every method, so paired differences
        # and discordant pairs can be computed after the fact on matching
        # (task_id, init_state) (inproc only). The seed is kept for auditing.
        entry = {"task_id": tid, "n": n, "k": k, "rate": round(k / n, 4) if n else 0.0, "episodes": eps}
        task_file.write_text(json.dumps(entry))   # incremental save (completed tasks survive a reclaim)
        per_task.append(entry)
        print(f"[eval] {args.suite}:{tid} {k}/{n}")

    n_total = sum(t["n"] for t in per_task)
    n_success = sum(t["k"] for t in per_task)
    result = {
        "suite": args.suite,
        "mode": args.mode,
        "success_rate": round(n_success / n_total, 4) if n_total else 0.0,
        "n_total": n_total,
        "n_success": n_success,
        "n_tasks": len(per_task),
        "ci95_episode": _wilson_ci(n_success, n_total),          # optimistic (independence assumption)
        "ci95_task": _task_bootstrap_ci(per_task, seed=args.seed),  # accounts for task correlation (use this for claims)
        "per_task": per_task,
        "task_ids": selected,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}, indent=2))


def aggregate(args: argparse.Namespace) -> None:
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    overall = {"per_suite": {}, "n_total": 0, "n_success": 0, "per_task_all": []}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        rf = sub / "result.json"
        if not rf.exists():
            continue
        r = json.loads(rf.read_text())
        overall["per_suite"][sub.name] = {
            "success_rate": r["success_rate"],
            "n_total": r["n_total"],
            "ci95_task": r.get("ci95_task"),
        }
        overall["n_total"] += r["n_total"]
        overall["n_success"] += r["n_success"]
        overall["per_task_all"].extend(r.get("per_task", []))
    overall["success_rate"] = (
        round(overall["n_success"] / overall["n_total"], 4) if overall["n_total"] else 0.0
    )
    if overall["per_task_all"]:
        overall["ci95_task"] = _task_bootstrap_ci(overall["per_task_all"], seed=args.seed)
    (root / "overall.json").write_text(json.dumps(overall, indent=2))
    print(json.dumps({k: v for k, v in overall.items() if k != "per_task_all"}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--suite", type=str)
    ap.add_argument("--mode", choices=["in_dist", "heldout"], default="in_dist")
    ap.add_argument("--eval_trials", type=int, default=10, help="episodes per task")
    ap.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="parallel envs per task (>1 uses an async vec env); capped at eval_trials",
    )
    ap.add_argument("--tasks_per_suite", type=int, default=10)
    ap.add_argument(
        "--min_init_states", type=int, default=0,
        help="restrict the lottery population to tasks with len(init_states) >= this value (0=off).\n             In LIBERO-Plus N is binary, either 1 or 50; requesting 3 episodes on an\n             N=1 task halts at the wave-0 guard, so drop them up front.",
    )
    ap.add_argument(
        "--exclude_task_ids", type=str, default=None,
        help="task ids to drop from the lottery population (comma-separated). Used when\n             building a preregistered novel task set, to exclude tasks whose results\n             have already been seen.",
    )
    ap.add_argument(
        "--engine",
        choices=["cli", "inproc"],
        default="cli",
        help="cli=launch lerobot-eval per task (proven, slow) / inproc=load the policy "
        "once and evaluate in our own env (fast; allows a paired comparison at fixed "
        "initial-state indices and records the per-episode init_state; cross-check one "
        "task on both engines before switching)",
    )
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--libero_root", type=Path, default=Path("/content/LIBERO-plus"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--init_state_offset", type=int, default=0,
        help="starting index of the initial states used for evaluation. In LIBERO-Plus "
             "it is this, not the seed, that determines the initial state. **Do NOT try "
             "to build a held-out evaluation on an already-trained task with this**: a "
             "terminated episode runs extra resets, so collection touches a scattered, "
             "outcome-dependent set rather than {0..31}, and it is not reliably "
             "reconstructible from the success bits we kept -- no offset can be shown "
             "to be held out (analysis of 2026-07-25, narrowed 2026-07-29; the "
             "'32 of 50, no window of width 15' figures it produced are an estimate, "
             "not a fact). Evaluating an untrained task cannot suffer this "
             "contamination in principle, so leaving it at 0 is fine.",
    )
    ap.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help="comma-separated explicit task ids. When given, these are evaluated instead "
        "of drawing from the split. If you pass already-trained tasks, changing the seed "
        "relative to training does NOT make the initial conditions held out (the initial "
        "state is fixed by init_state_id, not by the seed), so the result is a paired "
        "comparison ON THE TRAINING initial conditions rather than a generalization "
        "measurement.",
    )
    ap.add_argument("--aggregate", action="store_true", help="aggregation mode")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args)
    else:
        if not (args.checkpoint and args.suite):
            raise SystemExit("--checkpoint and --suite are required")
        run(args)


if __name__ == "__main__":
    main()
