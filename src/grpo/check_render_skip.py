"""Equivalence check for render_skip (~3 min on Colab).

Runs the same seed and the same action sequence in two envs,
  A: normal (render every step)
  B: render_skip (enabled=False mid-chunk, re-enabled before the final step)
and verifies that the chunk-boundary observation images (byte-identical),
staged reward components, and success checks all match. Physics is
independent of rendering, so any mismatch means a wrapper bug (e.g. a stale
image leaking into a boundary).

Also a quick benchmark: times chunk execution in each mode and reports the
reduction.

**Passing this check does NOT prove equivalence (re-audit, 2026-07).**
`RenderSkipWrapper._cameras()` walks the wrapper stack looking for robosuite's
`_observables`, but gymnasium 1.x removed `Wrapper.__getattr__`, so the probe does
not resolve through `RenderSkipWrapper(StagedRewardWrapper(LiberoEnv))`. It falls
back to an empty camera list, and `set_render_skip` returns having "controlled 0
cameras", i.e. having done nothing. In that state identical observations are
guaranteed and the timing ratio is 1.0x. So this check cannot distinguish "the
optimization is correct and worthless" from "the optimization never engaged". The
measured x0.99 was explained as "EGL rendering runs on the GPU while chunk
execution is CPU-bound on physics, so cutting renders buys nothing"; **that
explanation has been withdrawn** (paper, sec:negative). Before trusting the
verdict, confirm that `venv.call("set_render_skip", True)` returns non-zero (i.e.
cameras were actually stopped). Both GRPO runs ran with render_skip false, so no
downstream result depends on this.

Usage:
    python -m src.grpo.check_render_skip [--suite libero_spatial] [--task_id 79]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .spawn_env import make_spawn_vec_env


CHUNK = 10   # pseudo chunk length for verification (production uses n_action_steps=50)
N_CHUNKS = 3


def _numeric_leaves(x, prefix: str = "") -> dict:
    """Recursively flatten a nested dict observation, returning only numeric ndarray leaves as {path: array}.

    LIBERO's obs nests like {"pixels": {"image": ..., "image2": ...},
    "agent_pos": ...}, so a naive np.asarray(dict) becomes an object array and
    cannot be compared (measured: np.array_equal raises ValueError). Extracting
    only the numeric leaves allows byte comparison of both images and proprio
    (proprio match = extra confirmation that physics is also equivalent).
    """
    out: dict = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(_numeric_leaves(v, f"{prefix}{k}."))
    else:
        a = np.asarray(x)
        if a.dtype != object:
            out[prefix.rstrip(".")] = a.copy()
    return out


def run_mode(suite: str, task_id: int, skip: bool, seed: int):
    """Run N_CHUNKS×CHUNK steps in one env and return chunk-boundary observations plus metadata."""
    _, venv = make_spawn_vec_env(suite, task_id, None, n_envs=1, staged_reward=True)
    obs, _ = venv.reset(seed=[seed])
    rng = np.random.default_rng(seed)
    boundary_obs = []
    elapsed = 0.0
    for _c in range(N_CHUNKS):
        actions = rng.uniform(-0.3, 0.3, size=(CHUNK, venv.action_space.shape[-1])).astype(np.float32)
        t0 = time.perf_counter()
        if skip:
            venv.call("set_render_skip", True)
        for t in range(CHUNK):
            if skip and t == CHUNK - 1:
                venv.call("set_render_skip", False)
            obs, _r, _te, _tr, info = venv.step(actions[t : t + 1])
        elapsed += time.perf_counter() - t0
        # record the chunk-boundary observation (all numeric leaves: images + proprio)
        boundary_obs.append(_numeric_leaves(obs if isinstance(obs, dict) else {"obs": obs}))
    staged = venv.call("staged_now")[0]
    venv.close()
    return boundary_obs, staged, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task_id", type=int, default=79)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("[check_render_skip] mode A (normal render) ...")
    obs_a, staged_a, t_a = run_mode(args.suite, args.task_id, skip=False, seed=args.seed)
    print(f"  done ({t_a:.1f}s)")
    print("[check_render_skip] mode B (render_skip) ...")
    obs_b, staged_b, t_b = run_mode(args.suite, args.task_id, skip=True, seed=args.seed)
    print(f"  done ({t_b:.1f}s)")

    ok = True
    for c, (a, b) in enumerate(zip(obs_a, obs_b)):
        keys = sorted(set(a) | set(b))
        for k in keys:
            same = (
                k in a and k in b
                and a[k].shape == b[k].shape
                and a[k].dtype == b[k].dtype
                and np.array_equal(a[k], b[k])
            )
            print(f"  chunk {c} obs[{k}]: {'MATCH' if same else 'MISMATCH'} shape={a[k].shape if k in a else '?'}")
            ok &= same
    same_staged = staged_a == staged_b
    print(f"  staged_now: {'MATCH' if same_staged else f'MISMATCH A={staged_a} B={staged_b}'}")
    ok &= same_staged
    speedup = t_a / t_b if t_b > 0 else float("nan")
    print(f"  chunk-exec time: normal {t_a:.1f}s vs skip {t_b:.1f}s  (x{speedup:.2f})")

    if ok:
        print("[check_render_skip] OK — no difference in observations or rewards. Note "
              "that a wrapper degraded to a no-op gives the same result (see the "
              "docstring); confirm set_render_skip returned non-zero before enabling")
    else:
        print("[check_render_skip] FAIL — not equivalent; do not enable render_skip")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
