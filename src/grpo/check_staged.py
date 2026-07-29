"""Live verification of the staged reward path (~2 min on Colab).

LIBERO does not self-truncate: LiberoEnv.step returns truncated=False
unconditionally, and terminated = done or is_success -- overwhelmingly success
here, though the underlying env's done is a second possible cause (this used to
say "only on success"; corrected 2026-07-29). So after running some random
actions we call
StagedRewardWrapper.staged_now() directly via venv.call and check whether the
privileged probe is alive inside the spawn workers. If a success terminal
happens to occur, extraction from the gymnasium 1.x info aggregation is
verified as well.

Expected output:
  - "staged_now: [{'staged_reward': 0.03, 'staged_components': {...}}, ...]"
    with 5 keys inside components — a silent probe-failure fallback would
    leave components as an empty dict
  - finally "OK — staged reward probe works inside spawn workers"

Usage:
    python -m src.grpo.check_staged [--suite libero_spatial] [--task_id 79]
"""
from __future__ import annotations

import argparse

import numpy as np

from .privileged import make_staged_vec_env
from .rollout import _values_from_final_info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task_id", type=int, default=79)
    ap.add_argument("--n_envs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()

    _, venv = make_staged_vec_env(args.suite, args.task_id, None, args.n_envs)
    print(f"[check_staged] env ready: {args.suite}:{args.task_id} n_envs={args.n_envs}")
    venv.reset(seed=list(range(args.n_envs)))

    for step in range(1, args.steps + 1):
        _obs, _r, term, trunc, info = venv.step(venv.action_space.sample())
        jd = (np.asarray(term) | np.asarray(trunc)).reshape(-1)
        if jd.any():  # a success terminal also exercises the extraction path (rare with random actions)
            vals = _values_from_final_info(info, "staged_reward", args.n_envs)
            print(f"[check_staged] step {step}: terminal envs {np.flatnonzero(jd).tolist()}, "
                  f"extracted staged_reward={vals}")

    res = venv.call("staged_now")
    venv.close()
    print(f"[check_staged] staged_now after {args.steps} random steps:")
    ok = True
    for i, r in enumerate(res):
        print(f"  env {i}: {r}")
        if not (isinstance(r, dict) and r.get("staged_components")):
            ok = False

    if ok:
        print("[check_staged] OK — staged reward probe works inside spawn workers")
    else:
        print("[check_staged] FAIL — probe fell back (empty staged_components); "
              "privileged state is not reachable inside the worker")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
