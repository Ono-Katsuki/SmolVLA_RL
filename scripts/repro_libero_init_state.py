"""Minimal runnable reproduction: LIBERO initial-state allocation depends on outcomes.

    python scripts/repro_libero_init_state.py        # no arguments

WHAT IS BEING CLAIMED
---------------------
`LiberoEnv.reset()` does not pick the initial state from the reset seed. It calls

    self._env.set_init_state(self._init_states[self.init_state_id % N])
    self.init_state_id += self._reset_stride        # _reset_stride == n_envs

so the initial state is a function of *how many times this sub-env has been
reset*, not of the seed. Separately, `LiberoEnv.step()` calls `self.reset()`
itself the moment an episode terminates, and `terminated` is set from
`check_success()`. Under a gymnasium vector env with the default NEXT_STEP
autoreset mode, that terminated slot is reset *again* on the following
`venv.step()` call.

Net effect: a slot that terminated once, early, consumes AT LEAST THREE initial
states per rollout (the explicit reset, the internal reset inside `step`, the
autoreset) while a slot that never terminated consumes ONE. "At least" is the
honest bound: after the autoreset the slot is running a fresh episode and is
still being stepped, so it can terminate again and consume more. The two slots
therefore begin the next rollout at different, outcome-dependent offsets. Which
initial states an experiment visits then depends on which policy happened to
succeed, so two policies cannot be compared on a fixed set of initial states by
reusing a vector env across rollouts.

HOW TERMINATION IS FORCED
-------------------------
`terminated = done or is_success`, so success is a sufficient cause of
termination but not the only one -- `done` from the underlying env is a second
path, and this script does not prove that path cannot fire. What it does is make
every reset visible, so an unforced termination would show up in the trace
rather than be assumed away. No real policy reliably succeeds inside a short
scripted budget, and this reproduction sends zero actions.

So the script MONKEY-PATCHES the underlying robosuite env's `check_success()` to
return True exactly once, for exactly one slot, at a fixed step. That patch is
announced on stdout when it fires. It is a stand-in for a policy that genuinely
solved the task; nothing else about the env is altered, and the reset
bookkeeping under test is untouched.

A SyncVectorEnv is used so the instrumentation prints from the main process. The
mechanism is the autoreset, not the vectorisation: AsyncVectorEnv defaults to the
same NEXT_STEP autoreset mode, so it should behave the same way, but only the
Sync path is exercised here.

Requires lerobot with the libero env extra and the LIBERO-Plus assets. Expect a
few minutes of startup -- loading the LIBERO-Plus suite and building two
offscreen renderers dominates the runtime.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

SUITE_NAME = "libero_spatial"
TASK_ID = 0
N_ENVS = 2
STEPS_PER_ROLLOUT = 6
FORCE_SUCCESS_SLOT = 0
FORCE_SUCCESS_AT = 2  # step index within rollout 1


def get_suite(name: str):
    """The suite object LiberoEnv wants, via lerobot's helper or LIBERO directly."""
    try:
        from lerobot.envs.libero import _get_suite

        return _get_suite(name)
    except ImportError:  # private API moved
        from libero.libero import benchmark

        return benchmark.get_benchmark_dict()[name]()


def main() -> None:
    import gymnasium as gym
    from lerobot.envs.libero import LiberoEnv

    print(f"gymnasium {gym.__version__}   suite={SUITE_NAME} task_id={TASK_ID} n_envs={N_ENVS}")
    suite = get_suite(SUITE_NAME)

    # cause[slot] labels why the next reset of that slot happened; the harness
    # sets it before each phase. events collects (slot, cause, index_used).
    state = {"t": -1, "cause": {}}
    events: list[tuple[int, str, int]] = []

    def instrument(env: LiberoEnv, slot: int) -> None:
        orig_reset, orig_step = env.reset, env.step

        def reset(*a, **kw):
            idx = env.init_state_id % len(env._init_states)
            out = orig_reset(*a, **kw)
            cause = state["cause"].get(slot, "?")
            events.append((slot, cause, idx))
            print(f"  slot{slot}  RESET  init_state_index={idx:<3} "
                  f"(init_state_id -> {env.init_state_id})   cause: {cause}")
            return out

        def step(action):
            forced = slot == FORCE_SUCCESS_SLOT and state["t"] == FORCE_SUCCESS_AT
            if forced:
                real_check = env._env.check_success
                env._env.check_success = lambda: True
                print(f"  slot{slot}  [MONKEY-PATCH] forcing check_success()->True at step "
                      f"{state['t']} to stand in for a real success")
            try:
                return orig_step(action)
            finally:
                if forced:
                    env._env.check_success = real_check

        env.reset, env.step = reset, step

    def build(slot: int):
        def _f():
            env = LiberoEnv(
                task_suite=suite,
                task_id=TASK_ID,
                task_suite_name=SUITE_NAME,
                episode_index=slot,   # each sub-env starts at its own init state
                n_envs=N_ENVS,        # -> _reset_stride
                is_libero_plus=True,
            )
            instrument(env, slot)
            return env

        return _f

    venv = gym.vector.SyncVectorEnv([build(i) for i in range(N_ENVS)])
    n_init = len(venv.envs[0]._init_states)
    print(f"autoreset_mode: {venv.metadata.get('autoreset_mode')}   "
          f"n_init_states: {n_init}")
    zero = np.zeros((N_ENVS, *venv.single_action_space.shape), dtype=np.float32)

    print("\nROLLOUT 1")
    state["cause"] = {s: "explicit venv.reset()" for s in range(N_ENVS)}
    venv.reset(seed=list(range(N_ENVS)))

    prev_term = np.zeros(N_ENVS, dtype=bool)
    for t in range(STEPS_PER_ROLLOUT):
        state["t"] = t
        state["cause"] = {
            s: ("autoreset (gymnasium NEXT_STEP, after last step terminated)"
                if prev_term[s] else "internal reset inside LiberoEnv.step() on termination")
            for s in range(N_ENVS)
        }
        _, _, term, trunc, _ = venv.step(zero)
        term, trunc = np.asarray(term), np.asarray(trunc)
        if term.any() or trunc.any():
            print(f"  step={t} terminated={term.tolist()} truncated={trunc.tolist()}")
        prev_term = term

    print("\nROLLOUT 2 (same venv, explicit reset)")
    state["t"] = -1
    state["cause"] = {s: "explicit venv.reset() at start of rollout 2" for s in range(N_ENVS)}
    venv.reset()
    venv.close()

    # ---- verdict -----------------------------------------------------------
    resets = Counter(slot for slot, _, _ in events)
    mid = Counter(slot for slot, cause, _ in events if not cause.startswith("explicit"))
    start2 = {slot: idx for slot, cause, idx in events if "rollout 2" in cause}
    other = next(s for s in range(N_ENVS) if s != FORCE_SUCCESS_SLOT)

    print("\n---- SUMMARY ----")
    for s in range(N_ENVS):
        print(f"  slot{s}: {resets[s]} resets total, {mid[s]} of them mid-rollout; "
              f"starts rollout 2 at init_state_index {start2.get(s)}")
    print(f"  counterfactual (no early termination): every slot s would start "
          f"rollout 2 at index s + n_envs = {[s + N_ENVS for s in range(N_ENVS)]}")

    two_resets = mid[FORCE_SUCCESS_SLOT] == 2 and mid[other] == 0
    diverged = start2.get(FORCE_SUCCESS_SLOT) != FORCE_SUCCESS_SLOT + N_ENVS
    print(f"\n  terminating slot got exactly 2 mid-rollout resets, other slot 0: {two_resets}")
    print(f"  terminating slot's rollout-2 start moved off the contiguous block: {diverged}")
    print("REPRODUCED" if two_resets and diverged else "NOT REPRODUCED")


if __name__ == "__main__":
    main()
