**Title:** LIBERO evaluation: initial-state sequence depends on policy termination timing, via `LiberoEnv.step()`'s internal reset plus `NEXT_STEP` vector autoreset

### Summary

In `src/lerobot/envs/libero.py`, `LiberoEnv.reset()` advances the initial-state index every time it is called:

```python
raw_obs = self._env.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
self.init_state_id += self._reset_stride  # Change init_state_id when reset
```

and `LiberoEnv.step()` calls `reset()` itself when an episode ends:

```python
terminated = done or is_success
...
if terminated:
    self.reset()
return observation, reward, terminated, truncated, info
```

Because the increment lives inside `reset()`, the initial state a sub-env receives is a function of **how many times reset has been called**, not of how many episodes have been evaluated. Those counts can diverge: an early-terminated slot is reset here, and — with `NEXT_STEP` autoreset (which is what the LIBERO-Plus vector env reports) reset again when it is next stepped. `is_success` makes `terminated` outcome-dependent.

The consequence is that the initial-state index at the start of the next evaluated rollout is not solely a function of (vector slot, evaluated episode number, configured offset). It can depend on **when, and how often, the policy succeeded** while other slots were still running. Two policies evaluated with identical task ids, seeds, `n_episodes` and `batch_size` can therefore receive different initial states.

Under a paired evaluation protocol — McNemar, paired bootstrap, per-episode win/loss between two policies — that breaks the pairing assumption, and nothing in the output signals it.

### Why this is easy to miss

The failure is silent. Success rates look reasonable, per-episode records still line up by index, and the seeds match, so a paired analysis looks well-formed. The only visible symptom is that the arms diverge — which is indistinguishable from a real effect.

Related: `seed` does **not** select the initial state. `reset()` seeds and then `set_init_state` overwrites the simulator state. Protocols that vary the seed expecting to vary which initial condition is loaded are not doing so (seeding may still affect other randomness).

### Minimal reproduction

`scripts/repro_libero_init_state.py` in the repo below runs this with no arguments
and prints a `REPRODUCED` / `NOT REPRODUCED` verdict. It needs only lerobot and the
LIBERO-Plus assets. `libero_spatial` task 0, `n_envs=2`, so `_reset_stride=2`.
One slot's underlying `check_success()` is monkey-patched to return True once, to
stand in for a policy that actually solved the task; the reset bookkeeping under
test is untouched. Verbatim output on lerobot 0.6.1 / gymnasium 1.3.0:

```
gymnasium 1.3.0   suite=libero_spatial task_id=0 n_envs=2
autoreset_mode: AutoresetMode.NEXT_STEP   n_init_states: 50

ROLLOUT 1
  slot0  RESET  init_state_index=0   (init_state_id -> 2)   cause: explicit venv.reset()
  slot1  RESET  init_state_index=1   (init_state_id -> 3)   cause: explicit venv.reset()
  slot0  [MONKEY-PATCH] forcing check_success()->True at step 2 to stand in for a real success
  slot0  RESET  init_state_index=2   (init_state_id -> 4)   cause: internal reset inside LiberoEnv.step() on termination
  step=2 terminated=[True, False] truncated=[False, False]
  slot0  RESET  init_state_index=4   (init_state_id -> 6)   cause: autoreset (gymnasium NEXT_STEP, after last step terminated)

ROLLOUT 2 (same venv, explicit reset)
  slot0  RESET  init_state_index=6   (init_state_id -> 8)   cause: explicit venv.reset() at start of rollout 2
  slot1  RESET  init_state_index=3   (init_state_id -> 5)   cause: explicit venv.reset() at start of rollout 2

---- SUMMARY ----
  slot0: 4 resets total, 2 of them mid-rollout; starts rollout 2 at init_state_index 6
  slot1: 2 resets total, 0 of them mid-rollout; starts rollout 2 at init_state_index 3
  counterfactual (no early termination): every slot s would start rollout 2 at index s + n_envs = [2, 3]

  terminating slot got exactly 2 mid-rollout resets, other slot 0: True
  terminating slot's rollout-2 start moved off the contiguous block: True
REPRODUCED
```

The slot that ended early takes **two** mid-rollout resets — the internal one in
`LiberoEnv.step`, then the `NEXT_STEP` autoreset on the following `step()` call —
and begins rollout 2 at index 6, while the slot that ran to the step limit takes
none and begins at 3. A contiguous allocation would have given `[2, 3]`.
`truncated` stays `[False, False]` throughout, since `LiberoEnv` never sets it, so
running to the step limit causes no termination-triggered reset.

This was run through `SyncVectorEnv`. `AsyncVectorEnv` defaults to the same
`NEXT_STEP` mode so we would expect the same behaviour, but we have not verified
that path and do not claim it.

### Observed impact

We hit this in a paired 4-policy evaluation on `libero_spatial` (LIBERO-Plus, `batch_size=8`, `n_episodes=15` per task, identical seeds across policies). Reconstructing the visited indices from per-episode logs, wave 0 (episodes 0–7) is paired across policies, but episodes 8–14 — 7 of 15 per task — are not, because each policy's second wave starts from indices determined by its own wave-0 successes.

Restricting our analysis to the paired subset changed the conclusion: an exact McNemar p of 0.003 over the full sample became p = 0.092 on the 56 genuinely paired episodes per arm.

We offer this as motivation, not as proof — it is reconstructed downstream. The reproduction above is the evidence that matters.

### Suggested invariant

A regression test could assert:

```text
The kth evaluated episode in vector slot j uses an initial-state index determined
only by (j, k, configured offset) — never by termination timing.
```

### Provenance

This behaviour came in with #2899, which addressed #2375 ("each rollout will just reset the environment and start from the same initial state"). Before #2899 the index was assigned once and never advanced, so every episode reused one state; #2899 made it advance on reset.

This is not a claim that #2375 was reported incorrectly or that #2899 was the wrong fix — #2375 was accurate for the code at the time. The point is narrower: advancing on *reset* rather than on *evaluated episode* interacts with the internal reset in `step()` and with vector autoreset in a way that appears to reintroduce a different problem.

### Related sharp edge

`len(init_states)` is not uniform. Surveying all 2026 non-camera-perturbation tasks
of `libero_spatial` under LIBERO-Plus, we found it bimodal — 385 tasks have exactly
1 initial state and 1641 have 50, with nothing in between. On the N=1 tasks the
index wraps immediately, so every episode loads the same initial state and the run
measures no variation across initial conditions at all, however many episodes are
requested. It would help if that were surfaced rather than silently wrapped.

### Suggested directions

We are not sure what the maintainers would prefer, so rather than a patch:

- Make the initial state a function of the evaluated episode index rather than the reset count, so it cannot depend on outcomes.
- Expose the initial-state index in the eval API and record it per episode, so a paired protocol can assert the arms matched. Ours did not record it, and reconstructing it after the fact required deriving the mechanism from source.

We worked around it downstream by rebuilding the vector env once per wave with an explicit `episode_index` base, making the initial state a function of `(offset, wave, batch_size)` only. That is blunt — it pays env construction per wave — but it is outcome-independent.

### Environment

- lerobot **0.6.1**, commit `1bb9933215dcb7ffeeae6d3746cda3f73f5a59e2`
- gymnasium **1.3.0**; the LIBERO-Plus `AsyncVectorEnv` reports
  `metadata["autoreset_mode"] = <AutoresetMode.NEXT_STEP: 'NextStep'>`
- `--env.type=libero_plus`, suite `libero_spatial`
- Behaviour introduced by #2899

We will file the unrelated NumPy 2.0 failure we hit (`np.float_` removed) as a separate issue rather than dilute this one.
