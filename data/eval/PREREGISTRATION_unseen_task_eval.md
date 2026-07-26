# Preregistration — unseen-task generalization test (GRPO vs SFT)

> **Editorial note added at public release (2026-07-26) — the text below is
> unchanged.** Not one word of the preregistration itself has been edited; its
> value depends on that. Only its location changed, and two of the repository
> paths it cites therefore no longer resolve. For the reader:
>
> | Path as written below | Where it is now |
> |---|---|
> | `paper/data/…` | `data/eval/…` (this directory) |
> | `paper/preprint/main.tex` | **not in this repository** — the manuscript is unfinished and is a separate deliverable; see the README |
> | `paper/data/init_state_counts.json` | [`data/eval/init_state_counts.json`](init_state_counts.json) — the survey Amendment 1 rests on. It was a run artifact and had never been committed; it is committed now, because an amendment justified by a measurement is only checkable if the measurement is published. |
>
> The two raw per-arm outputs this study produced are in
> [`raw/`](raw/): `raw/sft_result.json` and `raw/grpo_result.json`, exactly as
> `src/eval_heldout.py` wrote them. They are the input to
> `analysis/analyze_unseen_prereg.py`, so the preregistered primary and secondary
> endpoints can be recomputed by anyone, not just by us.
>
> The commit that first added this file, and its timestamp relative to the run,
> are the actual evidence that the design was fixed in advance; both are in the
> git history.

**Written and committed BEFORE any episode of this study was run.** Commit this
file first; the run command below must not be executed until it is in git. The
point of the exercise is that every choice below was fixed while the outcome was
still unknown.

## Why this study exists

The trained-task evaluation (eval A) cannot support a generalization claim. Its
initial states were a subset of the states used to collect the RS-SFT training
data, because LIBERO-Plus selects the initial state from the environment index
rather than from the reset seed. See the confound paragraph in
`paper/preprint/main.tex` and `paper/data/eval_A_wave0_reanalysis.json`.

An offset into the initial-state list cannot repair this. With the `% N`
wraparound and the double-reset on success, at a realistic `N = 50` the training
run blocks 32 of 50 states and **no contiguous window of the required width is
free**. Evaluating on tasks that were never trained on removes the confound by
construction instead of working around it.

## Hypothesis (directional, fixed in advance)

GRPO (run8) attains a higher success rate than its own SFT initialization on
unseen LIBERO-Plus tasks.

## Arms — exactly two

| arm | checkpoint | role |
|---|---|---|
| `grpo` | run8, final iteration | the method under test |
| `sft`  | the checkpoint GRPO was initialized from | comparator |

**SFT is the comparator, chosen before running, for a stated reason:** it is the
policy GRPO started from, so the contrast isolates the effect of the RL stage and
nothing else. RS-SFT and DPO are *not* run here. Adding arms would reintroduce
the multiplicity problem that already forced the exploratory label onto the
earlier probe, and RS-SFT's advantage was measured on trained tasks under the
broken pairing, so it is not a meaningful reference here.

We record in advance that this is the *favourable* comparator for GRPO: RS-SFT
beat GRPO on trained tasks. This study therefore tests "does the RL stage improve
generalization over its initialization", **not** "is GRPO the best of the four".
The paper must state the claim in exactly those terms whatever the outcome.

## Design

- **Suite / pool**: `libero_spatial`, `--mode in_dist` (single pool — one test, no
  multiplicity correction needed). The `heldout` camera-viewpoint pool is *not*
  run; the two pools are different task sets, so their difference confounds
  viewpoint sensitivity with task difficulty.
- **Tasks**: 48, drawn by `random.seed(20260725); random.sample(pool, 48)`.
- **Exclusions applied before sampling** (`--exclude_task_ids`), so the sample
  cannot depend on results already seen:
  - 8 GRPO training tasks: `79,108,1477,1530,1817,1955,2126,2172`
  - 16 tasks used in the earlier exploratory probe:
    `1685,228,51,1894,563,501,457,285` (in_dist) and
    `935,665,620,748,733,722,679,660` (heldout)
- **Episodes**: 3 per task per arm → 144 per arm, **288 total**.
- **Initial states**: indices `{0,1,2}` for every task and both arms. The runner
  rebuilds the vector env per wave, so the initial state is a function of
  `(offset, wave, batch)` only and never of the arm's own success pattern. This
  is what was broken in eval A. `--init_state_offset` stays 0: on a task that was
  never trained on, every initial state is held out.
- **Pairing**: same task list, same initial-state indices, same seeds across arms.

### Why 48×3 and not the 24×3 the paper promised

Task count buys more power per episode than episodes-per-task here. Decomposing
the pilot's observed per-task spread (sd = 0.406) at 3 episodes gives measurement
noise 0.327 and true between-task heterogeneity 0.241 — most of the spread is the
coarseness of 3 episodes, and the clustered bootstrap's standard error falls as
1/√(tasks). Simulated power to exclude zero:

| design | total episodes | power @ pilot effect (0.208) | @ half (0.104) |
|---|---|---|---|
| 24 × 3 (as promised) | 144 | 0.72 | 0.27 |
| 24 × 8 | 384 | 0.90 | 0.41 |
| **48 × 3 (chosen)** | **288** | **0.93** | **0.43** |
| 32 × 10 | 640 | 0.97 | 0.53 |

Going from 24 to 48 tasks strengthens the promised design without changing its
shape; episodes per task stay at the promised 3.

**Stated honestly in advance: if the true effect is half the pilot estimate, no
design we can afford on a T4 resolves it** — even 640 episodes reaches only 0.53.
The pilot effect is itself likely inflated, because GRPO was picked as the best-
looking arm of an exploratory probe (winner's curse). A null result from this
study therefore does not establish that GRPO fails to generalize, and the paper
must not read it that way.

## Primary analysis — fixed in advance

1. **Primary**: mean per-task difference in success rate (GRPO − SFT) over the 48
   tasks, with a 95% task-clustered bootstrap CI (20000 resamples, seed 12345).
   The interval is the primary inferential object.
2. **Secondary**: exact McNemar binomial test on discordant episode pairs.
3. No subgroup analysis, no per-category breakdown, no dropping of tasks except
   the one prespecified exclusion rule below.

**Exclusion rule, fixed in advance**: a task is dropped only if it errors for
*both* arms (e.g. the deterministic renderer shape bug that removed task 1477
from eval A). A task that runs for one arm and errors for the other is *not*
dropped silently — the run is repaired and rerun, or the study reports the
imbalance. Dropped tasks are listed in the output JSON.

## Decision rule — fixed in advance

- CI excludes zero and the point estimate favours GRPO → report as a **confirmed
  generalization advantage over the SFT initialization**, with the CI, on 48
  preregistered unseen tasks.
- CI includes zero → report as **not established**, give the interval, and keep
  the paper's headline null. Do **not** re-slice, add arms, add pools, or add
  episodes and re-test. Any post-hoc analysis is labelled exploratory in the text.
- Either way the trained-task result stays as it is. This study cannot rescue it.

## Command

```bash
for ARM in grpo sft; do
  python -m src.eval_heldout \
    --checkpoint <ckpt_$ARM> --suite libero_spatial --mode in_dist \
    --engine inproc --tasks_per_suite 48 --eval_trials 3 \
    --eval_batch_size 8 --seed 20260725 --init_state_offset 0 \
    --exclude_task_ids 79,108,1477,1530,1817,1955,2126,2172,1685,228,51,1894,563,501,457,285,935,665,620,748,733,722,679,660 \
    --output_dir <drive>/eval_unseen_prereg/$ARM
done
```

Both arms must print the same `task_ids` list. If they differ, the run is void.

## Preconditions checked before running

- `--engine inproc` (the `cli` engine ignores initial-state control; the runner
  hard-exits on that combination).
- The runner reads `len(init_states)` on wave 0 and exits if `3` would wrap.
- Cached per-task files from an earlier protocol are rejected, not reused.
- Errored tasks are not written to the cache.

## Estimated cost

288 episodes ≈ 45 min of rollout on T4 plus ~15 min setup and model loading.

---

# Amendment 1 — 2026-07-25, after 0 episodes had been observed

**Status when this amendment was written: the study had been launched and had
aborted before a single episode completed.** Zero `task_*.json` files existed, no
`result.json` existed, and the log contained no `[eval]` lines. No outcome
information of any kind had been observed by anyone. That is the condition under
which an amendment is legitimate, and it is why this one is.

## What went wrong

The run aborted on its first task with:

```
libero_spatial:1816 has only 1 init states; offset 0+3 would wrap (% N) and
re-evaluate initial states already used in this same run.
```

This is the wave-0 guard listed under "Preconditions checked before running"
firing exactly as intended. The original design asked for 3 episodes per task but
never checked that every drawn task *has* 3 distinct initial states.

## What we measured before deciding

We surveyed `len(init_states)` for all 2026 tasks in the `libero_spatial`
`in_dist` pool (`paper/data/init_state_counts.json`; zero failures). The
distribution is **bimodal — every task has either N=1 or N=50**, with no
intermediate values:

| N | tasks |
|---|---|
| 1 | 385 |
| 50 | 1641 |

Of the 48 tasks the preregistered seed drew, **11 have N=1** and are infeasible at
3 episodes. That is close to the 9.2 expected from the pool's 19% N=1 rate, so the
draw was not unlucky — the original design was simply infeasible as written.
Lowering the episode requirement from 3 to 2 would rescue nothing, because no task
has N=2.

## The amendment

The sampling pool is restricted to tasks with `len(init_states) >= 3`, via a new
`--min_init_states 3`, applied **before** the exclusion list and **before**
`random.sample`. The seed (20260725), the task count (48), the episodes per task
(3), the two arms, the exclusion list, the pool, the primary statistic, and both
decision rules are **unchanged**. The draw is re-run under the same seed against
the filtered pool, which yields a different 48 tasks.

After exclusions the feasible pool holds 1628 tasks, so 48 is drawn with a margin
of 1580 and no risk of the filter distorting what is available.

## Why this does not compromise the preregistration

`len(init_states)` is a fixed property of the benchmark. It is knowable without
running a policy, it is identical for every arm, and it is independent of every
outcome. Filtering on it cannot encode any information about results, so it
creates no garden-of-forking-paths freedom. It enforces a precondition the
original document had already committed to; it does not relax one.

We record the alternatives we rejected, so the choice is auditable: lowering
`--eval_trials` below 3 (destroys the power the design was sized for, and would
not have worked anyway since no task has N=2); keeping the original draw and
dropping its 11 infeasible tasks (leaves 37 tasks, below the preregistered 48,
and yields a sample defined by an accident of the draw).

## Related finding, recorded here because it bears on eval A

The same survey shows that **2 of the 8 GRPO training tasks have N=1: 1817 and
1955.** Every episode ever run on those tasks --- rollout collection, GRPO
training, and all 15 eval-A episodes per arm --- used the identical initial state;
only policy stochasticity varied. Task 1817 is the single largest contributor to
RS-SFT's advantage in eval A. This is reported in the paper rather than used to
drop those tasks.
