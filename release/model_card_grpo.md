---
license: apache-2.0
base_model: lerobot/smolvla_base
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - vla
  - vision-language-action
  - reinforcement-learning
  - grpo
  - flow-matching
  - lerobot
  - libero
---

# smolvla-grpo-libero-spatial

SmolVLA fine-tuned with **online reinforcement learning (Flow-SDE GRPO)** on the
`libero_spatial` suite of LIBERO-Plus, on a single-GPU budget (1x Colab A100,
20 iterations x 64 episodes/iteration). This is the RL arm of a small,
diagnostics-focused proof-of-concept study; see the preprint (10.5281/zenodo.21596933) and repository for the
full story, including the failure modes the study documents.

> **Headline result is a null.** On the study's preregistered test — this model
> against its own SFT initialization on 46 unseen tasks, 138 paired episodes per
> arm — the difference is +0.72 pp with a 95% task-clustered CI of
> [−5.07, +6.52] pp (exact McNemar p = 1.0). By the rule fixed in advance this is
> **NOT ESTABLISHED**: the RL stage is not shown to generalize better than the
> checkpoint it started from. Details under *Evaluation*.

- **Repository (code, configs, diagnostics):** https://github.com/Ono-Katsuki/SmolVLA_RL
- **Paper:** [10.5281/zenodo.21596933](https://doi.org/10.5281/zenodo.21596933) (preprint; not published or accepted anywhere)
- **SFT baseline checkpoint (direct parent of this model):**
  `katsukiono/smolvla-libero-spatial-4arm` (subfolder `sft/`)

## Model description

| | |
|---|---|
| Architecture | SmolVLA (~450M params): SmolVLM2 vision-language backbone + flow-matching action expert |
| Base model | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| Lineage | `lerobot/smolvla_base` → SFT on LIBERO-Plus (camera-pose categories held out) → **Flow-SDE GRPO (this model)** |
| Trained parameters | Action expert only (~100M); VLM frozen throughout RL |
| Action head | Flow matching; actions sampled by SDE (training) or ODE (deterministic eval) integration |
| Inputs | Multi-view RGB (front + wrist), proprioceptive state, language instruction |
| Outputs | Continuous action chunks |

## Training procedure (RL stage)

The RL stage ports the Flow-SDE construction of π_RL (arXiv:2510.25889) to
SmolVLA: the deterministic flow ODE is converted to an SDE whose 10 Gaussian
denoising transitions give a closed-form per-step log-probability, enabling a
PPO-style per-transition ratio and a GRPO objective (group-relative advantage, group
size 8 per task, advantage-std floor 0.1, clip 0.2, **KL coefficient 0** — no
anchor to the SFT policy, DanceGRPO-style).

Two corrections documented in the paper are baked into this checkpoint's run:

1. **No element-count ratio normalization.** Normalizing the joint log-ratio by
   the ~1600 action elements scales the RAW SURROGATE gradient by ~1/1600 at the rollout parameters. Under AdamW this does NOT imply a clean 1600x cut in optimizer step size; learning was frozen while the normalization was present and started after we removed it, which is an uncontrolled debugging association rather than a demonstrated mechanism.
2. **Episode-equal advantage weighting** (`episode_equal_weight`). Chunk-mean
   loss aggregation over-weights long failed episodes ~2x relative to short
   successes, applying a persistently negative mean advantage; each episode is
   re-weighted to contribute equally regardless of length.

Reward: a monotonically gated **staged privileged reward** from simulator state
(0.15 approach + 0.20 reached + 0.20 grasp_attempted + 0.20 lifted +
0.25 success), with a target-identity lock to prevent reward hacking.

Full config: [`configs/grpo_run8.yaml`](https://github.com/Ono-Katsuki/SmolVLA_RL/blob/main/configs/grpo_run8.yaml).

## Training data and environment

- **Environment:** LIBERO-Plus (arXiv:2510.13626), `libero_spatial` suite,
  8 tasks selected by an explicit reproducible rule from a screening pass
  (task ids 79, 108, 1477, 1530, 1817, 1955, 2126, 2172; LeRobot 0-based).
- **SFT stage data:** `lerobot/libero_plus` with all camera-pose perturbation
  categories held out (leakage-safe split; recovered per-episode category labels
  are published in the repository).
- **RL stage data:** on-policy SDE rollouts only (64 episodes/iteration,
  max 300 env steps/episode); no demonstrations used during RL.

## Evaluation


> **Evaluation caveat (found in a late audit).** These are *not* held-out initial
> conditions. LIBERO-Plus selects the initial state from a fixed list indexed by
> sub-environment and reset count, ignoring the reset seed, so the evaluation's
> initial states are a subset of the ones the rollouts were collected from.
> RS-SFT's advantage in particular is confounded with memorising those
> trajectories. Read the table as a paired comparison on the training initial
> conditions, not as generalisation.

**The result that decides the comparison is the preregistered unseen-task test
below, not this table.** The four-arm table that follows was the study's original
primary evaluation; the audit quoted above moved it out of that role.

**Trained-task evaluation (eval A), superseded as the primary result.** Four arms
on the same 7 trained tasks, 15 episodes per task = 105 paired episodes per arm,
using evaluation seed 900000. That seed was *chosen* to be disjoint from every
training seed in the belief that it would produce held-out initial conditions; it
does not, for the reason in the caveat above. Marginal success with task-bootstrap
95% CI, plus the paired task-clustered contrast against SFT.

| Method | success (task-boot. 95% CI) | paired Δ vs SFT (task-clustered 95% CI) | McNemar p |
|---|---|---|---|
| SFT (baseline) | 28.6% [15.2, 41.9] | — (reference) | — |
| RS-SFT | 42.9% [24.8, 61.9] | **+15 [+2.9, +28.6] pp*** | 0.003 |
| flow-DPO | 35.2% [22.9, 48.6] | +7 [−3.8, +16.2] pp | 0.230 |
| **GRPO (this model)** | 31.4% [17.1, 44.8] | +3 [−1.9, +7.6] pp | 0.648 |

**Read this before using the model: GRPO does _not_ significantly beat the SFT
baseline here** (p = 0.648; the interval includes zero). Rejection-sampling SFT
is the only arm whose improvement reaches significance in this table (and it
survives a Bonferroni correction over the three contrasts) — but that is exactly
the ordering the bias in the caveat above predicts, since RS-SFT behaviour-clones
successful trajectories from these very initial states, and the two effects
cannot be separated from this evaluation. A second consequence of the same
mechanism is that outcome-dependent resets leave only 56 of the 105 episodes per
arm genuinely paired; restricted to that subset, RS-SFT leads SFT by 12.5 pp but
the contrast is inconclusive (exact McNemar p = 0.092, task-clustered CI
[−3.6, +30.4] pp). Part of the explanation for GRPO is budget shape, not learning
rule: this GRPO run took one full-batch optimizer step per iteration, i.e. 20
parameter updates, against RS-SFT's 400, while consuming five times the rollouts.

**Primary result — preregistered unseen-task test (GRPO vs its own SFT
initialization).** Design, primary statistic and decision rules were committed
before a single episode was run
([preregistration](https://github.com/Ono-Katsuki/SmolVLA_RL/blob/main/data/eval/PREREGISTRATION_unseen_task_eval.md)).
On tasks that were never trained on, the initial-state confound cannot arise by
construction. 48 tasks were drawn; 2 errored for *both* arms and were dropped
under the prespecified rule, leaving 46 effective tasks x 3 paired initial states
= 138 episodes per arm.

| Arm | Successes | Success rate (task-boot. 95% CI) |
|---|---|---|
| SFT (the initialization) | 19/138 | 13.8% [7.3, 21.7] |
| **GRPO (this model)** | 20/138 | 14.5% [8.0, 21.7] |

Primary statistic: mean per-task success-rate difference (GRPO − SFT) =
**+0.72 pp**, 95% task-clustered bootstrap CI **[−5.07, +6.52] pp** (20000
resamples) — **the interval includes zero**. Secondary: exact McNemar on
discordant episode pairs (8 GRPO wins, 7 SFT wins) gives **p = 1.0**. By the
prespecified decision rule this is reported as **NOT ESTABLISHED**: the RL stage
is not shown to improve generalization over its own initialization. The interval
rules out an effect as large as the exploratory probe suggested, but it does
**not** establish equivalence or inferiority.

**Secondary, exploratory (unseen tasks).** 8 unseen tasks x 3 episodes = 24
episodes per pool, same pool and seeds across arms. GRPO has the highest point
estimate in both pools (29.2% in-dist, 20.8% held-out camera, vs ≤12.5% for the
others), but this probe was run after the primary eval, was not prespecified, and
establishes **no** between-arm difference: the paired GRPO−SFT interval includes
zero in both pools ([−4.2, +50.0] and [−4.2, +29.2] pp). Treat it as a hypothesis,
not a generalization result. 6 of GRPO's 7 in-dist successes come from two tasks
at 3/3. **This is the hypothesis the preregistered test above was written to
decide, and it did not replicate.**

Evaluation protocol: per-task evaluation, paired across methods by
*initial-state index* — not by seed, which controls only action sampling. The
preregistered run rebuilds the vector environment per wave so the initial state
is a function of (offset, wave, batch) only and never of an arm's own success
pattern; this is what was broken in eval A. Aggregate CIs use a task bootstrap
(episode-level Wilson intervals are optimistic under within-task correlation).
Note that the in-dist and held-out task pools differ, so their gap conflates
viewpoint sensitivity with task difficulty.

## Intended use and limitations

**Research use only, in simulation.**

- Trained and evaluated **exclusively in the LIBERO / robosuite simulator**;
  never deployed on physical hardware. Do not use to control a real robot
  without extensive additional validation.
- Proof-of-concept scale: a **single RL run (n=1)** of 20 iterations on a single
  suite (`libero_spatial`, 8 training tasks). No claim of generality across
  suites, embodiments, or seeds.
- The RL stage uses **privileged simulator state** for its reward; the approach
  does not transfer as-is to settings without such state.
- Trained with **no KL anchor** (kl_coef=0); the paper documents monotonically
  increasing drift from the SFT policy and a late-run performance plateau.
- The staged-reward milestone detectors have task-dependent false negatives
  (~20% of successes carry no lift flag), and "grasped" means
  "grasp_attempted" (gripper-close command within 6 cm).

## How to use

Load with [LeRobot](https://github.com/huggingface/lerobot) (pin the commit
given in the repository's `requirements.txt` for exact reproduction):

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("katsukiono/smolvla-libero-spatial-4arm", subfolder="grpo/")
```

Evaluation-ready wiring (env construction, camera-name mapping, paired seeds) is
in `src/eval_heldout.py` of the repository.

## License

Apache-2.0. The training code ([lerobot](https://github.com/huggingface/lerobot))
and the SmolVLM2 backbone are Apache-2.0, and Hugging Face's own derivative of
the same base (`lerobot/smolvla_libero_plus`) is published under Apache-2.0.
Note: the `lerobot/smolvla_base` model card itself carries no explicit license
tag; this card follows the Apache-2.0 licensing used by its official
derivatives. The fine-tuning code in the release repository is MIT.

## Citation

```bibtex
@article{ono2026smolvlagrpo,
  title   = {Low-Budget Online RL for a Flow-Matching VLA: A Flow-SDE GRPO
             Proof of Concept and the Diagnostics Behind It},
  author  = {Ono, Katsuki},
  doi     = {10.5281/zenodo.21596933},
  note    = {Preprint; not published or accepted anywhere},
  year    = {2026}
}
```

