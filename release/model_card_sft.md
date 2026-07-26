---
license: apache-2.0
base_model: lerobot/smolvla_base
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - vla
  - vision-language-action
  - imitation-learning
  - lerobot
  - libero
---

# smolvla-sft-libero-spatial-heldout

SmolVLA supervised fine-tune (SFT) on LIBERO-Plus with **all camera-pose
perturbation categories held out** of the training data. This is the SFT
baseline — and the RL initialization — for the comparison study in the
repository below; it is released so the RL results can be reproduced from the
exact same starting point.

- **Repository:** https://github.com/Ono-Katsuki/SmolVLA_RL
- **Paper:** [10.5281/zenodo.21596933](https://doi.org/10.5281/zenodo.21596933) (preprint; not published or accepted anywhere)
- **RL fine-tune of this model:** `katsukiono/smolvla-libero-spatial-4arm`
  (subfolder `grpo/`)

## Training

| | |
|---|---|
| Base model | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) |
| Recipe | Identical to upstream [`lerobot/smolvla_libero_plus`](https://huggingface.co/lerobot/smolvla_libero_plus): 20k steps, batch 32, lr 1e-4 cosine |
| Data | `lerobot/libero_plus`, **minus all camera_pose categories** (leakage-safe split: 11,476 train / 2,871 held-out episodes) |
| Split provenance | Per-episode perturbation categories recovered from the RLDS source dataset; the labels ship in the repository (`data/episode_categories.csv`, all 14,347 episodes verified) |

The only difference from the upstream reference fine-tune is the held-out split;
this makes the model suitable for studying camera-viewpoint generalization and
for post-training comparisons (RS-SFT / flow-DPO / GRPO) from a common
checkpoint.

## Evaluation

<!-- Numbers below are final; the full 4-arm table is in the GRPO card. This model is the
     "SFT (baseline)" row. -->

Success rate 28.6% [15.2, 41.9] on the primary evaluation (7 trained tasks at held-out initial conditions, 105 episodes, task-bootstrap
95% CI, paired seeds).


> **Evaluation caveat (found in a late audit).** These are *not* held-out initial
> conditions. LIBERO-Plus selects the initial state from a fixed list indexed by
> sub-environment and reset count, ignoring the reset seed, so the evaluation's
> initial states are a subset of the ones the rollouts were collected from.
> RS-SFT's advantage in particular is confounded with memorising those
> trajectories. Read the table as a paired comparison on the training initial
> conditions, not as generalisation.

## Intended use and limitations

Research use only, in simulation (LIBERO / robosuite). Single training run;
camera-pose categories were deliberately excluded from training, so held-out
viewpoint performance is expected to be degraded. Note that no evaluation in the
study measures that gap cleanly: the trained-task evaluation holds the task set
fixed and varies only the initial conditions — which, per the caveat above, turned
out to be the training ones — and the in-dist vs held-out-camera comparison draws
from two *different* task pools, so its gap conflates viewpoint sensitivity with
task difficulty. The unseen-task/held-out-camera probe is small (24 episodes per
pool) and explicitly exploratory. The study's decisive result is a preregistered
unseen-task test of the GRPO fine-tune against this checkpoint, which returned a
null: 20/138 vs 19/138, mean per-task difference +0.72 pp, 95% task-clustered CI
[−5.07, +6.52] pp, exact McNemar p = 1.0, reported as NOT ESTABLISHED. Not
validated for physical robots.

## License

Apache-2.0 (matching the lerobot training code, the SmolVLM2 backbone, and the
upstream `lerobot/smolvla_libero_plus` derivative; `lerobot/smolvla_base` itself
carries no explicit license tag). Fine-tuning code in the repository is MIT.

## Citation

See the citation block in the repository's `CITATION.cff`, or the paper
(10.5281/zenodo.21596933).
