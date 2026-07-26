"""Post-SFT post-training methods (sharing the rollout collection infrastructure with GRPO).

- collect.py      collect rollouts and save them to disk with success labels (shared by RS-SFT / DPO)
- train_rs_sft.py rejection-sampling SFT (filtered BC on successful episodes only)
- train_dpo.py    flow-DPO (Diffusion-DPO-style ELBO surrogate)
- dpo_loss.py     DPO loss (pure function, unit-testable)
- data.py         episode save/load and batch assembly
"""
