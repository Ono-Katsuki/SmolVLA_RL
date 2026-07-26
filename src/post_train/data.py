"""Rollout episode save/load + training batch assembly (shared by RS-SFT / DPO).

Storage format (torch.save, 1 episode = 1 file):
    {
      "task_key":    "libero_spatial:108",
      "success":     bool,
      "n_env_steps": int,
      "obs":         list[dict[str, Tensor]]   # per-chunk preprocessed observations (fp16, CPU)
      "actions":     Tensor (n_chunks, chunk_size, max_action_dim)  # normalized space (fp32)
    }

The observations have already passed through the policy preprocessor, so
feeding them to policy.forward as-is (converted back to fp32) makes the
flow-matching loss computation match training exactly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class EpisodeData:
    task_key: str
    success: bool
    n_env_steps: int
    obs: list          # list[dict[str, Tensor]] (fp16, CPU)
    actions: torch.Tensor  # (n_chunks, chunk_size, max_action_dim) fp32


def save_episode(path: Path, record) -> None:
    """grpo.rollout.EpisodeRecord → disk. Images compressed to ~half via fp16."""
    obs = [
        {k: (v.half() if v.is_floating_point() else v) for k, v in ob.items()}
        for ob in record.obs_batches
    ]
    actions = torch.cat([s.action for s in record.sde_rollouts], dim=0).float()  # (n_chunks, C, D)
    torch.save(
        {
            "task_key": record.task_key,
            "success": bool(record.success),
            "n_env_steps": int(record.n_env_steps),
            "obs": obs,
            "actions": actions,
            "components": dict(getattr(record, "components", {}) or {}),
        },
        path,
    )


def load_episodes(data_dir: Path) -> list[EpisodeData]:
    episodes = []
    for p in sorted(data_dir.glob("ep_*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=False)
        episodes.append(
            EpisodeData(
                task_key=d["task_key"],
                success=d["success"],
                n_env_steps=d["n_env_steps"],
                obs=d["obs"],
                actions=d["actions"],
            )
        )
    return episodes


def chunk_pool(episodes: list[EpisodeData]) -> list[tuple[int, int]]:
    """All (episode_idx, chunk_idx) combinations."""
    return [(i, c) for i, ep in enumerate(episodes) for c in range(len(ep.obs))]


def group_by_task(episodes: list[EpisodeData]) -> dict[str, list[int]]:
    by_task: dict[str, list[int]] = {}
    for i, ep in enumerate(episodes):
        by_task.setdefault(ep.task_key, []).append(i)
    return by_task


def assemble_batch(
    episodes: list[EpisodeData],
    samples: list[tuple[int, int]],
    device: torch.device | str,
    action_dim: int | None = None,
) -> dict:
    """Bundle same-task (episode, chunk) samples into a batch for policy.forward.

    The ACTION key carries action chunks in normalized space (the same space
    as the post-preprocessor observations, so forward's loss computation
    matches SFT).

    If action_dim is given, dimensions beyond it (the padding dimensions up to
    max_action_dim) are filled with 0. The rollout samples all max_action_dim
    dimensions via the SDE, but the environment only executes the first
    action_dim dimensions; the rest is stochastic noise irrelevant to
    behavior. Using it verbatim as a BC/DPO target could make the model
    memorize particular noise realizations.

    [Why zero-fill, not masking, is correct (verified against lerobot 0.6.1 source)]
    SmolVLA's SFT zero-pads actions via pad_vector, and the flow loss includes
    all 32 dimensions with no per-dim mask (u_t = noise - actions; padded dims
    have actions=0 so u_t=noise; losses include all 32 dims and num_valid is
    also ×32). Therefore zero-filling here makes the loss match SFT exactly.
    Applying a per-dim mask on the loss side would make RS-SFT/DPO a different
    loss from SFT and break the controlled comparison, so we do not mask. With
    zero-fill, the target for the padded dims is zero-mean noise, pushing the
    model toward outputting 0 (= same as SFT).
    """
    from lerobot.utils.constants import ACTION

    keys = episodes[samples[0][0]].obs[samples[0][1]].keys()
    batch: dict = {}
    for k in keys:
        v = torch.cat([episodes[i].obs[c][k] for i, c in samples], dim=0)
        # floats saved as fp16 go back to fp32; integers (tokens etc.) stay as-is
        batch[k] = (v.float() if v.is_floating_point() else v).to(device)
    actions = torch.stack([episodes[i].actions[c] for i, c in samples], dim=0)
    if action_dim is not None and actions.shape[-1] > action_dim:
        actions = actions.clone()
        actions[..., action_dim:] = 0.0
    batch[ACTION] = actions.to(device)
    return batch


def sample_task_minibatch(
    by_task: dict[str, list[tuple[int, int]]],
    batch_size: int,
    rng: random.Random,
) -> tuple[str, list[tuple[int, int]]]:
    """Pick a task weighted by pool size, then draw samples from it with replacement."""
    tasks = list(by_task.keys())
    weights = [len(by_task[t]) for t in tasks]
    task = rng.choices(tasks, weights=weights, k=1)[0]
    pool = by_task[task]
    return task, [pool[rng.randrange(len(pool))] for _ in range(batch_size)]
