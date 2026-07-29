"""Collect SDE rollouts in LIBERO environments.

GRPO draws G episodes per prompt (= task) and normalizes rewards within the
group to obtain advantages. The G episodes run in parallel in an n_envs=G vec
env, with the SDE policy doing batched inference (on an A100 the rollout
wall time drops to ~1/G).

Observations and actions go through the same transform chain as lerobot-eval
(scripts/lerobot_eval.py::rollout):

    env obs → preprocess_observation → attach task → env_preprocessor
            → policy preprocessor → (action chunk via SDE rollout)
    action  → unpad → postprocessor (unnormalize) → env_postprocessor → env.step

Reward is +1 for success at episode end, 0 otherwise (standard LIBERO).
Envs that are done stop appending to their record (data after gymnasium's
autoreset is discarded). Note they are not frozen: the loop runs until *every*
slot is done, and policy.rollout / venv.step still cover them, so a done slot
goes on running a fresh episode whose transitions are dropped but whose resets
still advance init_state_id. See src/init_state_coverage.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .policy_wrapper import FlowMatchingSdePolicy, SdeRollout


COMPONENT_KEYS = (
    "approach", "reached", "grasped", "lifted", "probe_ready",
    "min_dist", "step_reached", "step_grasped", "step_lifted",
)


@dataclass
class EpisodeRecord:
    task_key: str
    obs_batches: list = field(default_factory=list)   # list[dict[str, Tensor]] (CPU, B=1)
    sde_rollouts: list = field(default_factory=list)  # list[SdeRollout] (CPU, B=1)
    success: bool = False
    reward: float = 0.0        # staged reward (equals success when unavailable)
    n_env_steps: int = 0
    components: dict = field(default_factory=dict)    # staged components / milestones (for diagnostic logging)


@dataclass
class GroupBatch:
    """Training batch bundling all groups' (per-chunk) samples into one."""
    obs_batches: list           # list[dict] — flatten over group × episode × chunk
    sde_rollouts: list          # list[SdeRollout]
    task_keys: list             # list[str] — task per sample (same task = same shapes, so used for minibatching)
    rewards: torch.Tensor       # (n_episodes,) — source of advantages (success or staged)
    successes: torch.Tensor     # (n_episodes,) — always binary success (for logging)
    advantages: torch.Tensor    # (n_samples,) — group-normalized, already broadcast to chunks


def tensors_to(d: dict, device: torch.device | str) -> dict:
    """Move only the tensors of a batch dict to device (drops task strings etc.)."""
    return {k: v.to(device) for k, v in d.items() if torch.is_tensor(v)}


def sde_to(sde: SdeRollout, device: torch.device | str) -> SdeRollout:
    return SdeRollout(
        action=sde.action.to(device),
        traj=sde.traj.to(device),
        v_traj=sde.v_traj.to(device),
        taus=sde.taus.to(device),
        dtau=sde.dtau,
        step_logprobs=sde.step_logprobs.to(device),
    )


def batch_slice(d: dict, i: int) -> dict:
    """Slice one sample out of batch dim 0 (B=G → B=1)."""
    return {k: v[i : i + 1] for k, v in d.items() if torch.is_tensor(v)}


def sde_slice(sde: SdeRollout, i: int) -> SdeRollout:
    """Slice out env i's component (B=1) of an SdeRollout. traj-like fields have batch at dim 1."""
    return SdeRollout(
        action=sde.action[i : i + 1],
        traj=sde.traj[:, i : i + 1],
        v_traj=sde.v_traj[:, i : i + 1],
        taus=sde.taus,
        dtau=sde.dtau,
        step_logprobs=sde.step_logprobs[:, i : i + 1],
    )


def cat_obs(items: list[dict], device: torch.device | str) -> dict:
    """Concatenate B=1 obs dicts along the batch dim (assumes same task: all tensors share shapes)."""
    keys = items[0].keys()
    return {k: torch.cat([it[k] for it in items], dim=0).to(device) for k in keys}


def cat_sde(items: list[SdeRollout], device: torch.device | str) -> SdeRollout:
    first = items[0]
    return SdeRollout(
        action=torch.cat([s.action for s in items], dim=0).to(device),
        traj=torch.cat([s.traj for s in items], dim=1).to(device),
        v_traj=torch.cat([s.v_traj for s in items], dim=1).to(device),
        taus=first.taus.to(device),
        dtau=first.dtau,
        step_logprobs=torch.cat([s.step_logprobs for s in items], dim=1).to(device),
    )


def _values_from_final_info(info: dict, key: str, n_envs: int) -> list[float | None]:
    """Extract an arbitrary per-env scalar from vec env info (handles all formats).

    gymnasium >= 1.0 vector envs have no final_info (removed in 1.0). The
    terminal step's info keys are aggregated directly as `info[key]` plus a
    presence mask `info["_"+key]`. Without this direct branch, staged_reward
    is always None and degrades to success (measured: reward==success matched
    exactly in the iter logs).
    """
    out: list[float | None] = [None] * n_envs
    if not isinstance(info, dict):
        return out
    if key in info:  # gymnasium >= 1.0: direct aggregation + "_<key>" mask
        a = np.asarray(info[key], dtype=object).reshape(-1)
        mask = np.asarray(info.get("_" + key, [True] * len(a))).reshape(-1)
        for i in range(min(n_envs, len(a))):
            if i < len(mask) and mask[i] and a[i] is not None:
                try:
                    out[i] = float(a[i])
                except (TypeError, ValueError):
                    pass
        return out
    final_info = info.get("final_info")
    if isinstance(final_info, dict):  # dict-of-arrays format in some versions
        arr = final_info.get(key)
        if arr is not None:
            a = np.asarray(arr, dtype=object).reshape(-1)
            for i in range(min(n_envs, len(a))):
                if a[i] is not None:
                    try:
                        out[i] = float(a[i])
                    except (TypeError, ValueError):
                        pass
        return out
    if final_info is not None:  # gymnasium < 1.0
        for i, item in enumerate(final_info):
            if i < n_envs and isinstance(item, dict) and key in item:
                try:
                    out[i] = float(item[key])
                except (TypeError, ValueError):
                    pass
    return out


def _successes_from_info(info: dict, n_envs: int) -> list[bool]:
    """Extract per-env is_success from vec env info (same search order as lerobot_eval)."""
    out = [False] * n_envs
    if not isinstance(info, dict):
        return out
    final_info = info.get("final_info")
    if isinstance(final_info, dict):  # gymnasium >= 1.0: dict of arrays
        arr = final_info.get("is_success")
        if arr is not None:
            a = np.asarray(arr).reshape(-1)
            for i in range(min(n_envs, len(a))):
                out[i] = bool(a[i])
        return out
    if final_info is not None:  # gymnasium < 1.0: per-env object array
        for i, item in enumerate(final_info):
            if i < n_envs and isinstance(item, dict) and "is_success" in item:
                out[i] = bool(item["is_success"])
        return out
    if "is_success" in info:
        a = np.asarray(info["is_success"]).reshape(-1)
        for i in range(min(n_envs, len(a))):
            out[i] = bool(a[i])
    return out


def collect_group_rollouts(
    venv,                              # gym.vector.VectorEnv (n_envs = group_size)
    policy: FlowMatchingSdePolicy,
    processors: dict,                  # env_preprocessor / env_postprocessor / preprocessor / postprocessor
    task_key: str,
    group_size: int,
    max_steps: int,
    n_action_steps: int,
    original_action_dim: int,
    base_seed: int = 0,
    render_skip: bool = False,         # stop camera rendering at mid-chunk steps (spawn_env.RenderSkipWrapper)
) -> list[EpisodeRecord]:
    """Collect 1 task × G episodes in parallel. Envs are reused via reset, not rebuilt."""
    from lerobot.envs import preprocess_observation
    from lerobot.utils.constants import ACTION

    G = venv.num_envs
    assert G == group_size, f"venv has {G} envs, expected group_size={group_size}"
    recs = [EpisodeRecord(task_key=task_key) for _ in range(G)]
    obs, info = venv.reset(seed=[base_seed + g for g in range(G)])
    try:
        env_max = venv.call("_max_episode_steps")[0]
        limit = min(max_steps, env_max) if env_max else max_steps
    except (AttributeError, NotImplementedError):
        limit = max_steps

    done = np.zeros(G, dtype=bool)
    success = np.zeros(G, dtype=bool)
    staged = np.full(G, np.nan)
    step = 0
    while not done.all() and step < limit:
        observation = preprocess_observation(obs)
        try:
            observation["task"] = list(venv.call("task_description"))
        except (AttributeError, NotImplementedError):
            observation["task"] = [""] * G
        observation = processors["env_preprocessor"](observation)
        observation = processors["preprocessor"](observation)

        sde = policy.rollout(observation)  # no_grad, B=G

        for i in np.flatnonzero(~done):
            recs[i].obs_batches.append(tensors_to(batch_slice(observation, int(i)), "cpu"))
            recs[i].sde_rollouts.append(sde_to(sde_slice(sde, int(i)), "cpu"))

        # Feed the chunk for n_action_steps to all envs (data from already-done envs is discarded).
        # render_skip: the only observation used for the next inference is the
        # one from the chunk's final step, so stop camera rendering at
        # intermediate steps and resume before the final step (saves up to T-1 frames)
        chunk = sde.action[:, :n_action_steps, :original_action_dim]
        T = chunk.shape[1]
        if render_skip and T > 1:
            try:
                venv.call("set_render_skip", True)
            except Exception:  # noqa: BLE001  (envs without the wrapper degrade to always rendering)
                render_skip = False
        for t in range(T):
            if render_skip and T > 1 and t == T - 1:
                venv.call("set_render_skip", False)  # render the final step fresh
            action = processors["postprocessor"](chunk[:, t, :])
            transition = processors["env_postprocessor"]({ACTION: action})
            action_np = transition[ACTION].to("cpu").numpy()
            obs, _reward, terminated, truncated, info = venv.step(action_np)
            step += 1
            just_done = (
                np.logical_or(np.asarray(terminated), np.asarray(truncated)).reshape(-1) & ~done
            )
            if just_done.any():
                succ_now = _successes_from_info(info, G)
                staged_now = _values_from_final_info(info, "staged_reward", G)
                comp_now = {
                    k: _values_from_final_info(info, f"staged_{k}", G) for k in COMPONENT_KEYS
                }
                for i in np.flatnonzero(just_done):
                    success[i] = succ_now[i]
                    if staged_now[i] is not None:
                        staged[i] = staged_now[i]
                    recs[i].n_env_steps = step
                    recs[i].components = {
                        k: v[i] for k, v in comp_now.items() if v[i] is not None
                    }
                done |= just_done
            if done.all() or step >= limit:
                break

    # LIBERO does not self-truncate: LiberoEnv.step returns truncated=False
    # unconditionally, and terminated = done or is_success -- overwhelmingly
    # success here, though the underlying env's done is a second possible cause
    # (this comment used to say "only on success"; corrected 2026-07-29). Envs
    # cut off externally at the step limit emit no terminal info, so call
    # StagedRewardWrapper.staged_now() directly to collect partial credit.
    # (Envs without the staged wrapper raise AttributeError
    # → degrade to success-only reward)
    if not done.all():
        try:
            res = venv.call("staged_now")
            for i in np.flatnonzero(~done):
                r = res[i]
                if isinstance(r, dict) and "staged_reward" in r:
                    staged[i] = float(r["staged_reward"])
                    recs[i].components = {
                        k: float(r[f"staged_{k}"]) for k in COMPONENT_KEYS if f"staged_{k}" in r
                    }
        except Exception:  # noqa: BLE001
            pass

    for i in range(G):
        recs[i].success = bool(success[i])
        recs[i].reward = float(staged[i]) if np.isfinite(staged[i]) else float(success[i])
        if recs[i].n_env_steps == 0:
            recs[i].n_env_steps = step
    return recs


def flatten_to_group_batch(
    episode_groups: list[list[EpisodeRecord]],
    reward_mode: str = "success",
    episode_equal_weight: bool = False,
) -> GroupBatch:
    """Bundle per-task groups into one batch and normalize advantages within each group.

    reward_mode="staged" uses the privileged-state staged reward
    (privileged.py). A group where all episodes share the same reward has
    std=0 → all-zero advantages, and the policy gradient vanishes (only the
    KL term remains). This is the correct GRPO behavior.

    episode_equal_weight=True corrects the chunk-weighting bias: the loss is a
    sample (chunk) average, so naively an episode's contribution is
    proportional to its chunk count. Failed episodes run to the limit and are
    long (many chunks) while successes end early (few chunks), so negative
    advantages are systematically overweighted (measured: mean_advantage in
    metrics was negative at every iter, -0.11 to -0.19). Multiplying each
    episode's advantage by (group-mean chunk count / own chunk count) gives
    episodes equal weight (overall scale is preserved via the mean chunk
    count).
    """
    obs_all: list = []
    sde_all: list = []
    task_keys: list[str] = []
    rewards: list[float] = []
    successes: list[float] = []
    advantages: list[float] = []

    for group in episode_groups:
        if reward_mode == "staged":
            rs = torch.tensor([float(ep.reward) for ep in group], dtype=torch.float32)
        else:
            rs = torch.tensor([float(ep.success) for ep in group], dtype=torch.float32)
        successes.extend(float(ep.success) for ep in group)
        # std floor 0.1: when within-group spread of the staged reward is tiny
        # (e.g. everyone at approach≈0.5), std normalization would amplify
        # noise into ±1-scale advantages and the gradient would chase noise;
        # the floor prevents this (relative to the [0,1] reward range)
        adv = (rs - rs.mean()) / rs.std().clamp(min=0.1)
        if episode_equal_weight:
            n_chunks = torch.tensor(
                [max(len(ep.obs_batches), 1) for ep in group], dtype=torch.float32
            )
            adv = adv * (n_chunks.mean() / n_chunks)
        for ep, a in zip(group, adv):
            for i in range(len(ep.obs_batches)):
                obs_all.append(ep.obs_batches[i])
                sde_all.append(ep.sde_rollouts[i])
                task_keys.append(ep.task_key)
                advantages.append(float(a))
        rewards.extend(rs.tolist())

    return GroupBatch(
        obs_batches=obs_all,
        sde_rollouts=sde_all,
        task_keys=task_keys,
        rewards=torch.tensor(rewards, dtype=torch.float32),
        successes=torch.tensor(successes, dtype=torch.float32),
        advantages=torch.tensor(advantages, dtype=torch.float32),
    )
