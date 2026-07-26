"""GRPO training entry point.

Usage:
    python -m src.grpo.train_grpo --config configs/grpo_libero.yaml

Builds the policy / env / processors via the public lerobot 0.6.x API (same
path as lerobot-eval). Each env is built once per task and reused via reset.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from .grpo_loss import grpo_loss
from .policy_wrapper import build_sde_policy
from .rollout import (
    cat_obs,
    cat_sde,
    collect_group_rollouts,
    flatten_to_group_batch,
)


@dataclass
class TrainConfig:
    sft_checkpoint: str            # SFT'd ckpt (init + reference)
    output_dir: str
    task_ids: list                 # "suite:task_id" format (e.g. "libero_spatial:0")
    model_type: str = "smolvla"    # picked from the registry in adapters/__init__.py: smolvla | pi0 | pi05
    group_size: int = 8
    num_iterations: int = 200
    num_epochs_per_iter: int = 1   # passes over the rollout data (>1 increases off-policyness)
    accum_microbatches: int = 4    # micro-batches accumulated per optimizer step
    max_env_steps: int = 400
    num_sde_steps: int = 10
    noise_level: float = 0.5
    clip_eps: float = 0.2
    kl_coef: float = 0.04
    lr: float = 1e-5
    micro_batch_size: int = 8     # samples whose logprobs are recomputed together during the gradient update
    rollout_workers: int = 2      # tasks rolled out concurrently (process count = this × group_size)
    render_skip: bool = False     # stop camera rendering at mid-chunk steps (enable only in envs where check_render_skip verified equivalence)
    episode_equal_weight: bool = False  # chunk-weighting bias correction (equalizes the overweighted negative advantage of long failed episodes)
    reward_mode: str = "success"  # "success" (binary) | "staged" (privileged-state staged reward)
    freeze_vlm: bool = True
    seed: int = 0
    save_every: int = 10
    drive_mirror: str = ""        # if non-empty, replicate metrics / checkpoints here every time
                                  # (guards against Colab VM preemption; measured: the VM vanished 3 times in one night)
    resume: bool = True           # auto-resume from drive_mirror/ckpt_latest if present


def _parse_task(spec: str) -> tuple[str, int]:
    """"libero_spatial:0" → ("libero_spatial", 0)."""
    suite, _, tid = spec.partition(":")
    if not tid:
        raise ValueError(f"task_ids must be given in 'suite:task_id' form (got: {spec!r})")
    return suite, int(tid)


def _resolve_pretrained_dir(ckpt: str) -> Path:
    """Resolve a checkpoint spec to the actual directory containing config.json.

    On Drive's FUSE mount, the "last" symlink created by lerobot can appear
    broken; lerobot's from_pretrained then fails the local-path check and
    treats it as an HF repo id. Fall back to the sibling directory with the
    highest step number.
    """
    p = Path(ckpt)
    if not p.exists() and p.name == "last" and p.parent.is_dir():
        numbered = sorted(d for d in p.parent.iterdir() if d.name.isdigit() and d.is_dir())
        if numbered:
            print(f"[train_grpo] broken 'last' symlink; falling back to {numbered[-1]}")
            p = numbered[-1]
    p = p.resolve()
    for cand in (p / "pretrained_model", p):
        if (cand / "config.json").is_file():
            return cand
    raise FileNotFoundError(
        f"config.json not found: {ckpt} (point this at a checkpoint directory containing pretrained_model/)"
    )


def _camera_name_mapping(policy_cfg) -> dict | None:
    """Map the env's LIBERO cameras to the policy's visual feature names.

    Same convention as _camera_name_mapping in eval_heldout.py:
    sorted order gives agentview → eye-in-hand (front<wrist, camera1<camera2).
    """
    from lerobot.configs.types import FeatureType

    visuals = sorted(
        k.removeprefix("observation.images.")
        for k, v in policy_cfg.input_features.items()
        if v.type == FeatureType.VISUAL and "empty_camera" not in k
    )
    if len(visuals) < 2:
        return None
    return {"agentview_image": visuals[0], "robot0_eye_in_hand_image": visuals[1]}


def _make_task_env(suite: str, task_id: int, camera_name_mapping: dict | None, n_envs: int):
    """Build a vec env for one task (n_envs=group_size). Returns: (env_cfg, vec_env).

    - n_envs>1 is async (process parallel). Synchronous would serialize MuJoCo
      steps at ~11 min/group.
    - Workers are started with spawn. Forking after CUDA init deadlocks
      (measured: the 3rd env build hung for 60 minutes).
    """
    from .spawn_env import make_spawn_vec_env

    return make_spawn_vec_env(suite, task_id, camera_name_mapping, n_envs, staged_reward=False)


def _load_policy(pretrained_dir: Path, env_cfg, device: torch.device):
    """Build the policy from a checkpoint (same path as lerobot-eval)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy

    policy_cfg = PreTrainedConfig.from_pretrained(str(pretrained_dir))
    policy_cfg.pretrained_path = str(pretrained_dir)
    policy_cfg.device = str(device)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.to(device)
    return policy_cfg, policy


def _build_processors(policy_cfg, env_cfg, pretrained_dir: Path, device: torch.device) -> dict:
    from lerobot.envs import make_env_pre_post_processors
    from lerobot.policies import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )
    return {
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "env_preprocessor": env_preprocessor,
        "env_postprocessor": env_postprocessor,
    }


def _save_checkpoint(actor, processors, ckpt_dir: Path) -> None:
    """Save in a format that lerobot-eval / eval_heldout.py can read directly."""
    model_dir = ckpt_dir / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    actor.policy.save_pretrained(str(model_dir))
    processors["preprocessor"].save_pretrained(
        str(model_dir), config_filename="policy_preprocessor.json"
    )
    processors["postprocessor"].save_pretrained(
        str(model_dir), config_filename="policy_postprocessor.json"
    )


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    pretrained_dir = _resolve_pretrained_dir(cfg.sft_checkpoint)
    tasks = [_parse_task(t) for t in cfg.task_ids]

    # ---- Read the policy config first to decide the camera mapping, then build envs ----
    # Importing lerobot.policies registers SmolVLAConfig etc. into draccus's
    # choice registry. Without it, from_pretrained raises KeyError: 'smolvla'.
    import lerobot.policies  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig

    cam_map = _camera_name_mapping(PreTrainedConfig.from_pretrained(str(pretrained_dir)))
    print(f"[train_grpo] camera_name_mapping={cam_map}")

    # Envs are built and closed per task at rollout time (holding all tasks at
    # once means tasks×G async worker processes and RAM exhaustion — measured
    # 83 GB full). Only the env_cfg for building the policy/processors is
    # prepared up front.
    from lerobot.envs.configs import LiberoPlusEnv

    env_cfg = LiberoPlusEnv(
        task=tasks[0][0], task_ids=[tasks[0][1]], camera_name_mapping=cam_map
    )

    # ---- Auto-resume after VM preemption ----
    # Every iter's checkpoint and ITER.txt (completed iter count) are mirrored
    # to drive_mirror/ckpt_latest; on restart the actor is initialized from
    # there and iteration continues. The reference is always the SFT
    # checkpoint (the KL anchor is invariant across resumes).
    mdir = Path(cfg.drive_mirror) if cfg.drive_mirror else None
    start_iter = 0
    actor_init_dir = pretrained_dir
    optimizer_restored = False
    if cfg.resume and mdir is not None:
        marker = mdir / "ckpt_latest" / "ITER.txt"
        cand = mdir / "ckpt_latest" / "pretrained_model"
        if marker.exists() and (cand / "config.json").is_file():
            start_iter = int(marker.read_text().strip())
            actor_init_dir = cand
            print(f"[train_grpo] resuming: actor from {cand}, starting at iter {start_iter}")

    # ---- actor (gradient target) and reference (frozen SFT) ----
    policy_cfg, actor_backbone = _load_policy(actor_init_dir, env_cfg, device)
    _, reference_backbone = _load_policy(pretrained_dir, env_cfg, device)

    processors = _build_processors(policy_cfg, env_cfg, pretrained_dir, device)

    actor = build_sde_policy(
        actor_backbone,
        model_type=cfg.model_type,
        num_steps=cfg.num_sde_steps,
        noise_level=cfg.noise_level,
        freeze_backbone=cfg.freeze_vlm,
    )
    # The reference is 100% frozen by design. freeze_backbone=True would
    # false-trigger the adapter's "all frozen = misconfiguration" sanity
    # assert, so freeze everything after construction instead.
    reference = build_sde_policy(
        reference_backbone,
        model_type=cfg.model_type,
        num_steps=cfg.num_sde_steps,
        noise_level=cfg.noise_level,
        freeze_backbone=False,
    )
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()

    n_action_steps = actor_backbone.config.n_action_steps
    original_action_dim = actor_backbone.config.action_feature.shape[0]

    trainable = [p for p in actor.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg.lr)

    # On resume, restore AdamW's 1st/2nd moments as well. Restoring only the
    # weights with a freshly built optimizer makes the first post-resume step
    # oversized (bias correction not yet effective), so it would not be a
    # faithful continuation from the same iter (this matters especially here,
    # where the VM is preempted frequently).
    if start_iter > 0 and mdir is not None:
        opt_path = mdir / "ckpt_latest" / "optim.pt"
        if opt_path.is_file():
            try:
                optim.load_state_dict(torch.load(opt_path, map_location=device))
                optimizer_restored = True
                print(f"[train_grpo] restored optimizer state from {opt_path}")
            except Exception as e:  # noqa: BLE001  (continue with a fresh AdamW even if corrupted)
                print(f"[train_grpo] optimizer restore failed, using fresh AdamW: {e}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility header (one line appended per launch: resume status, git sha, library versions, config)
    try:
        import subprocess as _sp

        import gymnasium as _gym
        import lerobot as _lr

        sha = _sp.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
        header = {
            "git_sha": sha,
            "torch": torch.__version__,
            "lerobot": getattr(_lr, "__version__", "?"),
            "gymnasium": _gym.__version__,
            "resume_from_iter": start_iter,
            "optimizer_restored": optimizer_restored,
            "config": cfg.__dict__,
        }
        with (out_dir / "environment.jsonl").open("a") as f:
            f.write(json.dumps(header, default=str) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[train_grpo] environment header skipped: {e}")

    for it in range(start_iter, cfg.num_iterations):
        # ---- 1. rollout collection ----
        # Roll out tasks with rollout_workers-way parallelism. MuJoCo steps +
        # rendering run in the async worker processes, and the main thread
        # mostly waits on pipes (GIL released), so threads parallelize fine.
        # 1 venv = G worker processes, so the concurrent process count is
        # rollout_workers × G (set it based on RAM and core count).
        t0 = time.perf_counter()
        actor.eval()

        def _rollout_task(suite: str, tid: int):
            if cfg.reward_mode == "staged":
                from .privileged import make_staged_vec_env

                _, venv = make_staged_vec_env(suite, tid, cam_map, n_envs=cfg.group_size)
            else:
                _, venv = _make_task_env(suite, tid, cam_map, n_envs=cfg.group_size)
            try:
                return collect_group_rollouts(
                    venv=venv,
                    policy=actor,
                    processors=processors,
                    task_key=f"{suite}:{tid}",
                    group_size=cfg.group_size,
                    max_steps=cfg.max_env_steps,
                    n_action_steps=n_action_steps,
                    original_action_dim=original_action_dim,
                    base_seed=cfg.seed + it * 10_000,
                    render_skip=cfg.render_skip,
                )
            finally:
                venv.close()

        if cfg.rollout_workers > 1:
            with ThreadPoolExecutor(max_workers=cfg.rollout_workers) as ex:
                episode_groups = list(ex.map(lambda t: _rollout_task(*t), tasks))
        else:
            episode_groups = [_rollout_task(suite, tid) for suite, tid in tasks]
        batch = flatten_to_group_batch(
            episode_groups,
            reward_mode=cfg.reward_mode,
            episode_equal_weight=cfg.episode_equal_weight,
        )
        rollout_s = time.perf_counter() - t0

        # Per-episode diagnostic log (with iter). analyze_episodes can then
        # track the per-iter funnel (which stage fails) and the evolution of
        # reward separation. Also mirrored to Drive.
        ep_lines = []
        for group in episode_groups:
            for g_i, ep in enumerate(group):
                ep_lines.append(json.dumps({
                    "iter": it,
                    "task_key": ep.task_key,
                    "group_index": g_i,
                    "seed": cfg.seed + it * 10_000 + g_i,
                    "success": bool(ep.success),
                    "staged_reward": float(ep.reward),
                    "n_env_steps": int(ep.n_env_steps),
                    "n_chunks": len(ep.obs_batches),
                    **{k: v for k, v in (ep.components or {}).items()},
                }) + "\n")
        with (out_dir / "episodes.jsonl").open("a") as f:
            f.writelines(ep_lines)
        if cfg.drive_mirror:
            try:
                _m = Path(cfg.drive_mirror)
                _m.mkdir(parents=True, exist_ok=True)
                with (_m / "episodes.jsonl").open("a") as f:
                    f.writelines(ep_lines)
            except OSError as e:
                print(f"[train_grpo] drive_mirror episodes failed: {e}")
        n_samples = len(batch.obs_batches)
        if n_samples == 0:
            continue

        # ---- 2. Gradient update (multiple epochs, same-task samples concatenated into micro-batches) ----
        # Language token length is constant per task, so only same-task samples
        # can be cat'ed along the batch dim. grpo_loss accepts (N, B) directly.
        # Gradients are computed while staying in the same eval mode as rollout
        # (with train(), mode differences like dropout would make logp_old and
        # logp_new diverge even under identical parameters and falsely trigger
        # the clip)
        t1 = time.perf_counter()
        loss_sum = kl_sum = clip_sum = drift_sum = grad_norm_sum = 0.0
        n_updates = 0
        by_task: dict[str, list[int]] = {}
        for i, tk in enumerate(batch.task_keys):
            by_task.setdefault(tk, []).append(i)
        # Keep optimizer steps per rollout to a few or a dozen (standard
        # PPO/GRPO practice). Previously "2 epochs × step per micro-batch" =
        # 168 steps/iter; since the clip barely binds the normalized ratio,
        # the policy was swept far each iter and success collapsed
        # monotonically from 0.23 to 0.02 (run3 measurement).
        n_optim_steps = 0
        for _epoch in range(cfg.num_epochs_per_iter):
            microbatches: list[list[int]] = []
            for idxs in by_task.values():
                idxs = idxs[:]
                random.shuffle(idxs)
                for s in range(0, len(idxs), cfg.micro_batch_size):
                    microbatches.append(idxs[s : s + cfg.micro_batch_size])
            random.shuffle(microbatches)
            n_mb = len(microbatches)
            optim.zero_grad()
            pending = 0
            for mi, idxs in enumerate(microbatches):
                obs = cat_obs([batch.obs_batches[i] for i in idxs], device)
                sde = cat_sde([batch.sde_rollouts[i] for i in idxs], device)
                adv = batch.advantages[torch.tensor(idxs)].to(device)

                logp_old = sde.step_logprobs.detach()             # (N, B)
                logp_new = actor.recompute_logprobs(obs, sde)     # (N, B) — with gradients
                with torch.no_grad():
                    logp_ref = reference.recompute_logprobs(obs, sde)

                out = grpo_loss(
                    logp_new=logp_new,
                    logp_old=logp_old,
                    logp_ref=logp_ref,
                    advantages=adv,
                    n_elem=sde.action.shape[-2] * sde.action.shape[-1],
                    clip_eps=cfg.clip_eps,
                    kl_coef=cfg.kl_coef,
                )
                # Average by dividing by the actual size of the accumulation
                # window this micro-batch belongs to. Dividing by
                # cfg.accum_microbatches would underscale the gradient by
                # (42/64) when accum(64) > actual micro-batch count(~42), and
                # also underweight the remainder window.
                win_start = (mi // cfg.accum_microbatches) * cfg.accum_microbatches
                win_size = min(win_start + cfg.accum_microbatches, n_mb) - win_start
                (out.loss / win_size).backward()
                pending += 1
                if pending == cfg.accum_microbatches or mi == n_mb - 1:
                    gn = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    grad_norm_sum += float(gn)   # pre-clip norm (direct indicator of whether a learning signal exists)
                    optim.step()
                    optim.zero_grad()
                    pending = 0
                    n_optim_steps += 1

                loss_sum += float(out.loss)
                kl_sum += float(out.kl_loss)
                clip_sum += float(out.clip_fraction)
                drift_sum += float(out.drift)
                n_updates += 1
        update_s = time.perf_counter() - t1

        # ---- 3. logging + ckpt ----
        info = {
            "iter": it,
            "success_rate": float(batch.successes.mean()),
            "mean_reward": float(batch.rewards.mean()),
            "n_episodes": int(batch.rewards.numel()),
            "n_samples": n_samples,
            "mean_advantage": float(batch.advantages.mean()),
            "loss": loss_sum / max(n_updates, 1),
            "kl": kl_sum / max(n_updates, 1),
            "clip_fraction": clip_sum / max(n_updates, 1),
            "drift": drift_sum / max(n_updates, 1),  # displacement from SFT (is it learning?)
            "grad_norm": grad_norm_sum / max(n_optim_steps, 1),  # pre-clip gradient norm
            "n_optim_steps": n_optim_steps,
            "rollout_s": round(rollout_s, 1),
            "update_s": round(update_s, 1),
        }
        with (out_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(info) + "\n")
        print(
            f"[iter {it}] success={info['success_rate']:.3f} reward={info['mean_reward']:.3f} n={n_samples} "
            f"loss={info['loss']:.4f} kl={info['kl']:.6f} drift={info['drift']:.3f} clip={info['clip_fraction']:.2f} "
            f"(rollout {rollout_s:.0f}s / update {update_s:.0f}s)"
        )
        if cfg.drive_mirror:
            try:
                mdir = Path(cfg.drive_mirror)
                mdir.mkdir(parents=True, exist_ok=True)
                with (mdir / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(info) + "\n")
            except OSError as e:  # do not stop training on transient Drive FUSE failures
                print(f"[train_grpo] drive_mirror metrics failed: {e}")

        if (it + 1) % cfg.save_every == 0 or (it + 1) == cfg.num_iterations:
            _save_checkpoint(actor, processors, out_dir / f"iter_{it + 1:05d}")

        if mdir is not None:
            # Every iter, mirror only the single latest checkpoint by
            # overwriting "ckpt_latest" (caps steady-state usage at ~1.8 GB
            # even with 2.5 GB of Drive space left). Even if the VM is
            # preempted, resume=True continues from here, so at most one
            # iteration is lost.
            try:
                import shutil

                local_latest = out_dir / "ckpt_latest"
                _save_checkpoint(actor, processors, local_latest)
                torch.save(optim.state_dict(), local_latest / "optim.pt")  # restored on resume
                mirror_dir = mdir / "ckpt_latest"
                shutil.copytree(local_latest, mirror_dir, dirs_exist_ok=True)
                (mirror_dir / "ITER.txt").write_text(f"{it + 1}\n")
                print(f"[train_grpo] ckpt_latest mirrored (iter {it + 1})")
            except OSError as e:
                print(f"[train_grpo] drive_mirror checkpoint failed: {e}")




def _parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--sft_checkpoint", type=str, default=None, help="overrides the config")
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()
    with args.config.open() as f:
        raw = yaml.safe_load(f)
    if args.sft_checkpoint:
        raw["sft_checkpoint"] = args.sft_checkpoint
    if args.output_dir:
        raw["output_dir"] = args.output_dir
    # Legacy config compatibility: chunk_size now comes from the policy config, so ignore it
    raw.pop("chunk_size", None)
    return TrainConfig(**raw)


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
