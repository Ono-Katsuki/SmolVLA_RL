"""Collect rollouts with the SDE policy and save them as success-labeled episodes.

Data collection shared by RS-SFT / DPO. Uses the same env / processor / SDE
path as GRPO.

Usage:
    python -m src.post_train.collect --config configs/post_train.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from ..grpo.policy_wrapper import build_sde_policy
from ..grpo.rollout import collect_group_rollouts
from ..grpo.train_grpo import (
    _build_processors,
    _camera_name_mapping,
    _load_policy,
    _make_task_env,
    _parse_task,
    _resolve_pretrained_dir,
)
from .data import save_episode


@dataclass
class CollectConfig:
    sft_checkpoint: str
    output_dir: str
    task_ids: list
    model_type: str = "smolvla"
    group_size: int = 8
    rounds: int = 4                # total episodes = len(task_ids) × group_size × rounds
    max_env_steps: int = 300
    num_sde_steps: int = 10
    noise_level: float = 0.5
    seed: int = 7
    render_skip: bool = False      # stop rendering at mid-chunk steps (enable after verifying with check_render_skip)


def collect(cfg: CollectConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    import lerobot.policies  # noqa: F401  (registers into the draccus choice registry)
    from lerobot.configs.policies import PreTrainedConfig

    pretrained_dir = _resolve_pretrained_dir(cfg.sft_checkpoint)
    tasks = [_parse_task(t) for t in cfg.task_ids]
    cam_map = _camera_name_mapping(PreTrainedConfig.from_pretrained(str(pretrained_dir)))
    print(f"[collect] camera_name_mapping={cam_map}")

    # Envs are "built one task at a time and closed when done" (in the loop
    # below). Holding all tasks at once means tasks×G async worker processes,
    # exhausting even 83 GB of RAM (measured: full at the 4th task).
    from lerobot.envs.configs import LiberoPlusEnv

    env_cfg = LiberoPlusEnv(task=tasks[0][0], task_ids=[tasks[0][1]], camera_name_mapping=cam_map)
    policy_cfg, backbone = _load_policy(pretrained_dir, env_cfg, device)
    processors = _build_processors(policy_cfg, env_cfg, pretrained_dir, device)
    policy = build_sde_policy(
        backbone,
        model_type=cfg.model_type,
        num_steps=cfg.num_sde_steps,
        noise_level=cfg.noise_level,
        freeze_backbone=False,  # collection only, no freezing needed
    )
    policy.eval()

    n_action_steps = backbone.config.n_action_steps
    original_action_dim = backbone.config.action_feature.shape[0]

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility header (once): git sha, key library versions, config
    try:
        import subprocess as _sp

        import gymnasium as _gym
        import lerobot as _lr

        sha = _sp.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
        (out_dir / "environment.json").write_text(json.dumps({
            "git_sha": sha,
            "torch": torch.__version__,
            "lerobot": getattr(_lr, "__version__", "?"),
            "gymnasium": _gym.__version__,
            "config": cfg.__dict__,
        }, indent=2, default=str))
    except Exception as e:  # noqa: BLE001
        print(f"[collect] environment.json skipped: {e}")

    episodes_log = (out_dir / "episodes.jsonl").open("a")
    idx = 0
    summary: dict[str, dict[str, int]] = {}
    # Task-major loop: build the env for one task, run all its rounds, then
    # close it (caps concurrent workers at G)
    for suite, tid in tasks:
        key = f"{suite}:{tid}"
        _, venv = _make_task_env(suite, tid, cam_map, n_envs=cfg.group_size)
        print(f"[collect] env ready: {key} (n_envs={cfg.group_size})")
        try:
            for r in range(cfg.rounds):
                t0 = time.perf_counter()
                group = collect_group_rollouts(
                    venv=venv,
                    policy=policy,
                    processors=processors,
                    task_key=key,
                    group_size=cfg.group_size,
                    max_steps=cfg.max_env_steps,
                    n_action_steps=n_action_steps,
                    original_action_dim=original_action_dim,
                    base_seed=cfg.seed + r * 10_000,
                    render_skip=cfg.render_skip,
                )
                for g_i, ep in enumerate(group):
                    save_episode(out_dir / f"ep_{idx:05d}.pt", ep)
                    episodes_log.write(json.dumps({
                        "ep_file": f"ep_{idx:05d}.pt",
                        "task_key": ep.task_key,
                        "round": r,
                        "group_index": g_i,
                        "seed": cfg.seed + r * 10_000 + g_i,
                        "success": bool(ep.success),
                        "staged_reward": float(ep.reward),
                        "n_env_steps": int(ep.n_env_steps),
                        "n_chunks": len(ep.obs_batches),
                        **{k: v for k, v in (ep.components or {}).items()},
                    }) + "\n")
                    episodes_log.flush()
                    idx += 1
                s = summary.setdefault(key, {"n": 0, "success": 0})
                s["n"] += len(group)
                s["success"] += sum(ep.success for ep in group)
                print(
                    f"[collect] round {r} {key}: {sum(ep.success for ep in group)}/{len(group)} "
                    f"success ({time.perf_counter() - t0:.0f}s)"
                )
        finally:
            venv.close()
        # Write incrementally (so per-task tallies survive an early stop).
        # Do not include "_overall" itself in the sum (it used to be double-counted, inflating n)
        per_task = {k: v for k, v in summary.items() if k != "_overall"}
        total = sum(s["n"] for s in per_task.values())
        wins = sum(s["success"] for s in per_task.values())
        summary["_overall"] = {"n": total, "success": wins}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    episodes_log.close()
    print(f"[collect] done: {summary.get('_overall')} -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--sft_checkpoint", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()
    raw = yaml.safe_load(args.config.open())["collect"]
    if args.sft_checkpoint:
        raw["sft_checkpoint"] = args.sft_checkpoint
    if args.output_dir:
        raw["output_dir"] = args.output_dir
    collect(CollectConfig(**raw))


if __name__ == "__main__":
    main()
