"""Rejection-sampling SFT (filtered BC): additional SFT on successful episodes only.

Uses only success=True episodes from the rollouts saved by collect.py and
fine-tunes the action expert with the flow-matching loss (policy.forward).

Usage:
    python -m src.post_train.train_rs_sft --config configs/post_train.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from ..grpo.adapters import build_adapter
from .data import assemble_batch, group_by_task, load_episodes, sample_task_minibatch


@dataclass
class RsSftConfig:
    data_dir: str
    sft_checkpoint: str
    output_dir: str
    model_type: str = "smolvla"
    steps: int = 400
    lr: float = 1e-5
    micro_batch_size: int = 8
    min_success: int = 4           # abort if there are fewer successful episodes than this
    seed: int = 0
    log_every: int = 20


def load_policy_and_processors(sft_checkpoint: str, device: torch.device):
    """Load policy + processors from a checkpoint without building an env."""
    import lerobot.policies  # noqa: F401  (registers into the draccus choice registry)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    from ..grpo.train_grpo import _resolve_pretrained_dir

    pretrained_dir = _resolve_pretrained_dir(sft_checkpoint)
    policy_cfg = PreTrainedConfig.from_pretrained(str(pretrained_dir))
    policy = get_policy_class(policy_cfg.type).from_pretrained(str(pretrained_dir))
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


def save_checkpoint(policy, preprocessor, postprocessor, ckpt_dir: Path) -> None:
    model_dir = ckpt_dir / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(model_dir))
    preprocessor.save_pretrained(str(model_dir), config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(str(model_dir), config_filename="policy_postprocessor.json")


def train(cfg: RsSftConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    episodes = load_episodes(Path(cfg.data_dir))
    wins = [ep for ep in episodes if ep.success]
    print(f"[rs_sft] episodes: {len(episodes)} total, {len(wins)} success")
    if len(wins) < cfg.min_success:
        raise SystemExit(
            f"[rs_sft] only {len(wins)} successful episodes (< {cfg.min_success})."
            " First raise the success rate with SFT/GRPO, or increase collect's rounds."
        )

    policy, preprocessor, postprocessor = load_policy_and_processors(cfg.sft_checkpoint, device)
    build_adapter(cfg.model_type, policy).freeze_backbone()
    policy.train()

    # Only same-task samples can be batch-concatenated (language token lengths match)
    by_task_eps = group_by_task(wins)
    by_task_pool = {
        t: [(i, c) for i in idxs for c in range(len(wins[i].obs))]
        for t, idxs in by_task_eps.items()
    }
    n_samples = sum(len(v) for v in by_task_pool.values())
    print(f"[rs_sft] chunk samples: {n_samples} across {len(by_task_pool)} tasks")

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg.lr)
    action_dim = policy.config.action_feature.shape[0]  # actually executed dims (pad is zero-filled)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    running = 0.0
    for step in range(1, cfg.steps + 1):
        _task, samples = sample_task_minibatch(by_task_pool, cfg.micro_batch_size, rng)
        batch = assemble_batch(wins, samples, device, action_dim=action_dim)
        loss, _ = policy.forward(batch)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optim.step()
        running += float(loss)
        if step % cfg.log_every == 0:
            avg = running / cfg.log_every
            running = 0.0
            info = {"step": step, "loss": avg}
            with metrics_path.open("a") as f:
                f.write(json.dumps(info) + "\n")
            print(f"[rs_sft] step {step}/{cfg.steps} loss={avg:.4f}")

    save_checkpoint(policy, preprocessor, postprocessor, out_dir / "checkpoints" / "last")
    print(f"[rs_sft] saved checkpoint -> {out_dir}/checkpoints/last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--sft_checkpoint", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()
    raw = yaml.safe_load(args.config.open())["rs_sft"]
    for k in ("data_dir", "sft_checkpoint", "output_dir"):
        if getattr(args, k):
            raw[k] = getattr(args, k)
    train(RsSftConfig(**raw))


if __name__ == "__main__":
    main()
