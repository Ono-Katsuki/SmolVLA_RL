"""flow-DPO: preference optimization on same-task success/failure episode pairs.

Applies the ELBO surrogate of Diffusion-DPO (Wallace+2023) to flow matching.
The four forwards of θ and the reference are given the identical noise / time
so the variance cancels in the differences (shared separately for the
winning and losing sides).

Usage:
    python -m src.post_train.train_dpo --config configs/post_train.yaml
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
from .data import assemble_batch, group_by_task, load_episodes
from .dpo_loss import flow_dpo_loss
from .train_rs_sft import load_policy_and_processors, save_checkpoint


@dataclass
class DpoConfig:
    data_dir: str
    sft_checkpoint: str
    output_dir: str
    model_type: str = "smolvla"
    steps: int = 300
    lr: float = 5e-6
    pairs_per_step: int = 4
    beta: float = 100.0            # coefficient matched to the flow loss difference (~1e-2 scale).
                                   # tune by watching metrics so mean_logits is O(1)
    sft_coef: float = 0.1          # winning-side SFT anchor (prevents unlearning, RPO style)
    seed: int = 0
    log_every: int = 10


def build_pairs(episodes) -> dict[str, tuple[list[int], list[int]]]:
    """Return (winning episode idxs, losing episode idxs) per task."""
    pairs: dict[str, tuple[list[int], list[int]]] = {}
    for task, idxs in group_by_task(episodes).items():
        wins = [i for i in idxs if episodes[i].success]
        losses = [i for i in idxs if not episodes[i].success]
        if wins and losses:
            pairs[task] = (wins, losses)
    return pairs


def train(cfg: DpoConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    episodes = load_episodes(Path(cfg.data_dir))
    pairs = build_pairs(episodes)
    n_win = sum(len(w) for w, _ in pairs.values())
    n_lose = sum(len(l) for _, l in pairs.values())
    print(f"[dpo] pairable tasks: {len(pairs)} (win {n_win} / lose {n_lose} episodes)")
    if not pairs:
        raise SystemExit(
            "[dpo] no task has both a win and a loss. Increase collect's rounds, or raise the success rate."
        )

    policy, preprocessor, postprocessor = load_policy_and_processors(cfg.sft_checkpoint, device)
    reference, _, _ = load_policy_and_processors(cfg.sft_checkpoint, device)
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()
    build_adapter(cfg.model_type, policy).freeze_backbone()
    policy.train()

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg.lr)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim
    action_dim = policy.config.action_feature.shape[0]  # actually executed dims (pad is zero-filled)
    tasks = list(pairs.keys())

    acc_sum = loss_sum = logit_sum = 0.0
    for step in range(1, cfg.steps + 1):
        # pick the task weighted by pair count
        weights = [len(pairs[t][0]) * len(pairs[t][1]) for t in tasks]
        task = rng.choices(tasks, weights=weights, k=1)[0]
        wins, losses = pairs[task]
        B = cfg.pairs_per_step
        win_samples = [
            (i, rng.randrange(len(episodes[i].obs)))
            for i in (wins[rng.randrange(len(wins))] for _ in range(B))
        ]
        lose_samples = [
            (i, rng.randrange(len(episodes[i].obs)))
            for i in (losses[rng.randrange(len(losses))] for _ in range(B))
        ]
        win_batch = assemble_batch(episodes, win_samples, device, action_dim=action_dim)
        lose_batch = assemble_batch(episodes, lose_samples, device, action_dim=action_dim)

        # θ / reference share the identical noise and time (separately for winning and losing sides)
        noise_w = torch.randn(B, chunk_size, max_action_dim, device=device)
        noise_l = torch.randn(B, chunk_size, max_action_dim, device=device)
        time_w = policy.model.sample_time(B, device)
        time_l = policy.model.sample_time(B, device)

        # SmolVLAPolicy.forward in lerobot 0.6.1 is annotated `-> dict`, but the
        # implementation returns a (loss, loss_dict) tuple. With
        # reduction="none" the first element is the per-sample flow-MSE of
        # shape (B,) (losses.sum(time,action)/num_valid). Verified against the
        # actual source.
        lw_theta, _ = policy.forward(dict(win_batch), noise_w, time_w, reduction="none")
        ll_theta, _ = policy.forward(dict(lose_batch), noise_l, time_l, reduction="none")
        with torch.no_grad():
            lw_ref, _ = reference.forward(dict(win_batch), noise_w, time_w, reduction="none")
            ll_ref, _ = reference.forward(dict(lose_batch), noise_l, time_l, reduction="none")

        out = flow_dpo_loss(
            lw_theta, ll_theta, lw_ref, ll_ref, beta=cfg.beta, sft_coef=cfg.sft_coef
        )
        optim.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optim.step()

        loss_sum += float(out.loss)
        acc_sum += float(out.implicit_acc)
        logit_sum += float(out.mean_logits)
        if step % cfg.log_every == 0:
            n = cfg.log_every
            info = {
                "step": step,
                "loss": loss_sum / n,
                "implicit_acc": acc_sum / n,
                "mean_logits": logit_sum / n,
            }
            loss_sum = acc_sum = logit_sum = 0.0
            with metrics_path.open("a") as f:
                f.write(json.dumps(info) + "\n")
            print(
                f"[dpo] step {step}/{cfg.steps} loss={info['loss']:.4f} "
                f"acc={info['implicit_acc']:.2f} logits={info['mean_logits']:.2f}"
            )

    save_checkpoint(policy, preprocessor, postprocessor, out_dir / "checkpoints" / "last")
    print(f"[dpo] saved checkpoint -> {out_dir}/checkpoints/last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--sft_checkpoint", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()
    raw = yaml.safe_load(args.config.open())["dpo"]
    for k in ("data_dir", "sft_checkpoint", "output_dir"):
        if getattr(args, k):
            raw[k] = getattr(args, k)
    train(DpoConfig(**raw))


if __name__ == "__main__":
    main()
