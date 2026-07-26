"""Generic Flow-Matching SDE policy.

Model-specific API differences are encapsulated in `adapters/`; this module
only runs "compute prefix → predict velocity at each τ → SDE step" through
the Adapter interface.

Supported: smolvla / pi0 / pi05 (see adapters/). Add models in adapters/__init__.py.

Usage:
    from lerobot.policies.factory import make_policy
    from src.grpo.policy_wrapper import build_sde_policy

    base = make_policy(pretrained_path=".../pretrained_model").to("cuda")
    sde  = build_sde_policy(base, model_type="smolvla",
                            num_steps=10, noise_level=0.5, freeze_backbone=True)
    rollout = sde.rollout(batch)          # no gradients
    logp_new = sde.recompute_logprobs(batch, rollout)   # with gradients
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .adapters import FlowMatchingAdapter, build_adapter
from .flow_sde import recompute_step_logprob, sde_step


@dataclass
class SdeRollout:
    """SDE denoising record for one batch."""
    action: torch.Tensor              # (B, chunk, action_dim) — final x
    traj: torch.Tensor                # (N+1, B, chunk, action_dim) — x at each τ
    v_traj: torch.Tensor              # (N,   B, chunk, action_dim) — v_θ at each step
    taus: torch.Tensor                # (N,) — τ at each step
    dtau: float                       # constant (negative)
    step_logprobs: torch.Tensor       # (N, B) — log π(x_{τ+δ}|x_τ) at each step


class FlowMatchingSdePolicy(nn.Module):
    """Model-agnostic SDE rollout + log-prob recomputation."""
    def __init__(
        self,
        adapter: FlowMatchingAdapter,
        num_steps: int = 10,
        noise_level: float = 0.5,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.adapter = adapter
        self.num_steps = num_steps
        self.noise_level = noise_level
        if freeze_backbone:
            self.adapter.freeze_backbone()

    @property
    def policy(self):
        return self.adapter.policy

    # ------------------------------------------------------------
    # rollout / logprob recompute
    # ------------------------------------------------------------
    @torch.no_grad()
    def rollout(self, batch: dict) -> SdeRollout:
        """batch is the observation dict after passing through the policy preprocessor."""
        inputs = self.adapter.prepare_inputs(batch)
        prefix = self.adapter.prefix_features(inputs)
        B = inputs["state"].shape[0]
        device = inputs["state"].device
        chunk, action_dim = self.adapter.action_shape(batch)

        x = torch.randn(B, chunk, action_dim, device=device)
        dtau = -1.0 / self.num_steps

        traj = [x]
        v_list, tau_list, logp_list = [], [], []
        for k in range(self.num_steps):
            tau_val = 1.0 + dtau * k
            tau = torch.full((B,), tau_val, device=device)
            v = self.adapter.predict_velocity(prefix, x, tau)
            res = sde_step(x, v, tau, dtau, self.noise_level)
            x = res.x_next
            traj.append(x)
            v_list.append(v)
            tau_list.append(tau_val)
            logp_list.append(res.log_prob)

        return SdeRollout(
            action=x,
            traj=torch.stack(traj, dim=0),
            v_traj=torch.stack(v_list, dim=0),
            taus=torch.tensor(tau_list, device=device),
            dtau=dtau,
            step_logprobs=torch.stack(logp_list, dim=0),
        )

    def recompute_logprobs(self, batch: dict, rollout: SdeRollout) -> torch.Tensor:
        inputs = self.adapter.prepare_inputs(batch)
        prefix = self.adapter.prefix_features(inputs)
        out = []
        for k in range(self.num_steps):
            x = rollout.traj[k]
            x_next = rollout.traj[k + 1]
            tau_val = float(rollout.taus[k])
            tau = torch.full((x.shape[0],), tau_val, device=x.device)
            v = self.adapter.predict_velocity(prefix, x, tau)
            logp = recompute_step_logprob(x, x_next, v, tau, rollout.dtau, self.noise_level)
            out.append(logp)
        return torch.stack(out, dim=0)


# ------------------------------------------------------------
# Factory
# ------------------------------------------------------------
def build_sde_policy(
    policy,
    model_type: str = "smolvla",
    num_steps: int = 10,
    noise_level: float = 0.5,
    freeze_backbone: bool = True,
) -> FlowMatchingSdePolicy:
    """Pick an Adapter for a lerobot policy and assemble a FlowMatchingSdePolicy.

    Args:
        policy: a lerobot SmolVLAPolicy / PI0Policy / PI05Policy
        model_type: "smolvla" | "pi0" | "pi05"  (see _REGISTRY in adapters/__init__.py)
    """
    adapter = build_adapter(model_type, policy)
    return FlowMatchingSdePolicy(
        adapter,
        num_steps=num_steps,
        noise_level=noise_level,
        freeze_backbone=freeze_backbone,
    )


# ------------------------------------------------------------
# Backward compatibility: keep the old SmolVLASdePolicy name
# ------------------------------------------------------------
def SmolVLASdePolicy(policy, num_steps=10, noise_level=0.5, freeze_vlm=True):
    """Legacy API compatibility wrapper. New code should prefer build_sde_policy."""
    return build_sde_policy(
        policy,
        model_type="smolvla",
        num_steps=num_steps,
        noise_level=noise_level,
        freeze_backbone=freeze_vlm,
    )
