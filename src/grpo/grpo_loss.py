"""GRPO loss (Group Relative Policy Optimization) with PPO-style clipping.

An importance ratio is computed per denoising step k and averaged over the K
steps (same form as Flow-GRPO Eq.5 / DanceGRPO Table6 / π_RL Eq.5):

    r_k    = exp( Σ_elem logπ_θ(x_{k+1}|x_k) - Σ_elem logπ_old(x_{k+1}|x_k) )
    L^clip = -E_{k,b}[ min( r_k · A, clip(r_k, 1-ε, 1+ε) · A ) ]

[Key design point] Do NOT divide the log-ratio by the number of elements
(chunk×action_dim). When we did so previously, the policy gradient collapsed
by 1/n_elem (≈1/1600) exactly as GSPO's gradient identity predicts, driving
the effective lr to ~6e-9 and stalling learning completely (run5 measurement:
kl≈1e-8, success merely oscillated around the SFT baseline). All reference
implementations (π_RL 2510.25889, Flow-GRPO 2505.05470, DanceGRPO 2505.07818)
SUM element logprobs within a step; the only averaging is over the number of
denoising steps K.

Numerical stability: GRPO takes exactly one optimizer step per iter
(full-batch). In that single gradient computation the current parameters ==
the rollout parameters, so logp_new == logp_old, i.e. log-ratio = 0 and
ratio = 1 (no exp overflow can occur). As a precaution the log-ratio is still
clamped to ±LOG_RATIO_CLAMP (the clip term bounds it anyway).

The KL is a loose anchor to the SFT reference. Using the element-mean k3
estimator keeps it O(1) so it cannot diverge no matter how far the policy
drifts (the regularizer strength is set by kl_coef). The policy term
(full-scale) and the KL term (normalized) play different roles, so treating
their normalization differently is fine — DanceGRPO omits the KL entirely.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


LOG_RATIO_CLAMP = 10.0  # exp-overflow safety valve (normally inert: ratio≈1 with 1 step/iter)


@dataclass
class GRPOLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    clip_fraction: torch.Tensor
    mean_ratio: torch.Tensor
    mean_advantage: torch.Tensor
    drift: torch.Tensor           # step-mean of |logπ_ref - logπ_new| = displacement from SFT


def grpo_loss(
    logp_new: torch.Tensor,         # (N, B) - per-step logprob under current params (element sum)
    logp_old: torch.Tensor,         # (N, B) - rollout-time logprob (fixed)
    logp_ref: torch.Tensor | None,  # (N, B) - SFT reference logprob (for KL)
    advantages: torch.Tensor,       # (B,)   - group-normalized advantage
    n_elem: int,                    # chunk_size × action_dim (used for KL normalization and diagnostics)
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
) -> GRPOLossOutput:
    """PPO-style GRPO. Ratio is per denoising step (no element-count normalization)."""
    adv = advantages.view(1, -1)  # (1, B), broadcast to (N, B)

    # --- Policy term: no division by element count (full-scale gradient) ---
    log_ratio = (logp_new - logp_old).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    ratio = torch.exp(log_ratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()

    # --- KL term: k3 estimator (Schulman) of KL(π_new‖π_ref) ---
    # diff is the per-step "joint" log-ratio (kept as an element sum). We used
    # to divide by n_elem, but that collapsed the gradient by ~1/n_elem² and
    # made the anchor a no-op (caught by Codex). Normalization belongs after
    # estimation, so here we take k3 without dividing.
    # Note: the ~1600-dim joint KL has a large magnitude, so an effective
    # anchor needs a correspondingly small kl_coef (uncalibrated). The plan:
    # first run uses kl_coef=0 — explicitly no anchor (DanceGRPO style) — then
    # measure necessity via the drift metric + held-out evaluation before
    # adding a calibrated analytic Gaussian KL. The clamp is an exp-overflow
    # safety valve.
    if logp_ref is not None and kl_coef > 0.0:
        diff = logp_ref - logp_new
        # Only the exp argument is upper-clamped (overflow safety valve). The
        # linear -diff term is NOT clamped, so the KL gradient survives even
        # when drift exceeds the threshold (clamping diff as a whole would
        # zero the derivative past the threshold, i.e. cut the anchor: caught
        # by Codex).
        kl_est = (torch.exp(diff.clamp(max=LOG_RATIO_CLAMP)) - diff - 1.0).mean()
        kl_loss = kl_coef * kl_est
    else:
        kl_loss = torch.zeros((), device=logp_new.device)

    total = policy_loss + kl_loss

    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
        mean_ratio = ratio.mean()
        mean_adv = advantages.mean()
        # Direct indicator that learning is happening: per-step logprob
        # displacement from SFT. Whether kl stays ≈0 or this grows across
        # iters tells us whether anything is moving.
        if logp_ref is not None:
            drift = (logp_ref - logp_new).abs().mean()
        else:
            drift = torch.zeros((), device=logp_new.device)

    return GRPOLossOutput(
        loss=total,
        policy_loss=policy_loss.detach(),
        kl_loss=kl_loss.detach() if isinstance(kl_loss, torch.Tensor) else torch.tensor(0.0),
        clip_fraction=clip_frac,
        mean_ratio=mean_ratio,
        mean_advantage=mean_adv,
        drift=drift,
    )
