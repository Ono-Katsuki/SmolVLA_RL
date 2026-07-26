"""Flow-SDE: ODE→SDE conversion for flow-matching policies (π_RL §5.1).

SmolVLA's denoising is an Euler ODE with τ: 1→0:
    A_{k+1} = A_k + dτ · v_θ(A_k, τ_k),   dτ = -1/N

π_RL's SDE version (Eq. 8) makes it stochastic while preserving the same marginals:
    dA^τ = [ v_θ + σ_τ²/(2τ) · (A + (1-τ) v_θ) ] dτ + σ_τ dw_τ
    σ_τ  = a · √( τ / (1-τ) )        # a = noise_level (RLinf default 0.5)

Euler-Maruyama discretization (dτ < 0):
    μ_τ     = A + dτ · [ v + σ²/(2τ) · (A + (1-τ) v) ]
    Σ_τ     = σ² · |dτ| · I
    A_next ~ N(μ_τ, Σ_τ)

This makes each denoising transition Gaussian, so the log-prob has a closed
form. For the GRPO ratio, the A_next stored during rollout is held fixed and
μ_τ is recomputed with the new parameters to obtain logp.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


EPS = 1e-5  # keeps τ from collapsing to 0
# Safety floor for std. Sampling and log-prob evaluation use the identical std
# so both are exactly the same Gaussian (previously sample=std vs
# eval=√(std²+EPS) diverged, leaving a common 1/var factor in the quadratic
# term that slightly scaled the log-ratio at low-noise steps: caught by Codex).
# Over the actual τ range (min τ=0.1 → min std≈0.037) this floor never fires;
# it is purely precautionary.
STD_FLOOR = 1e-3

# σ_τ = a·√(τ/(1-τ)) diverges as τ→1. With only the EPS clamp (1-1e-5), the
# discrete noise std at the first step (τ=1.0) becomes a×100 and the rollout is
# destroyed on the very first move (measured: a policy at 20% under
# deterministic evaluation dropped to 0% on all tasks under SDE rollout).
# Capping τ at 0.9 bounds σ ≤ 3a (step std ≤ 0.95a). Any bounded σ schedule is
# valid as long as it is used together with the paired drift correction (it is
# a member of the marginal-preserving family).
TAU_SIGMA_MAX = 0.9


@dataclass
class SdeStepResult:
    x_next: torch.Tensor      # (B, chunk, action_dim)
    mean: torch.Tensor        # μ_τ, same shape as x_next
    log_std: torch.Tensor     # log σ · √|dτ| (per element)
    log_prob: torch.Tensor    # (B,) — summed over chunk + action dims


def sigma_tau(tau: torch.Tensor, noise_level: float) -> torch.Tensor:
    """σ_τ = a · √(τ/(1-τ)), with τ clamped to [EPS, TAU_SIGMA_MAX]."""
    tau_c = tau.clamp(min=EPS, max=TAU_SIGMA_MAX)
    return noise_level * torch.sqrt(tau_c / (1.0 - tau_c))


def sde_step(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    dtau: float,
    noise_level: float,
    noise: torch.Tensor | None = None,
) -> SdeStepResult:
    """Perform one SDE transition step and return it with its log-prob.

    Args:
        x:      current A_τ, shape (B, chunk, action_dim)
        v:      velocity-field prediction v_θ(A_τ, τ)
        tau:    current τ, shape (B,) or scalar
        dtau:   τ increment (negative for SmolVLA)
        noise_level: SDE strength a
        noise:  pre-drawn ε (held fixed for recomputation). New sample if None.
    """
    if tau.ndim == 0:
        tau = tau.expand(x.shape[0])
    tau_view = tau.view(-1, 1, 1)
    sigma = sigma_tau(tau_view, noise_level)  # (B,1,1)

    # drift (π_RL Eq. 8)
    tau_safe = tau_view.clamp(min=EPS)
    drift = v + (sigma ** 2) / (2.0 * tau_safe) * (x + (1.0 - tau_view) * v)
    mean = x + dtau * drift  # (B, chunk, dim)

    # Diffusion term: √|dτ|·σ·ε. std is clamped by STD_FLOOR so sampling and
    # log-prob use the identical variance (var = std² below adds no EPS —
    # exactly matching the sampler).
    std = (sigma * math.sqrt(abs(dtau))).clamp(min=STD_FLOOR)  # (B,1,1)
    if noise is None:
        noise = torch.randn_like(x)
    x_next = mean + std * noise

    # Gaussian log-prob (diagonal covariance, elementwise independent)
    #   log N(x_next; mean, std²·I) = -0.5 · Σ [ (x_next-mean)²/std² + log(2π std²) ]
    var = std ** 2
    per_elem = -0.5 * ((x_next - mean) ** 2 / var + torch.log(2.0 * math.pi * var))
    log_prob = per_elem.flatten(1).sum(dim=1)  # (B,)

    log_std = torch.log(std).expand_as(x)
    return SdeStepResult(x_next=x_next, mean=mean, log_std=log_std, log_prob=log_prob)


def recompute_step_logprob(
    x: torch.Tensor,
    x_next: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    dtau: float,
    noise_level: float,
) -> torch.Tensor:
    """Recompute log-prob with a new v while holding rollout-time (x, x_next) fixed (GRPO's π_new)."""
    if tau.ndim == 0:
        tau = tau.expand(x.shape[0])
    tau_view = tau.view(-1, 1, 1)
    sigma = sigma_tau(tau_view, noise_level)
    tau_safe = tau_view.clamp(min=EPS)
    drift = v + (sigma ** 2) / (2.0 * tau_safe) * (x + (1.0 - tau_view) * v)
    mean = x + dtau * drift
    std = (sigma * math.sqrt(abs(dtau))).clamp(min=STD_FLOOR)  # exactly matches sde_step
    var = std ** 2
    per_elem = -0.5 * ((x_next - mean) ** 2 / var + torch.log(2.0 * math.pi * var))
    return per_elem.flatten(1).sum(dim=1)
