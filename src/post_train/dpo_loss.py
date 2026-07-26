"""flow-DPO loss (Diffusion-DPO, Wallace+2023's ELBO surrogate applied to flow matching).

For flow-matching policies with no exact log-likelihood, the flow-matching
loss (velocity MSE) serves as a proxy for -log π:

    logits = -β · [ (L_w^θ − L_w^ref) − (L_l^θ − L_l^ref) ]
    L_DPO  = -E[ log σ(logits) ]

Push the winning trajectory's loss down relative to the reference and the
losing trajectory's up. θ and ref must be given the identical noise / time so
the variance cancels in the difference (caller's responsibility).

To keep unlearning of losing trajectories from destroying shared behavior, a
winning-side SFT loss anchor term (RPO style) can be mixed in via sft_coef.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DPOLossOutput:
    loss: torch.Tensor
    dpo_loss: torch.Tensor        # detached
    sft_anchor: torch.Tensor      # detached
    implicit_acc: torch.Tensor    # fraction with logits > 0 (rate at which the winner improves vs. ref)
    mean_logits: torch.Tensor
    mean_margin: torch.Tensor     # (L_l^θ − L_l^ref) − (L_w^θ − L_w^ref)


def flow_dpo_loss(
    loss_w_theta: torch.Tensor,   # (B,) per-sample flow loss of winning chunks (θ, with gradients)
    loss_l_theta: torch.Tensor,   # (B,) losing chunks (θ, with gradients)
    loss_w_ref: torch.Tensor,     # (B,) winning chunks (reference, no_grad)
    loss_l_ref: torch.Tensor,     # (B,) losing chunks (reference, no_grad)
    beta: float = 100.0,
    sft_coef: float = 0.1,
) -> DPOLossOutput:
    margin = (loss_l_theta - loss_l_ref) - (loss_w_theta - loss_w_ref)
    logits = beta * margin
    dpo = -F.logsigmoid(logits).mean()
    anchor = loss_w_theta.mean()
    total = dpo + sft_coef * anchor
    with torch.no_grad():
        acc = (logits > 0).float().mean()
    return DPOLossOutput(
        loss=total,
        dpo_loss=dpo.detach(),
        sft_anchor=anchor.detach(),
        implicit_acc=acc,
        mean_logits=logits.detach().mean(),
        mean_margin=margin.detach().mean(),
    )
