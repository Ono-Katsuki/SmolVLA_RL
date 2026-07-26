"""Adapter base class.

Common interface for flow-matching VLA models (SmolVLA / π0 / π0.5 / ...).
`FlowMatchingSdePolicy` accesses models through this interface and never
touches model-specific API differences.

Subclasses implement:
    prefix_features(inputs) -> dict         compute VLM prefix + KV-cache (no_grad)
    predict_velocity(prefix, x_t, tau)      one-step velocity prediction (with gradients)
    freeze_backbone()                       freeze the VLM (only the action expert stays trainable)
    action_shape(batch) -> (chunk, dim)     noise shape used during rollout

The `inputs` received by `prefix_features` is the return value of
`prepare_inputs()` (images / img_masks / lang_tokens / lang_masks / state).
The conversion from the post-preprocessor batch is implemented once in this
base class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class FlowMatchingAdapter(nn.Module, ABC):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    @abstractmethod
    def prefix_features(self, batch: dict) -> dict:
        """Compute and return the VLM-side prefix embeds + KV-cache. The cache is reused throughout denoising."""
        ...

    @abstractmethod
    def predict_velocity(self, prefix: dict, x_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """One-step velocity prediction by the action expert. Gradients flow through this."""
        ...

    @abstractmethod
    def freeze_backbone(self) -> None:
        """Freeze the VLM (backbone). Only the action expert remains trainable."""
        ...

    @abstractmethod
    def action_shape(self, batch: dict) -> tuple[int, int]:
        """(chunk_size, action_dim). Used for noise init during rollout."""
        ...

    # ------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------
    def prepare_inputs(self, batch: dict) -> dict:
        """Convert the post-preprocessor batch into the internal input format read by adapters.

        SmolVLA / π0 / π0.5 all provide prepare_images / prepare_state on the
        lerobot side, and the language tokens are delivered by the preprocessor
        pipeline (tokenizer step) under the observation.language.tokens /
        attention_mask keys.
        """
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        return {
            "images": images,
            "img_masks": img_masks,
            "lang_tokens": batch[OBS_LANGUAGE_TOKENS],
            "lang_masks": batch[OBS_LANGUAGE_ATTENTION_MASK],
            "state": state,
        }

    def report_trainable_params(self) -> tuple[int, int, float]:
        """Call right after freezing to print trainable / total and sanity-check.
        Returns: (trainable, total, ratio_pct)."""
        trainable = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.policy.parameters())
        pct = 100.0 * trainable / max(total, 1)
        print(f"[{type(self).__name__}] trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
        if trainable == 0:
            raise RuntimeError("All parameters frozen — nothing to train. Check freeze_backbone().")
        if trainable == total:
            raise RuntimeError("No parameters frozen — VLM will be updated. Check freeze_backbone().")
        if pct > 50.0:
            print(f"[{type(self).__name__}] ⚠️  trainable ratio {pct:.1f}% is unusually high for expert-only training")
        return trainable, total, pct
