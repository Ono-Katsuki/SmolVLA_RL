"""π0 adapter (lerobot.policies.pi0).

Reference: modeling_pi0.py::PI0FlowMatching.sample_actions L814-880
    - embed_prefix does NOT take state (not processed on the VLM side)
    - denoise_step DOES require state (proprio is mixed into the action expert)
    - the VLM is paligemma_with_expert (not vlm_with_expert)
    - quirks: _prepare_attention_masks_4d + _attn_implementation="eager"
"""
from __future__ import annotations

import torch

from .base import FlowMatchingAdapter


class Pi0Adapter(FlowMatchingAdapter):
    def prefix_features(self, batch: dict) -> dict:
        from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks  # lazy import
        model = self.policy.model
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                batch["images"],
                batch["img_masks"],
                batch["lang_tokens"],
                batch["lang_masks"],
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
            # π0 assumes eager attn (same as sample_actions)
            model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

            _, past_key_values = model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        return {
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
            "state": batch["state"],  # denoise_step requires state
        }

    def predict_velocity(self, prefix: dict, x_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.policy.model.denoise_step(
            state=prefix["state"],
            prefix_pad_masks=prefix["prefix_pad_masks"],
            past_key_values=prefix["past_key_values"],
            x_t=x_t,
            timestep=tau,
        )

    def freeze_backbone(self) -> None:
        """π0 is expected to freeze via PaliGemmaWithExpertModel.set_requires_grad(train_expert_only=True) as well.
        A fallback is provided in case the attribute names differ."""
        pwe = self.policy.model.paligemma_with_expert
        if hasattr(pwe, "set_requires_grad") and hasattr(pwe, "train_expert_only"):
            pwe.train_expert_only = True
            pwe.set_requires_grad()
        else:
            # fallback: freeze the paligemma side, leave the expert side trainable
            for p in pwe.paligemma.parameters():
                p.requires_grad = False
        self.report_trainable_params()

    def action_shape(self, batch: dict) -> tuple[int, int]:
        cfg = self.policy.config
        return cfg.chunk_size, cfg.max_action_dim
