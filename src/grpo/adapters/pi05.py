"""π0.5 adapter (lerobot.policies.pi05).

Reference: modeling_pi05.py::PI05FlowMatching.sample_actions L792-860
    - embed_prefix takes (images, img_masks, tokens, masks) — state is mixed into the discretized tokens
    - denoise_step takes (prefix_pad_masks, past_key_values, x_t, timestep) — no state needed (already in prefix)
    - same as π0: paligemma_with_expert + eager attn
"""
from __future__ import annotations

import torch

from .base import FlowMatchingAdapter


class Pi05Adapter(FlowMatchingAdapter):
    def prefix_features(self, batch: dict) -> dict:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        model = self.policy.model
        with torch.no_grad():
            # π0.5 takes tokens/masks (state already included)
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                batch["images"],
                batch["img_masks"],
                batch.get("tokens", batch.get("lang_tokens")),
                batch.get("masks", batch.get("lang_masks")),
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
            model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

            _, past_key_values = model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        return {"prefix_pad_masks": prefix_pad_masks, "past_key_values": past_key_values}

    def predict_velocity(self, prefix: dict, x_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.policy.model.denoise_step(
            prefix_pad_masks=prefix["prefix_pad_masks"],
            past_key_values=prefix["past_key_values"],
            x_t=x_t,
            timestep=tau,
        )

    def freeze_backbone(self) -> None:
        pwe = self.policy.model.paligemma_with_expert
        if hasattr(pwe, "set_requires_grad") and hasattr(pwe, "train_expert_only"):
            pwe.train_expert_only = True
            pwe.set_requires_grad()
        else:
            for p in pwe.paligemma.parameters():
                p.requires_grad = False
        self.report_trainable_params()

    def action_shape(self, batch: dict) -> tuple[int, int]:
        cfg = self.policy.config
        return cfg.chunk_size, cfg.max_action_dim
