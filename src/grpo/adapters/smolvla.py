"""SmolVLA adapter (lerobot.policies.smolvla).

Reference: modeling_smolvla.py::VLAFlowMatching.sample_actions L830-881
    embed_prefix takes state as a keyword argument (because state is embedded into the prefix)
    denoise_step takes (prefix_pad_masks, past_key_values, x_t, timestep)
    freezing is done via SmolVLMWithExpertModel.set_requires_grad(train_expert_only=True)
"""
from __future__ import annotations

import torch

from .base import FlowMatchingAdapter


class SmolVLAAdapter(FlowMatchingAdapter):
    def prefix_features(self, batch: dict) -> dict:
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks  # lazy import
        model = self.policy.model
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                batch["images"],
                batch["img_masks"],
                batch["lang_tokens"],
                batch["lang_masks"],
                state=batch["state"],
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=model.config.use_cache,
                fill_kv_cache=True,
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
        vwe = self.policy.model.vlm_with_expert
        vwe.train_expert_only = True
        vwe.set_requires_grad()
        self.report_trainable_params()

    def action_shape(self, batch: dict) -> tuple[int, int]:
        cfg = self.policy.config
        return cfg.chunk_size, cfg.max_action_dim
