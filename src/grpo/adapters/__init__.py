"""Flow-matching VLA adapter registry.

Per-model API differences (embed_prefix signature, whether denoise_step needs
state, how to freeze, etc.) are encapsulated in an Adapter, which
`FlowMatchingSdePolicy` calls through a common interface.

Supported:
    - smolvla  (lerobot.policies.smolvla)
    - pi0      (lerobot.policies.pi0)
    - pi05     (lerobot.policies.pi05)

Add new models via register_adapter().
"""
from __future__ import annotations

from .base import FlowMatchingAdapter
from .pi0 import Pi0Adapter
from .pi05 import Pi05Adapter
from .smolvla import SmolVLAAdapter


_REGISTRY: dict[str, type[FlowMatchingAdapter]] = {
    "smolvla": SmolVLAAdapter,
    "pi0": Pi0Adapter,
    "pi05": Pi05Adapter,
}


def register_adapter(name: str, cls: type[FlowMatchingAdapter]) -> None:
    _REGISTRY[name] = cls


def build_adapter(policy_type: str, policy) -> FlowMatchingAdapter:
    if policy_type not in _REGISTRY:
        raise KeyError(f"unknown policy_type={policy_type}. registered: {list(_REGISTRY)}")
    return _REGISTRY[policy_type](policy)


__all__ = [
    "FlowMatchingAdapter",
    "SmolVLAAdapter",
    "Pi0Adapter",
    "Pi05Adapter",
    "build_adapter",
    "register_adapter",
]
