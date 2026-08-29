"""Stable interface implemented by every supported model architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelArchitecture:
    """Cache-relevant dimensions derived from a loaded model configuration."""

    num_layers: int
    num_heads: int
    hidden_size: int
    num_kv_heads: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class ModelRunner(ABC):
    """Architecture adapter used by the scheduler and KV-cache engine."""

    model_id: str
    model: torch.nn.Module
    tokenizer: object
    architecture: ModelArchitecture

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @abstractmethod
    def forward_one_sequence(self, kv_cache, block_manager, sequence, input_ids, *, is_prefill: bool):
        """Run one prefill or decode forward pass and return last-token logits."""
