"""pruner — removes token positions from the KV cache under memory pressure.

Three strategies:
    * ``oldest_first`` — drop the earliest cached tokens.
    * ``least_used`` — drop tokens with the lowest accumulated attention score.
    * ``attention_score`` — drop tokens with the lowest attention weight
      from the most recent forward pass.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from kv_cache import KVCacheEntry

logger = logging.getLogger(__name__)


class BasePruner(ABC):
    """Abstract base for KV cache pruning strategies."""

    @abstractmethod
    def prune(
        self,
        entry: KVCacheEntry,
        target_tokens: int,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Return the *keep* indices after pruning to *target_tokens*.

        Args:
            entry: The cache entry to prune.
            target_tokens: Maximum number of tokens to retain.
            attention_scores: Optional tensor of shape ``(num_heads, seq_len)``
                with per-token attention weights for the most recent step.

        Returns:
            List of column indices to keep.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short strategy name."""


# ---------------------------------------------------------------------------
# Oldest-first
# ---------------------------------------------------------------------------


class OldestFirstPruner(BasePruner):
    """Drop the earliest cached tokens, retaining the most recent."""

    def prune(
        self,
        entry: KVCacheEntry,
        target_tokens: int,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> List[int]:
        seq_len = entry.seq_len
        if seq_len <= target_tokens:
            return list(range(seq_len))
        keep = list(range(seq_len - target_tokens, seq_len))
        return keep

    @property
    def name(self) -> str:
        return "oldest_first"


# ---------------------------------------------------------------------------
# Least-used (attention-score accumulator)
# ---------------------------------------------------------------------------


class LeastUsedPruner(BasePruner):
    """Drop tokens with the lowest cumulative attention score.

    Attention scores accumulate across steps; tokens that receive little
    attention across all steps get pruned first.
    """

    def __init__(self) -> None:
        self.accumulated_scores: Optional[torch.Tensor] = None

    def prune(
        self,
        entry: KVCacheEntry,
        target_tokens: int,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> List[int]:
        seq_len = entry.seq_len
        if seq_len <= target_tokens:
            return list(range(seq_len))

        if attention_scores is not None:
            # Accumulate: average across heads, sum across the batch dim
            head_avg = attention_scores.mean(dim=0)
            if head_avg.dim() > 1:
                head_avg = head_avg.sum(dim=0)
            if self.accumulated_scores is None or self.accumited_scores.shape[0] != seq_len:
                self.accumulated_scores = head_avg
            else:
                self.accumulated_scores = self.accumulated_scores[:seq_len] + head_avg

        if self.accumulated_scores is not None and self.accumulated_scores.shape[0] >= seq_len:
            scores = self.accumulated_scores[:seq_len].clone()
        else:
            # Fallback: token age (older = less important)
            scores = torch.arange(seq_len, dtype=torch.float, device=entry.keys.device)

        _, keep_idx = torch.topk(scores, k=target_tokens, largest=True)
        return sorted(keep_idx.tolist())

    @property
    def name(self) -> str:
        return "least_used"


# ---------------------------------------------------------------------------
# Attention-score based (single-step)
# ---------------------------------------------------------------------------


class AttentionScorePruner(BasePruner):
    """Drop tokens with the lowest attention weight from the most recent step.

    This pruner *requires* ``attention_scores``; if not provided it
    falls back to ``oldest_first``.
    """

    def prune(
        self,
        entry: KVCacheEntry,
        target_tokens: int,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> List[int]:
        seq_len = entry.seq_len
        if seq_len <= target_tokens:
            return list(range(seq_len))

        if attention_scores is None:
            logger.debug("AttentionScorePruner: no scores, falling back to oldest_first")
            return OldestFirstPruner().prune(entry, target_tokens)

        # Average across heads, sum across batch
        scores = attention_scores.mean(dim=0)
        if scores.dim() > 1:
            scores = scores.sum(dim=0)

        _, keep_idx = torch.topk(scores, k=target_tokens, largest=True)
        return sorted(keep_idx.tolist())

    @property
    def name(self) -> str:
        return "attention_score"


# ---------------------------------------------------------------------------
# Pruner registry
# ---------------------------------------------------------------------------

PRUNER_REGISTRY = {
    "oldest_first": OldestFirstPruner(),
    "least_used": LeastUsedPruner(),
    "attention_score": AttentionScorePruner(),
}


def get_pruner(name: str) -> BasePruner:
    """Return a pruner by name.  Raises ``KeyError`` on unknown name."""
    if name not in PRUNER_REGISTRY:
        raise KeyError(
            f"Unknown pruner '{name}'.  Choose from {list(PRUNER_REGISTRY.keys())}"
        )
    return PRUNER_REGISTRY[name]
