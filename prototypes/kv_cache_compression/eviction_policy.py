"""eviction_policy — decides *which* layer/token to evict when the cache is full.

Policies operate at two levels:
    1. Per-layer eviction: which layer's tokens to drop first.
    2. Per-token eviction: which positions within a layer to drop.

Current implementation focuses on per-layer scoring with three strategies:
    * ``LRU`` — evict from the layer accessed least recently.
    * ``LFU`` — evict from the layer accessed least frequently.
    * ``Adaptive`` — switches between LRU and LFU based on cache pressure.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from kv_cache import KVCacheEntry, KVCacheManager
from pruner import BasePruner, get_pruner

logger = logging.getLogger(__name__)


class BaseEvictionPolicy(ABC):
    """Abstract base for eviction policies."""

    def __init__(self, pruner: Optional[BasePruner] = None) -> None:
        self.pruner = pruner or get_pruner("oldest_first")

    @abstractmethod
    def select_eviction_candidates(
        self,
        manager: KVCacheManager,
        tokens_to_free: int,
    ) -> Dict[int, List[int]]:
        """Return ``{layer_idx: [token_indices_to_drop]}``.

        Args:
            manager: The cache manager holding all layer entries.
            tokens_to_free: Approximate number of token positions to free.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short policy name."""


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------


class LRUPolicy(BaseEvictionPolicy):
    """Evict from the layer with the oldest last-access time."""

    def select_eviction_candidates(
        self,
        manager: KVCacheManager,
        tokens_to_free: int,
    ) -> Dict[int, List[int]]:
        candidates: Dict[int, List[int]] = {}

        # Score layers by last_access_time (lower = older = more evictable)
        scored = [
            (layer_idx, entry.last_access_time)
            for layer_idx, entry in enumerate(manager.entries)
            if entry.seq_len > 0
        ]
        scored.sort(key=lambda x: x[1])

        remaining = tokens_to_free
        for layer_idx, _ in scored:
            if remaining <= 0:
                break
            entry = manager.entries[layer_idx]
            drop_count = min(remaining, entry.seq_len // 2) if entry.seq_len > 1 else 0
            if drop_count > 0:
                keep_count = entry.seq_len - drop_count
                keep = self.pruner.prune(entry, keep_count)
                candidates[layer_idx] = [i for i in range(entry.seq_len) if i not in keep]
                remaining -= drop_count

        return candidates

    @property
    def name(self) -> str:
        return "lru"


# ---------------------------------------------------------------------------
# LFU
# ---------------------------------------------------------------------------


class LFUPolicy(BaseEvictionPolicy):
    """Evict from the layer with the lowest access count."""

    def select_eviction_candidates(
        self,
        manager: KVCacheManager,
        tokens_to_free: int,
    ) -> Dict[int, List[int]]:
        candidates: Dict[int, List[int]] = {}

        scored = [
            (layer_idx, entry.access_count)
            for layer_idx, entry in enumerate(manager.entries)
            if entry.seq_len > 0
        ]
        scored.sort(key=lambda x: x[1])

        remaining = tokens_to_free
        for layer_idx, _ in scored:
            if remaining <= 0:
                break
            entry = manager.entries[layer_idx]
            drop_count = min(remaining, entry.seq_len // 2) if entry.seq_len > 1 else 0
            if drop_count > 0:
                keep_count = entry.seq_len - drop_count
                keep = self.pruner.prune(entry, keep_count)
                candidates[layer_idx] = [i for i in range(entry.seq_len) if i not in keep]
                remaining -= drop_count

        return candidates

    @property
    def name(self) -> str:
        return "lfu"


# ---------------------------------------------------------------------------
# Adaptive
# ---------------------------------------------------------------------------


class AdaptivePolicy(BaseEvictionPolicy):
    """Switch between LRU and LFU based on cache pressure.

    Under low pressure (<50% full), use LFU (remove genuinely unused).
    Under high pressure (>=50% full), use LRU (safely remove oldest).
    """

    def __init__(
        self,
        pruner: Optional[BasePruner] = None,
        pressure_threshold: float = 0.5,
    ) -> None:
        super().__init__(pruner)
        self.pressure_threshold = pressure_threshold
        self._lru = LRUPolicy(pruner)
        self._lfu = LFUPolicy(pruner)

    def _cache_pressure(self, manager: KVCacheManager) -> float:
        """Fraction of max cache length currently used (per-layer average)."""
        if manager.num_layers == 0:
            return 0.0
        fractions = [
            entry.seq_len / max(manager.max_cache_len, 1)
            for entry in manager.entries
        ]
        return sum(fractions) / manager.num_layers

    def select_eviction_candidates(
        self,
        manager: KVCacheManager,
        tokens_to_free: int,
    ) -> Dict[int, List[int]]:
        pressure = self._cache_pressure(manager)
        if pressure >= self.pressure_threshold:
            return self._lru.select_eviction_candidates(manager, tokens_to_free)
        else:
            return self._lfu.select_eviction_candidates(manager, tokens_to_free)

    @property
    def name(self) -> str:
        return "adaptive"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EVICTION_REGISTRY = {
    "lru": LRUPolicy,
    "lfu": LFUPolicy,
    "adaptive": AdaptivePolicy,
}


def get_eviction_policy(
    name: str,
    pruner: Optional[BasePruner] = None,
) -> BaseEvictionPolicy:
    """Return an eviction policy by name.  Raises ``KeyError`` on unknown name."""
    if name not in EVICTION_REGISTRY:
        raise KeyError(
            f"Unknown eviction policy '{name}'.  "
            f"Choose from {list(EVICTION_REGISTRY.keys())}"
        )
    return EVICTION_REGISTRY[name](pruner=pruner)
