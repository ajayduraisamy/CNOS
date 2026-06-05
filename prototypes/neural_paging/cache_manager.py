"""CacheManager — LRU eviction cache for the Neural Paging Engine.

Maintains a fixed-capacity RAM cache of loaded transformer layers.
When the cache is full and a new layer needs to be admitted, the
least-recently-used layer is evicted.  All operations are O(1)
amortised via an internal ``OrderedDict``.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Tuple

from layer_store import LayerMeta

logger = logging.getLogger(__name__)


class CacheFullError(RuntimeError):
    """Raised when the cache is full and no eviction candidate exists."""


class CacheManager:
    """LRU-eviction cache that tracks hits, misses, and memory usage.

    Args:
        max_ram_mb: Maximum amount of RAM (MB) the cache may consume.
    """

    def __init__(self, max_ram_mb: float) -> None:
        if max_ram_mb <= 0:
            raise ValueError("max_ram_mb must be positive")

        self.max_ram_mb = max_ram_mb
        self._cache: OrderedDict[int, LayerMeta] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

        logger.info("CacheManager initialised (max RAM: %.0f MB)", max_ram_mb)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_usage_mb(self) -> float:
        """Total size of cached layers in MB."""
        return sum(meta.size_mb for meta in self._cache.values())

    @property
    def available_mb(self) -> float:
        """Remaining headroom before the cache is full."""
        return self.max_ram_mb - self.current_usage_mb

    @property
    def is_full(self) -> bool:
        """Whether the cache has reached its maximum capacity."""
        return self.current_usage_mb >= self.max_ram_mb

    @property
    def size(self) -> int:
        """Number of layers currently in the cache."""
        return len(self._cache)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def evictions(self) -> int:
        return self._evictions

    @property
    def hit_rate(self) -> float:
        """Fraction of total lookups that were cache hits (0.0 – 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Fraction of total lookups that were cache misses."""
        return 1.0 - self.hit_rate

    @property
    def cached_layer_ids(self) -> List[int]:
        """Ordered list of layer IDs currently in cache (LRU → MRU)."""
        return list(self._cache.keys())

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, layer_id: int) -> Optional[LayerMeta]:
        """Look up a layer in the cache.

        Returns the :class:`LayerMeta` if found (recording a hit and
        promoting the entry to MRU position), or ``None`` on a miss.
        """
        if layer_id in self._cache:
            self._cache.move_to_end(layer_id)
            self._hits += 1
            logger.debug("Cache HIT  layer %d  (size: %.1f MB)", layer_id, self._cache[layer_id].size_mb)
            return self._cache[layer_id]

        self._misses += 1
        logger.debug("Cache MISS layer %d", layer_id)
        return None

    def put(self, layer_id: int, meta: LayerMeta) -> Optional[LayerMeta]:
        """Insert a layer into the cache, evicting if necessary.

        If the layer already exists this is a no-op (but promotes to
        MRU).  If the cache is full, the LRU entry is evicted.

        Returns the :class:`LayerMeta` of the evicted layer, or ``None``.
        """
        # Already present → promote and return
        if layer_id in self._cache:
            self._cache.move_to_end(layer_id)
            return None

        # Ensure capacity
        evicted = self._make_room(meta.size_mb)

        self._cache[layer_id] = meta
        self._cache.move_to_end(layer_id)
        logger.debug(
            "Cache PUT  layer %d  (%.1f MB, cache now %.0f / %.0f MB)",
            layer_id,
            meta.size_mb,
            self.current_usage_mb,
            self.max_ram_mb,
        )
        return evicted

    def _make_room(self, needed_mb: float) -> Optional[LayerMeta]:
        """Evict LRU entries until *needed_mb* of space is available.

        If a single layer is larger than the entire cache, all existing
        entries are evicted and the layer is admitted anyway (graceful
        degradation).

        Returns the first evicted :class:`LayerMeta`, or ``None`` if no
        eviction was required.
        """
        evicted: Optional[LayerMeta] = None

        # Single layer larger than cache → evict everything to make room
        if needed_mb > self.max_ram_mb:
            while self._cache:
                lid, meta = self._cache.popitem(last=False)
                self._evictions += 1
                if evicted is None:
                    evicted = meta
                logger.debug(
                    "Evicted layer %d (%.1f MB) — layer exceeds cache capacity",
                    lid, meta.size_mb,
                )
            return evicted

        while self.max_ram_mb - self.current_usage_mb < needed_mb:
            if not self._cache:
                # Should not reach here given the guard above, but safety net
                break
            lid, meta = self._cache.popitem(last=False)  # LRU
            self._evictions += 1
            if evicted is None:
                evicted = meta
            logger.debug(
                "Evicted layer %d  (%.1f MB) — eviction #%d",
                lid, meta.size_mb, self._evictions,
            )
        return evicted

    def remove(self, layer_id: int) -> bool:
        """Explicitly remove a layer from the cache.

        Returns ``True`` if the layer was present and removed.
        """
        if layer_id in self._cache:
            del self._cache[layer_id]
            logger.debug("Removed layer %d from cache", layer_id)
            return True
        return False

    def clear(self) -> None:
        """Empty the cache entirely."""
        self._cache.clear()
        logger.debug("Cache cleared")

    def __contains__(self, layer_id: int) -> bool:
        return layer_id in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __iter__(self) -> Iterator[int]:
        return iter(self._cache)

    def __repr__(self) -> str:
        return (
            f"CacheManager(used={self.current_usage_mb:.0f}/{self.max_ram_mb:.0f} MB, "
            f"layers={self.size}, hits={self._hits}, misses={self._misses}, "
            f"evictions={self._evictions})"
        )
