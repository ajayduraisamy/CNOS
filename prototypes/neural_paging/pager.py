"""NeuralPager — the central orchestrator of layer-level paged inference.

The NeuralPager sits between the inference engine and the layer store /
cache.  When a layer is requested it:

1. Checks the :class:`CacheManager` (fast path — cache hit).
2. On miss, it instructs the :class:`LayerStore` to load the layer,
   then inserts it into the cache (possibly evicting older layers).
3. Notifies the :class:`Prefetcher` so it can learn access patterns.
4. Instructs the prefetcher to predict and preload the next likely
   layer(s).

All paging events are logged and metrics are exposed for monitoring.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from cache_manager import CacheManager, CacheFullError
from layer_store import LayerMeta, LayerStore
from prefetcher import Prefetcher, PrefetchStrategy

logger = logging.getLogger(__name__)


@dataclass
class PagingEvent:
    """A single observable event from the paging lifecycle."""

    class Type:
        HIT = "hit"
        MISS = "miss"
        LOAD = "load"
        EVICT = "evict"
        PREFETCH = "prefetch"
        ERROR = "error"

    event_type: str
    layer_id: int
    timestamp: float
    ram_usage_mb: float
    details: str = ""


@dataclass
class PagerMetrics:
    """Aggregate runtime metrics exposed by the NeuralPager."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    layer_loads: int = 0
    evictions: int = 0
    prefetches: int = 0
    errors: int = 0
    current_ram_mb: float = 0.0
    max_ram_mb: float = 0.0
    total_load_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate


class NeuralPager:
    """Orchestrates paged access to transformer layers.

    Args:
        layer_store: The backing store (disk) for all layers.
        cache_manager: The RAM cache with LRU eviction.
        prefetcher: Strategy-based predictor for preloading.
        event_callback: Optional hook called on every paging event
            (useful for live dashboards or logging extensions).
    """

    def __init__(
        self,
        layer_store: LayerStore,
        cache_manager: CacheManager,
        prefetcher: Prefetcher,
        event_callback: Optional[Callable[[PagingEvent], None]] = None,
    ) -> None:
        self.store = layer_store
        self.cache = cache_manager
        self.prefetcher = prefetcher
        self._event_callback = event_callback
        self._metrics = PagerMetrics(max_ram_mb=cache_manager.max_ram_mb)
        self._event_log: List[PagingEvent] = []

        logger.info(
            "NeuralPager ready  —  %d layers, %.0f MB cache",
            layer_store.num_layers,
            cache_manager.max_ram_mb,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> PagerMetrics:
        """Snapshot of current paging metrics."""
        self._metrics.current_ram_mb = self.cache.current_usage_mb
        self._metrics.cache_hits = self.cache.hits
        self._metrics.cache_misses = self.cache.misses
        self._metrics.evictions = self.cache.evictions
        return self._metrics

    @property
    def event_log(self) -> List[PagingEvent]:
        """Immutable view of all recorded paging events."""
        return list(self._event_log)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def access_layer(self, layer_id: int) -> LayerMeta:
        """Request a layer for inference, handling paging transparently.

        Returns the :class:`LayerMeta` of the requested layer, which is
        guaranteed to be in cache (and therefore RAM) after this call.

        Raises:
            KeyError: If *layer_id* is out of range.
            RuntimeError: If a critical paging error occurs.
        """
        if layer_id < 0 or layer_id >= self.store.num_layers:
            raise KeyError(f"Layer {layer_id} out of range [0, {self.store.num_layers})")

        self._metrics.total_requests += 1

        # ── Fast path: cache hit ──────────────────────────────────
        cached = self.cache.get(layer_id)
        if cached is not None:
            self._emit(PagingEvent.Type.HIT, layer_id, f"cache hit (size={cached.size_mb:.1f} MB)")
            self.prefetcher.observe(layer_id)
            self._prefetch_after_access(layer_id)
            return cached

        # ── Slow path: cache miss → load from store ───────────────
        self._emit(PagingEvent.Type.MISS, layer_id, "cache miss")
        self.prefetcher.observe(layer_id)

        start = time.perf_counter()
        try:
            meta = self.store.load_layer(layer_id)
        except Exception as exc:
            self._metrics.errors += 1
            self._emit(PagingEvent.Type.ERROR, layer_id, f"load failed: {exc}")
            raise RuntimeError(f"Failed to load layer {layer_id}") from exc

        load_time_ms = (time.perf_counter() - start) * 1000
        self._metrics.layer_loads += 1
        self._metrics.total_load_time_ms += load_time_ms
        self._emit(PagingEvent.Type.LOAD, layer_id, f"loaded in {load_time_ms:.1f} ms")

        # Insert into cache, evicting if necessary
        try:
            evicted = self.cache.put(layer_id, meta)
        except CacheFullError as exc:
            self._metrics.errors += 1
            self._emit(PagingEvent.Type.ERROR, layer_id, str(exc))
            raise RuntimeError(f"Cache too small for layer {layer_id}") from exc

        if evicted is not None:
            self._emit(PagingEvent.Type.EVICT, evicted.layer_id, size_mb=evicted.size_mb)
            self.store.unload_layer(evicted.layer_id)
            logger.debug("Evicted & unloaded layer %d  (%.1f MB)", evicted.layer_id, evicted.size_mb)

        self._prefetch_after_access(layer_id)
        return meta

    # ------------------------------------------------------------------
    # Prefetch hook
    # ------------------------------------------------------------------

    def _prefetch_after_access(self, layer_id: int) -> None:
        """Predict next layers and preload them."""
        predicted = self.prefetcher.predict_next(layer_id)

        attempts, successes = self.prefetcher.prefetch(
            predicted,
            load_fn=lambda lid: self._prefetch_load(lid),
            cache_contains_fn=lambda lid: lid in self.cache,
        )

        if attempts > 0:
            self._metrics.prefetches += successes
            self._emit(
                PagingEvent.Type.PREFETCH,
                layer_id,
                f"prefetched {successes}/{attempts} layers: {predicted}",
            )

    def _prefetch_load(self, layer_id: int) -> LayerMeta:
        """Load a layer into the cache for prefetching.

        This is intentionally separate from ``access_layer`` to avoid
        recursive prefetch triggering and double-counting.
        """
        meta = self.store.load_layer(layer_id)
        evicted = self.cache.put(layer_id, meta)
        if evicted is not None:
            self._emit(PagingEvent.Type.EVICT, evicted.layer_id, size_mb=evicted.size_mb)
            self.store.unload_layer(evicted.layer_id)
        return meta

    # ------------------------------------------------------------------
    # Event management
    # ------------------------------------------------------------------

    def _emit(
        self,
        event_type: str,
        layer_id: int,
        details: str = "",
        size_mb: Optional[float] = None,
    ) -> None:
        """Record and dispatch a paging event."""
        event = PagingEvent(
            event_type=event_type,
            layer_id=layer_id,
            timestamp=time.time(),
            ram_usage_mb=self.cache.current_usage_mb,
            details=details,
        )
        self._event_log.append(event)
        if self._event_callback:
            self._event_callback(event)

    def print_summary(self) -> None:
        """Print a human-readable summary of paging metrics to stdout."""
        m = self.metrics
        print("\n" + "=" * 60)
        print("  NeuralPager — Performance Summary")
        print("=" * 60)
        print(f"  Total requests:           {m.total_requests}")
        print(f"  Cache hits:               {m.cache_hits}")
        print(f"  Cache misses:             {m.cache_misses}")
        print(f"  Hit rate:                 {m.hit_rate:.2%}")
        print(f"  Miss rate:                {m.miss_rate:.2%}")
        print(f"  Layer loads (disk I/O):   {m.layer_loads}")
        print(f"  Evictions:                {m.evictions}")
        print(f"  Prefetches:               {m.prefetches}")
        print(f"  Errors:                   {m.errors}")
        print(f"  Current RAM usage:        {m.current_ram_mb:.1f} MB")
        print(f"  Max cache size:           {m.max_ram_mb:.1f} MB")
        print(f"  Total load time (sim):    {m.total_load_time_ms:.1f} ms")
        print(f"  Avg load time:            {m.total_load_time_ms / max(m.layer_loads, 1):.1f} ms")
        print("=" * 60)
