"""memory_controller  bridges VirtualMemorySystem + NeuralPager.

Manages layer residency in RAM by combining:
  * **Neural Paging**  per-layer LRU cache with prefetching.
  * **Memory Virtualization**  page-level address translation across
    GPU / RAM / Compressed KV / SSD tiers.

Provides a unified ``prepare_layer(layer_id)``  ensure a layer is
loaded in fast memory before inference.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

_PROTO = os.path.join(os.path.dirname(__file__), "..")
for _dir in ("neural_paging", "memory_virtualization"):
    _p = os.path.join(_PROTO, _dir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    ram_used_gb: float = 0.0
    ram_capacity_gb: float = 0.0
    ssd_used_gb: float = 0.0
    page_faults: int = 0
    page_hits: int = 0
    hit_rate_pct: float = 0.0
    evictions: int = 0
    prefetches: int = 0
    prefetch_accuracy_pct: float = 0.0
    cache_usage_mb: float = 0.0
    cache_capacity_mb: float = 0.0
    loaded_layers: int = 0
    total_layers: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "ram_used_gb": round(self.ram_used_gb, 2),
            "ram_util_pct": round(
                self.ram_used_gb / self.ram_capacity_gb * 100, 1
            ) if self.ram_capacity_gb > 0 else 0.0,
            "page_faults": self.page_faults,
            "page_hits": self.page_hits,
            "hit_rate_pct": round(self.hit_rate_pct, 1),
            "evictions": self.evictions,
            "prefetches": self.prefetches,
            "prefetch_accuracy_pct": round(self.prefetch_accuracy_pct, 1),
            "cache_usage_mb": round(self.cache_usage_mb, 1),
            "loaded_layers": self.loaded_layers,
        }


class MemoryController:
    """Combines NeuralPager (layer-level) with VirtualMemorySystem (page-level).

    Args:
        ram_gb: CPU RAM capacity in GB.
        ssd_gb: SSD capacity in GB.
        page_size: Virtual memory page size in bytes.
        num_layers: Total model layers.
        eviction_policy: One of ``"lru"``, ``"lfu"``, ``"adaptive"``.
        prefetch_enabled: Allow automatic prefetching.
    """

    def __init__(
        self,
        ram_gb: float = 4.0,
        ssd_gb: float = 100.0,
        page_size: int = 1024 * 1024,
        num_layers: int = 22,
        eviction_policy: str = "lru",
        prefetch_enabled: bool = True,
    ) -> None:
        self.ram_gb = ram_gb
        self.num_layers = num_layers

        from layer_store import LayerStore
        from cache_manager import CacheManager
        from prefetcher import Prefetcher, PrefetchStrategy
        from pager import NeuralPager
        from virtual_memory import VirtualMemorySystem

        self.layer_store = LayerStore(num_layers=num_layers, seed=42)
        self.cache_manager = CacheManager(max_ram_mb=ram_gb * 1024)
        self.prefetcher = Prefetcher(
            strategy=PrefetchStrategy.SEQUENTIAL,
            num_layers=num_layers,
        )
        self.pager = NeuralPager(
            layer_store=self.layer_store,
            cache_manager=self.cache_manager,
            prefetcher=self.prefetcher,
        )

        pages_per_layer = max(4, int(ram_gb * 256 / num_layers))
        self.vm = VirtualMemorySystem(
            ram_gb=ram_gb,
            ssd_gb=ssd_gb,
            page_size=page_size,
            eviction_policy=eviction_policy,
            prefetch_enabled=prefetch_enabled,
        )

        self._layer_vm_pages: Dict[int, int] = {}
        for lid in range(num_layers):
            comp = self.vm.create_virtual_component(
                name=f"layer_{lid}",
                num_pages=pages_per_layer,
                preferred_tier=3,
            )
            self._layer_vm_pages[lid] = comp.virtual_id

        self._prefetched_pages: Set[int] = set()

        logger.info(
            "MemoryController: %.1fGB RAM, %d layers, policy=%s, prefetch=%s",
            ram_gb, num_layers, eviction_policy, prefetch_enabled,
        )

    def prepare_layer(self, layer_id: int) -> float:
        if not (0 <= layer_id < self.num_layers):
            raise ValueError(f"layer_id {layer_id} out of range [0, {self.num_layers})")

        paging_latency = 0.0
        try:
            self.pager.access_layer(layer_id)
        except KeyError:
            logger.warning("Layer %d not found in store, skipping paging", layer_id)

        vm_id = self._layer_vm_pages.get(layer_id)
        if vm_id is not None:
            self.vm.access(vm_id, 0)

        return paging_latency

    def prepare_layers(self, layer_ids: Set[int]) -> float:
        total_latency = 0.0
        for lid in sorted(layer_ids):
            total_latency += self.prepare_layer(lid)
        return total_latency

    def evict_layer(self, layer_id: int) -> bool:
        self.layer_store.unload_layer(layer_id)
        self.cache_manager.remove(layer_id)
        logger.debug("Evicted layer %d", layer_id)
        return True

    def get_metrics(self) -> MemoryMetrics:
        pager_metrics = self.pager.metrics
        vm_summary = self.vm.summary()
        ram_tier = self.vm.tier_manager[1]
        ssd_tier = self.vm.tier_manager[3]

        return MemoryMetrics(
            ram_used_gb=ram_tier.used / (1024 ** 3),
            ram_capacity_gb=self.ram_gb,
            ssd_used_gb=ssd_tier.used / (1024 ** 3),
            page_faults=vm_summary.get("page_faults", 0),
            page_hits=vm_summary.get("page_hits", 0),
            hit_rate_pct=vm_summary.get("hit_rate_pct", 0.0),
            evictions=vm_summary.get("evictions", 0),
            prefetches=self.prefetcher.total_prefetches if hasattr(self.prefetcher, 'total_prefetches') else 0,
            prefetch_accuracy_pct=(
                self.prefetcher.get_accuracy() * 100
                if hasattr(self.prefetcher, 'get_accuracy')
                else 0.0
            ),
            cache_usage_mb=self.cache_manager.current_usage_mb,
            cache_capacity_mb=self.cache_manager.max_ram_mb,
            loaded_layers=len(self.cache_manager),
            total_layers=self.num_layers,
        )

    def reset(self) -> None:
        self.cache_manager.clear()
        self.vm.reset()
        logger.info("MemoryController reset")
