"""virtual_memory  unified virtual memory manager for LLM execution.

Orchestrates page allocation, tier placement, eviction, and prefetching
across the memory hierarchy (GPU  RAM  Compressed KV  SSD).

Usage::

    vm = VirtualMemorySystem(ram_gb=8)
    layer_0 = vm.create_virtual_component(num_pages=64)
    vm.access(layer_0, page_index=5)
    report = vm.metrics.produce_report(...)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from memory_tiers import DEFAULT_TIERS, TierConfig, TierManager
from page_table import PageTable, PageTableEntry
from allocator import Allocator
from eviction_manager import EvictionManager
from prefetch_engine import PrefetchEngine
from metrics import VirtualMemoryMetrics, VirtualMemoryReport

logger = logging.getLogger(__name__)


class VirtualMemorySystem:
    """Unified virtual memory manager for LLM execution.

    Args:
        ram_gb: Size of CPU RAM tier in GB.
        ssd_gb: Size of SSD tier in GB.
        gpu_gb: Size of GPU VRAM tier in GB (0 = disable).
        compressed_kv_gb: Size of compressed KV tier in GB.
        page_size: Page size in bytes (default 1 MB).
        eviction_policy: Eviction policy name.
        prefetch_enabled: Enable automatic prefetching.
    """

    def __init__(
        self,
        ram_gb: float = 8.0,
        ssd_gb: float = 500.0,
        gpu_gb: float = 0.0,
        compressed_kv_gb: float = 4.0,
        page_size: int = 1024 * 1024,
        eviction_policy: str = "lru",
        prefetch_enabled: bool = True,
    ) -> None:
        self.page_size = page_size
        self.prefetch_enabled = prefetch_enabled

        # Build tier configs with user-specified capacities
        configs: Dict[int, TierConfig] = {}
        if gpu_gb > 0:
            configs[0] = DEFAULT_TIERS[0]
            configs[0].capacity = int(gpu_gb * (1024 ** 3))
        configs[1] = DEFAULT_TIERS[1]
        configs[1].capacity = int(ram_gb * (1024 ** 3))
        configs[2] = DEFAULT_TIERS[2]
        configs[2].capacity = int(compressed_kv_gb * (1024 ** 3))
        configs[3] = DEFAULT_TIERS[3]
        configs[3].capacity = int(ssd_gb * (1024 ** 3))

        self.tier_manager = TierManager(configs)
        self.page_table = PageTable()
        self.eviction = EvictionManager(
            self.page_table, self.tier_manager, eviction_policy
        )
        self.allocator = Allocator(
            self.tier_manager, self.page_table, self.eviction, page_size
        )
        self.prefetcher = PrefetchEngine(
            self.allocator, self.page_table, self.tier_manager
        )
        self.metrics = VirtualMemoryMetrics()

        self._virtual_components: Dict[int, VirtualComponent] = {}

        logger.info(
            "VirtualMemorySystem: RAM=%s GB  SSD=%s GB  page=%s  policy=%s",
            ram_gb, ssd_gb, self._fmt_bytes(page_size), eviction_policy,
        )

    # ------------------------------------------------------------------
    # Virtual components
    # ------------------------------------------------------------------

    def create_virtual_component(
        self,
        name: str = "",
        size_bytes: Optional[int] = None,
        num_pages: Optional[int] = None,
        preferred_tier: int = 1,
    ) -> VirtualComponent:
        """Register a new virtual memory component (e.g. a model layer).

        Args:
            name: Human-readable name.
            size_bytes: Total size in bytes (alternative to *num_pages*).
            num_pages: Number of pages (alternative to *size_bytes*).
            preferred_tier: Initial placement tier.

        Returns:
            A :class:`VirtualComponent` with its virtual_id.
        """
        if size_bytes is not None:
            num_p = (size_bytes + self.page_size - 1) // self.page_size
        elif num_pages is not None:
            num_p = num_pages
        else:
            num_p = 1

        vid = self.allocator.allocate_virtual_id()
        entries = self.allocator.allocate_pages(vid, num_p, preferred_tier)
        comp = VirtualComponent(
            virtual_id=vid,
            name=name or f"component_{vid}",
            num_pages=num_p,
            page_size=self.page_size,
            entries=entries,
        )
        self._virtual_components[vid] = comp
        return comp

    def get_component(self, virtual_id: int) -> Optional[VirtualComponent]:
        return self._virtual_components.get(virtual_id)

    # ------------------------------------------------------------------
    # Memory access
    # ------------------------------------------------------------------

    def access(
        self,
        virtual_id: int,
        page_index: int,
        is_write: bool = False,
    ) -> float:
        """Simulate a memory access.  Returns latency in ns.

        Steps:
            1. Look up the page in the page table.
            2. If in tier 0-2 (GPU/RAM/CompKV) → hit (fast path).
            3. If on tier 3 (SSD) → page fault, promote to RAM.
            4. Record hit/fault, optionally trigger prefetch.
        """
        entry = self.page_table.get(virtual_id, page_index)

        if entry is not None and entry.tier <= 2:
            # Hit: page is already in fast memory
            self.page_table.mark_accessed(virtual_id, page_index)
            if is_write:
                entry.dirty = True
            self.metrics.record_hit()
            self.eviction.record_hit()
            tier = self.tier_manager[entry.tier]
            latency = float(tier.access_cost_ns)
            self.metrics.record_latency(int(latency))
            if self.prefetch_enabled:
                self.prefetcher.predict_and_prefetch(virtual_id, page_index)
            return latency

        # Fault: page is on slow tier or not present
        return self._handle_fault(virtual_id, page_index, entry, is_write)

    def _handle_fault(
        self,
        virtual_id: int,
        page_index: int,
        entry: Optional[PageTableEntry],
        is_write: bool,
    ) -> float:
        """Handle a page fault: move page from SSD to RAM."""
        self.metrics.record_fault()
        self.eviction.record_fault()

        # Free space in RAM if needed
        if not self.tier_manager[1].can_allocate(self.page_size):
            self.eviction.free_space(1, self.page_size)
            self.metrics.record_eviction()

        source_latency: float
        if entry is not None and entry.tier == 3:
            # Page on SSD  move it to RAM
            src_tier = self.tier_manager[entry.tier]
            source_latency = float(src_tier.access_cost_ns)
            self.allocator.move_page(virtual_id, page_index, 1)

        else:
            # Cold fault: allocate fresh page in RAM
            source_latency = float(self.tier_manager[3].access_cost_ns)
            entries = self.allocator.allocate_pages(
                virtual_id, 1, preferred_tier=1,
            )
            if entries:
                entries[0].access_count = 1
                entries[0].last_access_time = 1
                entries[0].dirty = is_write

        self.page_table.mark_accessed(virtual_id, page_index)

        if self.prefetch_enabled:
            self.prefetcher.predict_and_prefetch(virtual_id, page_index)

        ram = self.tier_manager[1]
        latency = source_latency + float(ram.access_cost_ns)
        self.metrics.record_latency(int(latency))
        return latency

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, object]:
        pt = self.page_table.summary()
        tm = self.tier_manager.summary()
        pf = self.prefetcher.summary()
        return {
            **pt,
            **tm,
            **pf,
            "page_faults": self.metrics.page_faults,
            "page_hits": self.metrics.page_hits,
            "hit_rate_pct": round(self.metrics.hit_rate * 100, 1),
            "evictions": self.metrics.evictions,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n{'=' * 60}")
        print(f"  Virtual Memory System Summary")
        print(f"{'=' * 60}")
        for k, v in s.items():
            print(f"  {k:<25} {v}")
        print(f"{'=' * 60}")

    def reset(self) -> None:
        self.page_table.clear()
        self.tier_manager.reset()
        self.allocator.reset()
        self.metrics.reset()
        self.prefetcher.reset()
        self._virtual_components.clear()
        self.eviction = EvictionManager(
            self.page_table, self.tier_manager,
            self.eviction.policy.name,
        )
        self.allocator.evict = self.eviction
        logger.info("VirtualMemorySystem reset")

    @staticmethod
    def _fmt_bytes(b: int) -> str:
        if b >= 1024 ** 3:
            return f"{b / (1024**3):.1f}GB"
        if b >= 1024 ** 2:
            return f"{b / (1024**2):.1f}MB"
        return f"{b}B"


class VirtualComponent:
    """A contiguous virtual memory region (e.g. one model layer).

    Attributes:
        virtual_id: Unique identifier.
        name: Human-readable name.
        num_pages: Number of pages in this component.
        page_size: Size of each page in bytes.
        entries: List of :class:`PageTableEntry` objects.
    """

    def __init__(
        self,
        virtual_id: int,
        name: str,
        num_pages: int,
        page_size: int,
        entries: List[PageTableEntry],
    ) -> None:
        self.virtual_id = virtual_id
        self.name = name
        self.num_pages = num_pages
        self.page_size = page_size
        self.entries = entries

    @property
    def size_bytes(self) -> int:
        return self.num_pages * self.page_size

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)
