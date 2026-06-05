"""eviction_manager  selects pages for eviction across memory tiers.

Policies:
    * LRU  evict pages with the oldest last_access_time.
    * LFU  evict pages with the lowest access_count.
    * Adaptive  switch between LRU/LFU based on page fault rate.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from memory_tiers import TierManager
from page_table import PageTable, PageTableEntry

logger = logging.getLogger(__name__)


class BaseEvictionPolicy(ABC):
    """Abstract eviction policy."""

    def __init__(self, page_table: PageTable, tier_manager: TierManager) -> None:
        self.pt = page_table
        self.tiers = tier_manager

    @abstractmethod
    def select_victims(
        self,
        tier: int,
        bytes_needed: int,
    ) -> List[Tuple[int, int]]:
        """Return ``[(virtual_id, page_index)]`` list to evict."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------


class LRUPolicy(BaseEvictionPolicy):
    """Evict pages with the oldest last_access_time."""

    def select_victims(
        self,
        tier: int,
        bytes_needed: int,
    ) -> List[Tuple[int, int]]:
        candidates = list(self.pt.entries_in_tier(tier))
        candidates.sort(key=lambda e: e.last_access_time)

        victims: List[Tuple[int, int]] = []
        freed = 0
        for entry in candidates:
            if entry.locked:
                continue
            freed += entry.size
            victims.append((entry.virtual_id, entry.page_index))
            if freed >= bytes_needed:
                break
        return victims

    @property
    def name(self) -> str:
        return "lru"


# ---------------------------------------------------------------------------
# LFU
# ---------------------------------------------------------------------------


class LFUPolicy(BaseEvictionPolicy):
    """Evict pages with the lowest access_count."""

    def select_victims(
        self,
        tier: int,
        bytes_needed: int,
    ) -> List[Tuple[int, int]]:
        candidates = list(self.pt.entries_in_tier(tier))
        candidates.sort(key=lambda e: e.access_count)

        victims: List[Tuple[int, int]] = []
        freed = 0
        for entry in candidates:
            if entry.locked:
                continue
            freed += entry.size
            victims.append((entry.virtual_id, entry.page_index))
            if freed >= bytes_needed:
                break
        return victims

    @property
    def name(self) -> str:
        return "lfu"


# ---------------------------------------------------------------------------
# Adaptive
# ---------------------------------------------------------------------------


class AdaptivePolicy(BaseEvictionPolicy):
    """Switch between LRU and LFU based on page fault rate.

    Low fault rate  LFU (target genuinely unused pages).
    High fault rate  LRU (quickly evict oldest to make room).
    """

    def __init__(
        self,
        page_table: PageTable,
        tier_manager: TierManager,
        threshold: float = 0.05,
        window: int = 1000,
    ) -> None:
        super().__init__(page_table, tier_manager)
        self.threshold = threshold
        self.window = window
        self._faults = 0
        self._accesses = 0
        self._lru = LRUPolicy(page_table, tier_manager)
        self._lfu = LFUPolicy(page_table, tier_manager)

    def _fault_rate(self) -> float:
        if self._accesses == 0:
            return 0.0
        return self._faults / self._accesses

    def record_access(self, is_fault: bool) -> None:
        self._accesses += 1
        if is_fault:
            self._faults += 1
        if self._accesses > self.window:
            self._faults = 0
            self._accesses = 0

    def select_victims(
        self,
        tier: int,
        bytes_needed: int,
    ) -> List[Tuple[int, int]]:
        if self._fault_rate() >= self.threshold:
            return self._lru.select_victims(tier, bytes_needed)
        else:
            return self._lfu.select_victims(tier, bytes_needed)

    @property
    def name(self) -> str:
        return "adaptive"


# ---------------------------------------------------------------------------
# Eviction manager (orchestrator)
# ---------------------------------------------------------------------------


EVICTION_POLICIES = {
    "lru": LRUPolicy,
    "lfu": LFUPolicy,
    "adaptive": AdaptivePolicy,
}


class EvictionManager:
    """Coordinates eviction across tiers using a configurable policy.

    Args:
        page_table: System page table.
        tier_manager: Memory tier hierarchy.
        policy_name: One of ``"lru"``, ``"lfu"``, ``"adaptive"``.
    """

    def __init__(
        self,
        page_table: PageTable,
        tier_manager: TierManager,
        policy_name: str = "lru",
    ) -> None:
        self.pt = page_table
        self.tiers = tier_manager
        if policy_name not in EVICTION_POLICIES:
            raise ValueError(
                f"Unknown policy '{policy_name}'. "
                f"Choose from {list(EVICTION_POLICIES.keys())}"
            )
        self.policy: BaseEvictionPolicy = EVICTION_POLICIES[policy_name](
            page_table, tier_manager
        )
        logger.info("EvictionManager: policy=%s", self.policy.name)

    def free_space(self, tier: int, bytes_needed: int) -> int:
        """Evict pages from *tier* until *bytes_needed* are free.

        Evicted pages are either moved to SSD (if coming from RAM/KV)
        or dropped entirely (if coming from SSD).

        Returns the number of bytes freed.
        """
        victims = self.policy.select_victims(tier, bytes_needed)
        total_freed = 0

        for virtual_id, page_index in victims:
            entry = self.pt.get(virtual_id, page_index)
            if entry is None:
                continue

            if tier < 3:
                # Move to SSD (next tier down)
                dest_tier = tier + 1
                if dest_tier > 3:
                    dest_tier = 3
                src = self.tiers[tier]
                dest = self.tiers[dest_tier]

                src.free_space(entry.size)
                entry.dirty = False

                if dest.can_allocate(entry.size):
                    dest.allocate(entry.size)
                    self.pt.update_tier(virtual_id, page_index, dest_tier, 0)
                else:
                    # SSD is full  drop the page
                    self.pt.remove(virtual_id, page_index)
                    total_freed += entry.size
                    continue

                total_freed += entry.size
            else:
                # Dropping from SSD
                self.tiers[3].free_space(entry.size)
                self.pt.remove(virtual_id, page_index)
                total_freed += entry.size

        return total_freed

    def record_fault(self) -> None:
        """Notify the policy of a page fault (for adaptive)."""
        if isinstance(self.policy, AdaptivePolicy):
            self.policy.record_access(is_fault=True)

    def record_hit(self) -> None:
        if isinstance(self.policy, AdaptivePolicy):
            self.policy.record_access(is_fault=False)
