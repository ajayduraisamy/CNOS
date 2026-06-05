"""allocator  allocates pages across memory tiers with promotion/demotion.

Decides which tier a new page should live in, and handles moving pages
between tiers when the working set changes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from memory_tiers import TierManager
from page_table import PageTable, PageTableEntry
from eviction_manager import EvictionManager

logger = logging.getLogger(__name__)

# Default page size (1 MB)
DEFAULT_PAGE_SIZE = 1024 * 1024


class Allocator:
    """Allocates pages across memory tiers.

    Args:
        tier_manager: The tier hierarchy.
        page_table: The system page table.
        eviction_manager: Eviction policy for freeing space.
        page_size: Default page size in bytes.
    """

    def __init__(
        self,
        tier_manager: TierManager,
        page_table: PageTable,
        eviction_manager: Optional[EvictionManager] = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        self.tiers = tier_manager
        self.page_table = page_table
        self.evict = eviction_manager or EvictionManager(page_table, tier_manager)
        self.page_size = page_size
        self._next_virtual_id: int = 0
        self._tier_offsets: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

    # ------------------------------------------------------------------
    # Virtual ID allocation
    # ------------------------------------------------------------------

    def allocate_virtual_id(self) -> int:
        """Return a new unique virtual component ID."""
        vid = self._next_virtual_id
        self._next_virtual_id += 1
        return vid

    # ------------------------------------------------------------------
    # Page allocation
    # ------------------------------------------------------------------

    def allocate_pages(
        self,
        virtual_id: int,
        num_pages: int,
        preferred_tier: int = 1,
        size: Optional[int] = None,
    ) -> List[PageTableEntry]:
        """Allocate contiguous pages for a virtual component.

        Args:
            virtual_id: The virtual component ID.
            num_pages: Number of pages to allocate.
            preferred_tier: Target tier (default RAM).
            size: Page size in bytes (defaults to ``self.page_size``).

        Returns:
            List of created :class:`PageTableEntry` objects.
        """
        size = size or self.page_size
        entries: List[PageTableEntry] = []

        # Try preferred tier first; fall back to slower tiers
        for attempt_tier in [preferred_tier, 3, 2, 1, 0]:
            tier = self.tiers[attempt_tier]
            needed = num_pages * size

            if tier.can_allocate(needed):
                for i in range(num_pages):
                    offset = self._tier_offsets[attempt_tier]
                    entry = PageTableEntry(
                        virtual_id=virtual_id,
                        page_index=i,
                        tier=attempt_tier,
                        offset=offset,
                        size=size,
                        last_access_time=0,
                        access_count=0,
                    )
                    self.page_table.add(entry)
                    self._tier_offsets[attempt_tier] += size
                    entries.append(entry)
                tier.allocate(needed)
                break

            # Need to evict from this tier
            if attempt_tier <= 2:
                to_free = needed - tier.free
                evicted = self.evict.free_space(attempt_tier, to_free)
                if tier.can_allocate(needed):
                    for i in range(num_pages):
                        offset = self._tier_offsets[attempt_tier]
                        entry = PageTableEntry(
                            virtual_id=virtual_id,
                            page_index=i,
                            tier=attempt_tier,
                            offset=offset,
                            size=size,
                            access_count=0,
                        )
                        self.page_table.add(entry)
                        self._tier_offsets[attempt_tier] += size
                        entries.append(entry)
                    tier.allocate(needed)
                    break
        else:
            raise MemoryError(
                f"Cannot allocate {num_pages} pages of {size} bytes "
                f"in any tier"
            )

        return entries

    # ------------------------------------------------------------------
    # Page movement
    # ------------------------------------------------------------------

    def move_page(
        self,
        virtual_id: int,
        page_index: int,
        target_tier: int,
    ) -> bool:
        """Move a page to a different tier.  Returns ``True`` on success.

        Handles allocation in the target tier and freeing from the source.
        """
        entry = self.page_table.get(virtual_id, page_index)
        if entry is None:
            return False
        if entry.tier == target_tier:
            return True

        source_tier = self.tiers[entry.tier]
        dest_tier = self.tiers[target_tier]

        # Free from source
        source_tier.free_space(entry.size)

        # Allocate in destination
        if not dest_tier.can_allocate(entry.size):
            self.evict.free_space(target_tier, entry.size - dest_tier.free)

        if not dest_tier.can_allocate(entry.size):
            # Rollback: put back in source
            source_tier.allocate(entry.size)
            return False

        offset = self._tier_offsets[target_tier]
        dest_tier.allocate(entry.size)
        self._tier_offsets[target_tier] += entry.size

        # Update page table
        self.page_table.update_tier(virtual_id, page_index, target_tier, offset)

        return True

    def promote(self, virtual_id: int, page_index: int) -> bool:
        """Move a page one tier closer to compute (e.g. SSD  RAM)."""
        entry = self.page_table.get(virtual_id, page_index)
        if entry is None or entry.tier <= 0:
            return False
        return self.move_page(virtual_id, page_index, entry.tier - 1)

    def demote(self, virtual_id: int, page_index: int) -> bool:
        """Move a page one tier away from compute (e.g. RAM  SSD)."""
        entry = self.page_table.get(virtual_id, page_index)
        if entry is None or entry.tier >= 3:
            return False
        return self.move_page(virtual_id, page_index, entry.tier + 1)

    def free_pages(self, virtual_id: int) -> int:
        """Free all pages for a virtual component.  Returns bytes freed."""
        freed = 0
        to_remove: List[Tuple[int, int]] = []
        for entry in self.page_table.entries_for_virtual(virtual_id):
            self.tiers[entry.tier].free_space(entry.size)
            freed += entry.size
            to_remove.append((entry.virtual_id, entry.page_index))
        for vid, pidx in to_remove:
            self.page_table.remove(vid, pidx)
        return freed

    def reset(self) -> None:
        self._next_virtual_id = 0
        self._tier_offsets = {0: 0, 1: 0, 2: 0, 3: 0}
