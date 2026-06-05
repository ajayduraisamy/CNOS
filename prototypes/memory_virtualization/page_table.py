"""page_table  maps virtual pages to physical locations across memory tiers.

Each entry records which tier a page lives in, its byte offset, size,
access timestamps, and flags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PageTableEntry:
    """Single entry mapping one virtual page to a physical location.

    Attributes:
        virtual_id: Logical component identifier (e.g. layer number).
        page_index: Page offset within the virtual component.
        tier: Current tier ID (03).
        offset: Byte offset within the tier's storage.
        size: Page size in bytes.
        last_access_time: Global timestamp of most recent access.
        access_count: Total number of accesses.
        dirty: Whether the page has been modified since loaded.
        locked: Whether the page is pinned (cannot be evicted).
        present: Whether the page is loaded in any tier.
    """
    virtual_id: int = 0
    page_index: int = 0
    tier: int = 1  # default: RAM
    offset: int = 0
    size: int = 0
    last_access_time: int = 0
    access_count: int = 0
    dirty: bool = False
    locked: bool = False
    present: bool = True


class PageTable:
    """Page table for one virtual address space.

    Supports lookup by ``(virtual_id, page_index)``, iteration by tier,
    and bulk tracking of free/used entries.

    Args:
        max_pages: Maximum number of tracked entries.
    """

    def __init__(self, max_pages: int = 1_000_000) -> None:
        self.max_pages = max_pages
        self._entries: Dict[Tuple[int, int], PageTableEntry] = {}
        self._by_tier: Dict[int, Dict[Tuple[int, int], PageTableEntry]] = {
            0: {}, 1: {}, 2: {}, 3: {},
        }
        self._global_clock: int = 0

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, virtual_id: int, page_index: int) -> Optional[PageTableEntry]:
        """Look up an entry by virtual address.  Returns ``None`` if absent."""
        return self._entries.get((virtual_id, page_index))

    def __getitem__(self, key: Tuple[int, int]) -> PageTableEntry:
        entry = self.get(*key)
        if entry is None:
            raise KeyError(f"Page ({key[0]}, {key[1]}) not in page table")
        return entry

    def __contains__(self, key: Tuple[int, int]) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, entry: PageTableEntry) -> None:
        """Insert a new page table entry."""
        key = (entry.virtual_id, entry.page_index)
        if key in self._entries:
            raise ValueError(f"Page {key} already exists")
        if len(self._entries) >= self.max_pages:
            raise MemoryError("Page table full")
        self._entries[key] = entry
        self._by_tier[entry.tier][key] = entry

    def remove(self, virtual_id: int, page_index: int) -> None:
        """Remove a page from the table."""
        key = (virtual_id, page_index)
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._by_tier[entry.tier].pop(key, None)

    def update_tier(
        self,
        virtual_id: int,
        page_index: int,
        new_tier: int,
        new_offset: int,
    ) -> None:
        """Change the physical location of a page."""
        key = (virtual_id, page_index)
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"Page {key} not found")
        self._by_tier[entry.tier].pop(key, None)
        entry.tier = new_tier
        entry.offset = new_offset
        self._by_tier[new_tier][key] = entry

    def mark_accessed(self, virtual_id: int, page_index: int) -> None:
        """Record an access (updates last_access_time and access_count)."""
        entry = self.get(virtual_id, page_index)
        if entry is not None:
            self._global_clock += 1
            entry.last_access_time = self._global_clock
            entry.access_count += 1

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def entries_in_tier(self, tier: int) -> Iterator[PageTableEntry]:
        """Yield all entries currently in a given tier."""
        yield from self._by_tier[tier].values()

    def entries_for_virtual(self, virtual_id: int) -> Iterator[PageTableEntry]:
        """Yield all entries for a given virtual component."""
        for key, entry in self._entries.items():
            if key[0] == virtual_id:
                yield entry

    @property
    def total_pages(self) -> int:
        return len(self._entries)

    @property
    def pages_by_tier(self) -> Dict[int, int]:
        return {
            tid: len(self._by_tier[tid]) for tid in range(4)
        }

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._entries.clear()
        for tid in range(4):
            self._by_tier[tid].clear()
        self._global_clock = 0

    def summary(self) -> Dict[str, object]:
        return {
            "total_entries": self.total_pages,
            "max_entries": self.max_pages,
            "pages_by_tier": self.pages_by_tier,
            "global_clock": self._global_clock,
        }
