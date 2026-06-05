"""prefetch_engine  predicts future page accesses and preloads pages.

Predictors:
    * Sequential  if pages 0, 1, 2 are accessed, predict 3.
    * Stride  detect constant strides (e.g. pages 0, 2, 4  predict 6).
    * Frequency  predict pages with the highest recent access frequency.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from allocator import Allocator
from memory_tiers import TierManager
from page_table import PageTable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prefetch predictors
# ---------------------------------------------------------------------------


class SequentialPredictor:
    """Predict the next page index based on sequential access patterns.

    If the last *k* accesses for a virtual component were consecutive,
    predict the next one.
    """

    def __init__(self, window: int = 3) -> None:
        self.window = window
        self._history: Dict[int, List[int]] = defaultdict(
            lambda: deque(maxlen=window + 1)
        )

    def record(self, virtual_id: int, page_index: int) -> None:
        self._history[virtual_id].append(page_index)

    def predict(self, virtual_id: int) -> Optional[int]:
        hist = list(self._history.get(virtual_id, []))
        if len(hist) < self.window:
            return None
        last_n = hist[-self.window:]
        if all(last_n[i] + 1 == last_n[i + 1] for i in range(self.window - 1)):
            return last_n[-1] + 1
        return None


class StridePredictor:
    """Detect constant-stride access patterns.

    If a component is accessed with a regular stride (e.g. 2, 4, 6 stride 2),
    predict the next step.
    """

    def __init__(self, min_observations: int = 3) -> None:
        self.min_obs = min_observations
        self._history: Dict[int, List[int]] = defaultdict(list)

    def record(self, virtual_id: int, page_index: int) -> None:
        self._history[virtual_id].append(page_index)

    def predict(self, virtual_id: int) -> Optional[int]:
        hist = self._history.get(virtual_id, [])
        if len(hist) < self.min_obs:
            return None
        recent = hist[-self.min_obs:]
        stride = recent[1] - recent[0]
        if all(recent[i + 1] - recent[i] == stride for i in range(len(recent) - 1)):
            return recent[-1] + stride
        return None


class FrequencyPredictor:
    """Predict the most frequently accessed pages.

    Useful for KV cache pages that receive high attention.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self._freq: Dict[Tuple[int, int], int] = defaultdict(int)

    def record(self, virtual_id: int, page_index: int) -> None:
        self._freq[(virtual_id, page_index)] += 1

    def predict(self) -> List[Tuple[int, int]]:
        sorted_pages = sorted(
            self._freq.items(), key=lambda x: x[1], reverse=True
        )
        return [page for page, _ in sorted_pages[:self.top_k]]


# ---------------------------------------------------------------------------
# Prefetch engine
# ---------------------------------------------------------------------------


class PrefetchEngine:
    """Orchestrates prefetching across multiple predictors.

    Args:
        allocator: System page allocator.
        page_table: System page table.
        tier_manager: Memory tier hierarchy.
        prefetch_window: Maximum number of pages to prefetch per trigger.
        prefetch_tier: Target tier for prefetched pages (default RAM).
    """

    def __init__(
        self,
        allocator: Allocator,
        page_table: PageTable,
        tier_manager: TierManager,
        prefetch_window: int = 4,
        prefetch_tier: int = 1,
    ) -> None:
        self.allocator = allocator
        self.pt = page_table
        self.tiers = tier_manager
        self.prefetch_window = prefetch_window
        self.prefetch_tier = prefetch_tier

        # Predictors
        self.sequential = SequentialPredictor()
        self.stride = StridePredictor()
        self.frequency = FrequencyPredictor()

        # Stats
        self.total_prefetches: int = 0
        self.useful_prefetches: int = 0
        self.useless_prefetches: int = 0

    def record_access(self, virtual_id: int, page_index: int) -> None:
        """Record a page access for all predictors."""
        self.sequential.record(virtual_id, page_index)
        self.stride.record(virtual_id, page_index)
        self.frequency.record(virtual_id, page_index)

    def predict_and_prefetch(self, virtual_id: int, page_index: int) -> int:
        """Run predictors and prefetch predicted pages.

        Args:
            virtual_id: Currently accessed virtual component.
            page_index: Currently accessed page index.

        Returns:
            Number of pages prefetched.
        """
        self.record_access(virtual_id, page_index)
        predictions: Set[Tuple[int, int]] = set()

        # Sequential prediction
        seq = self.sequential.predict(virtual_id)
        if seq is not None:
            for i in range(self.prefetch_window):
                predictions.add((virtual_id, seq + i))

        # Stride prediction
        stride = self.stride.predict(virtual_id)
        if stride is not None:
            predictions.add((virtual_id, stride))

        # Frequency prediction (for KV cache)
        for freq_page in self.frequency.predict():
            predictions.add(freq_page)

        # Filter to pages not already in prefetch_tier
        to_prefetch: List[Tuple[int, int]] = []
        for vid, pidx in predictions:
            entry = self.pt.get(vid, pidx)
            if entry is None:
                continue
            if entry.tier == self.prefetch_tier:
                continue
            if entry.tier < self.prefetch_tier:
                continue
            to_prefetch.append((vid, pidx))

        # Prefetch
        count = 0
        for vid, pidx in to_prefetch:
            entry = self.pt.get(vid, pidx)
            if entry is None:
                continue
            success = self.allocator.move_page(vid, pidx, self.prefetch_tier)
            if success:
                count += 1
                self.total_prefetches += 1

        return count

    def mark_useful(self, virtual_id: int, page_index: int) -> None:
        """Mark a previously prefetched page as actually used."""
        entry = self.pt.get(virtual_id, page_index)
        if entry is not None:
            self.useful_prefetches += 1

    def mark_useless(self, virtual_id: int, page_index: int) -> None:
        self.useless_prefetches += 1

    @property
    def prefetch_accuracy(self) -> float:
        total = self.useful_prefetches + self.useless_prefetches
        if total == 0:
            return 1.0
        return self.useful_prefetches / max(total, 1)

    def summary(self) -> Dict[str, object]:
        return {
            "total_prefetches": self.total_prefetches,
            "useful_prefetches": self.useful_prefetches,
            "useless_prefetches": self.useless_prefetches,
            "prefetch_accuracy_pct": round(self.prefetch_accuracy * 100, 1),
        }

    def reset(self) -> None:
        self.sequential = SequentialPredictor()
        self.stride = StridePredictor()
        self.frequency = FrequencyPredictor()
        self.total_prefetches = 0
        self.useful_prefetches = 0
        self.useless_prefetches = 0
