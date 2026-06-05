"""importance_router — importance-aware layer skipping for transformer inference.

Uses the :class:`LayerProfile` to decide which layers to skip while
preserving critical (high-impact) layers.  Supports three operating modes
with user-configurable skip budgets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from layer_profile import LayerProfile

logger = logging.getLogger(__name__)


class RoutingMode(Enum):
    """Operating mode controlling the aggressiveness of layer skipping."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

    @staticmethod
    def from_str(s: str) -> RoutingMode:
        for m in RoutingMode:
            if m.value == s.lower():
                return m
        raise ValueError(f"Unknown mode: {s}.  Choose from conservative, balanced, aggressive.")


# Default skip budgets per mode: (min, max)
_MODE_BUDGET: Dict[RoutingMode, Tuple[int, int]] = {
    RoutingMode.CONSERVATIVE: (1, 2),
    RoutingMode.BALANCED: (2, 3),
    RoutingMode.AGGRESSIVE: (3, 4),
}


@dataclass
class RoutingDecision:
    """Result of a routing decision.

    Attributes:
        mode: Operating mode used.
        skip_layers: Indices of layers to skip.
        active_layers: Indices of layers to execute.
        num_skipped: Count of skipped layers.
        compute_reduction_pct: Percentage of layers skipped.
        budget: (min, max) skip budget for this mode.
        impact_scores_skipped: Impact scores of skipped layers.
    """
    mode: RoutingMode = RoutingMode.CONSERVATIVE
    skip_layers: Set[int] = field(default_factory=set)
    active_layers: Set[int] = field(default_factory=set)
    num_skipped: int = 0
    compute_reduction_pct: float = 0.0
    budget: Tuple[int, int] = (1, 2)
    impact_scores_skipped: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode.value,
            "skip_layers": sorted(self.skip_layers),
            "num_skipped": self.num_skipped,
            "compute_reduction_pct": round(self.compute_reduction_pct, 2),
            "budget_min": self.budget[0],
            "budget_max": self.budget[1],
        }


class ImportanceRouter:
    """Decides which layers to skip based on importance profile.

    Guarantees:
        * Never skips critical (high impact) layers.
        * Skips only medium/low impact layers.
        * Respects the skip budget for the selected mode.

    Args:
        profile: A :class:`LayerProfile` with per-layer scores.
    """

    def __init__(self, profile: LayerProfile) -> None:
        self.profile = profile
        self.num_layers = profile.num_layers
        logger.info(
            "ImportanceRouter ready  %d layers  %d skip candidates",
            self.num_layers, profile.max_skippable(),
        )

    def decide(
        self,
        mode: RoutingMode = RoutingMode.CONSERVATIVE,
        custom_budget: Optional[Tuple[int, int]] = None,
    ) -> RoutingDecision:
        """Produce a routing decision for the given mode.

        Args:
            mode: Operating mode.
            custom_budget: Optional ``(min, max)`` override.

        Returns:
            A :class:`RoutingDecision` with skip/active layer sets.
        """
        budget = custom_budget or _MODE_BUDGET[mode]
        min_skip, max_skip = budget

        candidates = self.profile.skip_candidates
        if not candidates:
            logger.warning("No skip candidates available")
            return self._all_active(mode, budget)

        # Pick the lowest-impact candidates up to the budget
        num_to_skip = min(len(candidates), max_skip)
        # Ensure we skip at least min_skip if possible
        num_to_skip = max(num_to_skip, min(min_skip, len(candidates)))
        # But cap at max_skip
        num_to_skip = min(num_to_skip, max_skip)

        selected = candidates[:num_to_skip]
        skip_layers = {l for l, _ in selected}
        impact_scores = [s for _, s in selected]

        active_layers = set(range(self.num_layers)) - skip_layers
        reduction = (len(skip_layers) / self.num_layers) * 100.0

        logger.info(
            "Routing decision  mode=%s  skip=%s  reduction=%.1f%%",
            mode.value, sorted(skip_layers), reduction,
        )

        return RoutingDecision(
            mode=mode,
            skip_layers=skip_layers,
            active_layers=active_layers,
            num_skipped=len(skip_layers),
            compute_reduction_pct=reduction,
            budget=budget,
            impact_scores_skipped=impact_scores,
        )

    def decide_pct(
        self,
        reduction_target_pct: float = 10.0,
    ) -> RoutingDecision:
        """Decide routing to achieve a given compute reduction percentage.

        Skips the lowest-impact candidates until the reduction target
        is met or all candidates are exhausted.

        Args:
            reduction_target_pct: Target percentage of layers to skip.

        Returns:
            A :class:`RoutingDecision`.
        """
        candidates = self.profile.skip_candidates
        target_count = max(1, int(self.num_layers * reduction_target_pct / 100.0))
        num_to_skip = min(len(candidates), target_count)

        selected = candidates[:num_to_skip]
        skip_layers = {l for l, _ in selected}

        active_layers = set(range(self.num_layers)) - skip_layers
        reduction = (len(skip_layers) / self.num_layers) * 100.0

        return RoutingDecision(
            mode=RoutingMode.BALANCED,
            skip_layers=skip_layers,
            active_layers=active_layers,
            num_skipped=len(skip_layers),
            compute_reduction_pct=reduction,
            budget=(num_to_skip, num_to_skip),
            impact_scores_skipped=[s for _, s in selected],
        )

    def _all_active(self, mode: RoutingMode, budget: Tuple[int, int]) -> RoutingDecision:
        active = set(range(self.num_layers))
        return RoutingDecision(
            mode=mode,
            active_layers=active,
            num_skipped=0,
            compute_reduction_pct=0.0,
            budget=budget,
        )

    def to_dict(self) -> Dict:
        return {
            "num_layers": self.num_layers,
            "skip_candidates": self.profile.skip_candidates,
        }
