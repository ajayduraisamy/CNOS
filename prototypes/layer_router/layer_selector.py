"""LayerSelector — unified interface for complexity-aware layer selection.

Wraps a :class:`RoutingPolicy` and a :class:`ComplexityDetector` into a
single callable that takes a user query and returns the set of layers
the inference engine should execute, together with explanatory metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from complexity_detector import ComplexityDetector, ComplexityResult
from routing_policy import RoutingPolicy, create_policy

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Output of a single layer-selection pass.

    Attributes:
        query: Original user query.
        complexity: Full complexity analysis result.
        selected_layers: Ordered list of layer indices to execute.
        policy_name: Name of the routing policy used.
        num_skipped: Number of layers skipped (total - selected).
        compute_reduction_pct: Percentage of layers skipped.
    """

    query: str
    complexity: ComplexityResult
    selected_layers: List[int]
    policy_name: str
    num_skipped: int = 0
    compute_reduction_pct: float = 0.0


class LayerSelector:
    """Selects transformer layers based on query complexity.

    Args:
        detector: A :class:`ComplexityDetector` instance.
        policy: A :class:`RoutingPolicy` instance.
        num_layers: Total layers in the model (used for reporting).
    """

    def __init__(
        self,
        detector: Optional[ComplexityDetector] = None,
        policy: Optional[RoutingPolicy] = None,
        num_layers: int = 80,
    ) -> None:
        self.detector = detector or ComplexityDetector(num_layers)
        self.policy = policy or create_policy("static", num_layers)
        self.num_layers = num_layers

        logger.info(
            "LayerSelector ready  —  policy=%s  num_layers=%d",
            self.policy.name, num_layers,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select(self, query: str) -> SelectionResult:
        """Analyse a query and return the selected execution plan.

        Args:
            query: Raw user input.

        Returns:
            A :class:`SelectionResult` with the layer plan and metadata.
        """
        complexity = self.detector.analyse(query)
        layers = self.policy.select_layers(complexity.complexity_score, complexity.query_type)

        num_selected = len(layers)
        num_skipped = self.num_layers - num_selected
        reduction = (num_skipped / self.num_layers) * 100.0

        logger.info(
            "Query [%s] -> %s (%s)  |  layers=%d/%d  skipped=%d  reduction=%.1f%%",
            query[:50].replace("\n", " "),
            complexity.query_type,
            complexity.reasoning_depth,
            num_selected,
            self.num_layers,
            num_skipped,
            reduction,
        )

        return SelectionResult(
            query=query,
            complexity=complexity,
            selected_layers=layers,
            policy_name=self.policy.name,
            num_skipped=num_skipped,
            compute_reduction_pct=round(reduction, 2),
        )

    def set_policy(self, policy: RoutingPolicy) -> None:
        """Swap the routing policy at runtime."""
        self.policy = policy
        logger.info("LayerSelector policy changed to: %s", policy.name)
