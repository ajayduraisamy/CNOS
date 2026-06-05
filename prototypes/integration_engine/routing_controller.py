"""routing_controller  bridges ComplexityDetector + LayerSelector.

Analyses a user query, classifies its complexity, and selects which
transformer layers to execute (and which to skip).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_PROTO = os.path.join(os.path.dirname(__file__), "..", "layer_router")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    query: str
    complexity_score: float
    query_type: str
    reasoning_depth: str
    confidence: float
    selected_layers: List[int]
    skipped_layers: List[int]
    policy_name: str
    num_selected: int
    num_skipped: int
    compute_reduction_pct: float
    features: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "query": self.query[:60] + "..." if len(self.query) > 60 else self.query,
            "complexity": self.query_type,
            "score": round(self.complexity_score, 3),
            "confidence": round(self.confidence, 3),
            "layers_selected": self.num_selected,
            "layers_skipped": self.num_skipped,
            "reduction_pct": round(self.compute_reduction_pct, 1),
            "policy": self.policy_name,
        }


class RoutingController:
    """Analyses query complexity and selects the optimal layer execution plan.

    Args:
        num_layers: Total transformer layers.
        policy_name: Routing policy name (``"static"``, ``"adaptive"``,
            ``"experimental/density"``).
    """

    def __init__(
        self,
        num_layers: int = 22,
        policy_name: str = "adaptive",
    ) -> None:
        self.num_layers = num_layers

        from complexity_detector import ComplexityDetector
        from layer_selector import LayerSelector
        from routing_policy import create_policy

        self.detector = ComplexityDetector(num_layers=num_layers)
        self.selector = LayerSelector(num_layers=num_layers)

        self.policy = create_policy(policy_name, num_layers=num_layers)
        self.selector.set_policy(self.policy)

        logger.info(
            "RoutingController: num_layers=%d policy=%s",
            num_layers, policy_name,
        )

    def select_layers(self, query: str) -> RoutingResult:
        selection = self.selector.select(query)
        all_layers = set(range(self.num_layers))

        # Clamp to valid layer range (policies may be designed for 80 layers)
        valid_selected = [l for l in selection.selected_layers if 0 <= l < self.num_layers]
        active = set(valid_selected)
        skipped = sorted(all_layers - active)
        num_skipped = len(skipped)
        total = self.num_layers
        reduction_pct = (num_skipped / total * 100) if total > 0 else 0.0

        return RoutingResult(
            query=selection.query,
            complexity_score=selection.complexity.complexity_score,
            query_type=selection.complexity.query_type,
            reasoning_depth=selection.complexity.reasoning_depth,
            confidence=selection.complexity.confidence,
            selected_layers=valid_selected,
            skipped_layers=skipped,
            policy_name=selection.policy_name,
            num_selected=len(valid_selected),
            num_skipped=num_skipped,
            compute_reduction_pct=reduction_pct,
            features=selection.complexity.features,
        )

    def set_policy(self, policy_name: str) -> None:
        from routing_policy import create_policy
        self.policy = create_policy(policy_name, num_layers=self.num_layers)
        self.selector.set_policy(self.policy)
        logger.info("RoutingController: policy changed to %s", policy_name)
