"""RoutingPolicy — generates per-complexity layer execution plans.

A policy maps a complexity score (or type label) to a concrete list of
transformer layer indices that should be executed for a given query.
Three strategy families are provided:

    * **Static** — fixed, hand-crafted plans for each complexity tier.
    * **Adaptive** — starts from the static plan but adjusts based on
      a provided feedback signal (e.g. correctness or confidence).
    * **Experimental** — novel strategies for research exploration.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class RoutingPolicy(ABC):
    """Base class for all routing policies.

    Args:
        num_layers: Total number of layers in the transformer model.
    """

    def __init__(self, num_layers: int = 80) -> None:
        self.num_layers = num_layers
        self._layer_ids: List[int] = list(range(num_layers))

    @abstractmethod
    def select_layers(self, complexity_score: float, query_type: str) -> List[int]:
        """Return an ordered list of layer indices to execute.

        Args:
            complexity_score: Float in [0.0, 1.0].
            query_type: One of ``"simple"``, ``"medium"``, ``"complex"``.

        Returns:
            Sorted list of layer indices (0-based).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable policy name."""
        ...

    def describe(self) -> Dict[str, object]:
        """Return a dictionary describing the policy configuration."""
        return {"name": self.name, "num_layers": self.num_layers}


# ---------------------------------------------------------------------------
# Static policy
# ---------------------------------------------------------------------------


class StaticPolicy(RoutingPolicy):
    """Hand-crafted fixed plans for simple / medium / complex queries.

    Design rationale:
        * Early layers (0–5) and very late layers (75–79) are always
          kept because they handle token embedding and output decoding.
        * For **simple** queries we keep ~20 evenly spaced critical layers.
        * For **medium** queries we keep ~40 layers with denser mid-model sampling.
        * For **complex** queries all 80 layers are used.
    """

    # Layers that are *always* included regardless of complexity
    ALWAYS_INCLUDE: Set[int] = {0, 1, 2, 3, 4, 5, 75, 76, 77, 78, 79}

    def __init__(self, num_layers: int = 80) -> None:
        super().__init__(num_layers)
        self._plans: Dict[str, List[int]] = {
            "simple": self._build_simple_plan(),
            "medium": self._build_medium_plan(),
            "complex": self._build_complex_plan(),
        }
        logger.info(
            "StaticPolicy initialised  —  simple:%d medium:%d complex:%d layers",
            len(self._plans["simple"]),
            len(self._plans["medium"]),
            len(self._plans["complex"]),
        )

    def _build_simple_plan(self) -> List[int]:
        """~20 layers: early + sparse middle + late."""
        selected = set(self.ALWAYS_INCLUDE)
        # Sample every ~5th layer from the middle block
        for i in range(6, 75, 5):
            selected.add(i)
        return sorted(selected)

    def _build_medium_plan(self) -> List[int]:
        """~40 layers: early + dense middle + late."""
        selected = set(self.ALWAYS_INCLUDE)
        for i in range(6, 75, 2):
            selected.add(i)
        return sorted(selected)

    def _build_complex_plan(self) -> List[int]:
        """All layers."""
        return list(range(self.num_layers))

    def select_layers(self, complexity_score: float, query_type: str) -> List[int]:
        return list(self._plans.get(query_type, self._plans["simple"]))

    @property
    def name(self) -> str:
        return "static"


# ---------------------------------------------------------------------------
# Adaptive policy
# ---------------------------------------------------------------------------


class AdaptivePolicy(RoutingPolicy):
    """Starts with the static plan but grows the layer set when the
    model's confidence on intermediate logits falls below a threshold.

    This simulates a *speculative early-exit* mechanism: we execute a
    subset of layers, check an early-exit confidence, and add more
    layers if the confidence is too low.

    Args:
        num_layers: Total model layers.
        base_policy: The static policy to initialise from.
        confidence_threshold: Minimum early-exit confidence (0.0–1.0).
        growth_step: How many extra layers to add on each expansion.
        max_layers: Maximum layers to ever execute.
    """

    def __init__(
        self,
        num_layers: int = 80,
        base_policy: Optional[StaticPolicy] = None,
        confidence_threshold: float = 0.85,
        growth_step: int = 8,
        max_layers: Optional[int] = None,
    ) -> None:
        super().__init__(num_layers)
        self._base = base_policy or StaticPolicy(num_layers)
        self._threshold = confidence_threshold
        self._growth_step = growth_step
        self._max_layers = max_layers or num_layers
        self._feedback_history: List[float] = []

        logger.info(
            "AdaptivePolicy initialised  —  threshold=%.2f step=%d max=%d",
            confidence_threshold, growth_step, self._max_layers,
        )

    def select_layers(self, complexity_score: float, query_type: str) -> List[int]:
        base_layers = self._base.select_layers(complexity_score, query_type)
        base_set = set(base_layers)

        # Estimate how many more layers we might need based on recent feedback
        if self._feedback_history and sum(self._feedback_history) / len(self._feedback_history) < self._threshold:
            extra_count = min(self._growth_step, self._max_layers - len(base_set))
            candidates = [i for i in range(self.num_layers) if i not in base_set]
            if extra_count > 0 and candidates:
                # Pick layers adjacent to the existing gaps
                extra = self._pick_adjacent(candidates, base_set, extra_count)
                base_set.update(extra)
                logger.debug("AdaptivePolicy added %d extra layers", len(extra))

        return sorted(base_set)

    def _pick_adjacent(self, candidates: List[int], current: Set[int], count: int) -> List[int]:
        """Pick layers that fill gaps in the current execution set."""
        gaps = []
        for i in candidates:
            if i - 1 in current or i + 1 in current:
                gaps.append(i)
        random.shuffle(gaps)
        return gaps[:count]

    def record_feedback(self, confidence: float) -> None:
        """Record an early-exit confidence signal for future adjustments."""
        self._feedback_history.append(confidence)
        if len(self._feedback_history) > 100:
            self._feedback_history.pop(0)

    @property
    def name(self) -> str:
        return "adaptive"


# ---------------------------------------------------------------------------
# Experimental policy
# ---------------------------------------------------------------------------


class ExperimentalPolicy(RoutingPolicy):
    """Novel routing strategies for research exploration.

    Current strategies:
        * ``"even-odd"``: Split layers into two interleaved sets and
          alternate between queries (context-caching friendly).
        * ````"cluster"``: Group layers into functional clusters and select
          entire clusters based on query type.
        * ``"density"``: Sample layers with higher density near the input
          and output ends, sparse in the middle.
        * ``"random-topk"``: Select a random subset of *k* layers weighted
          by layer position (later layers weighted higher).

    Args:
        num_layers: Total model layers.
        strategy: One of the strategies listed above.
    """

    VALID_STRATEGIES = {"even-odd", "cluster", "density", "random-topk"}

    def __init__(self, num_layers: int = 80, strategy: str = "density") -> None:
        super().__init__(num_layers)
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Unknown experimental strategy: {strategy}. Choose from {self.VALID_STRATEGIES}")
        self._strategy = strategy
        logger.info("ExperimentalPolicy initialised  —  strategy=%s", strategy)

    def select_layers(self, complexity_score: float, query_type: str) -> List[int]:
        method = getattr(self, f"_strategy_{self._strategy.replace('-', '_')}")
        return method(complexity_score, query_type)

    def _strategy_even_odd(self, complexity_score: float, query_type: str) -> List[int]:
        """Interleaved half-layer execution."""
        if query_type == "complex":
            return list(range(self.num_layers))
        parity = 1 if complexity_score > 0.4 else 0
        return sorted({i for i in range(self.num_layers) if i % 2 == parity or i < 6 or i >= 74})

    def _strategy_cluster(self, complexity_score: float, query_type: str) -> List[int]:
        """Select functional clusters of layers."""
        clusters = {
            "simple":  [list(range(0, 10)), list(range(70, 80))],
            "medium":  [list(range(0, 15)), list(range(30, 45)), list(range(65, 80))],
            "complex": [list(range(0, 80))],
        }
        selected: Set[int] = set()
        for cluster in clusters.get(query_type, clusters["simple"]):
            selected.update(cluster)
        return sorted(selected)

    def _strategy_density(self, complexity_score: float, query_type: str) -> List[int]:
        """Denser near input/output, sparse in the middle.

        The higher the complexity, the denser the sampling.
        """
        if query_type == "complex":
            return list(range(self.num_layers))

        # Density factor: simple → 0.15, medium → 0.4
        density_map = {"simple": 0.15, "medium": 0.40, "complex": 1.0}
        density = density_map.get(query_type, 0.15)

        selected: Set[int] = set()
        for i in range(self.num_layers):
            # Always keep boundaries
            if i < 6 or i >= 74:
                selected.add(i)
                continue
            # Variable density: more at edges, less in the middle
            normalised_pos = i / self.num_layers  # 0.0 – 1.0
            edge_weight = 1.0 - 2.0 * abs(normalised_pos - 0.5)  # peaks at 0 and 1
            keep_prob = density * (0.5 + 0.5 * edge_weight)
            if random.random() < keep_prob:
                selected.add(i)

        return sorted(selected)

    def _strategy_random_topk(self, complexity_score: float, query_type: str) -> List[int]:
        """Weighted random subset — later layers have higher weight."""
        if query_type == "complex":
            return list(range(self.num_layers))

        k_map = {"simple": 20, "medium": 40}
        k = k_map.get(query_type, 20)

        # Weights increase linearly with layer index
        weights = [1.0 + i / self.num_layers for i in range(self.num_layers)]
        # Boost boundaries
        for i in list(range(6)) + list(range(74, self.num_layers)):
            weights[i] *= 5.0

        chosen = set(random.choices(range(self.num_layers), weights=weights, k=k))
        return sorted(chosen)

    @property
    def name(self) -> str:
        return f"experimental/{self._strategy}"


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------


def create_policy(
    name: str,
    num_layers: int = 80,
    **kwargs,
) -> RoutingPolicy:
    """Factory function — returns a policy instance by name.

    Args:
        name: ``"static"``, ``"adaptive"``, or ``"experimental/<strategy>"``.
        num_layers: Total model layers.
        **kwargs: Passed to the policy constructor.

    Returns:
        A configured :class:`RoutingPolicy` instance.
    """
    if name == "static":
        return StaticPolicy(num_layers, **kwargs)
    if name == "adaptive":
        return AdaptivePolicy(num_layers, **kwargs)
    if name.startswith("experimental/"):
        strategy = name.split("/", 1)[1]
        return ExperimentalPolicy(num_layers, strategy=strategy, **kwargs)
    raise ValueError(f"Unknown policy: {name}")
