"""Prefetcher — predicts and preloads upcoming transformer layers.

Uses a configurable prediction strategy to forecast which layer(s)
will be needed next and proactively loads them into the cache,
reducing future page-fault latency.

Strategies implemented:
    - ``sequential``: Always predict the next layer (``current + 1``).
    - ``transition_matrix``: Learn a 1st-order Markov chain over layer
      transitions from observed access patterns.  Predict the top-K most
      probable successors.
    - ``oracle``: Perfect prediction using a pre-recorded access sequence
      (useful for evaluation / upper-bound measurement).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


class PrefetchStrategy:
    """Enum-like namespace for strategy names."""
    SEQUENTIAL = "sequential"
    TRANSITION_MATRIX = "transition_matrix"
    ORACLE = "oracle"


class Prefetcher:
    """Prefetches transformer layers into the cache before they are needed.

    Args:
        strategy: One of ``"sequential"``, ``"transition_matrix"``,
            ``"oracle"``.
        num_layers: Total number of layers in the model (for bounds
            checking).
        top_k: Number of successor layers to prefetch per call
            (only used by transition-matrix strategy).
        oracle_sequence: If *strategy* is ``"oracle"``, this sequence
            of layer IDs is used for perfect prediction.
        prefetch_threshold: Only prefetch if the predicted layer is
            not already in cache.  This is a performance safeguard.
    """

    def __init__(
        self,
        strategy: str = PrefetchStrategy.SEQUENTIAL,
        num_layers: int = 80,
        top_k: int = 2,
        oracle_sequence: Optional[Sequence[int]] = None,
        prefetch_threshold: int = 1,
    ) -> None:
        if strategy not in (PrefetchStrategy.SEQUENTIAL, PrefetchStrategy.TRANSITION_MATRIX, PrefetchStrategy.ORACLE):
            raise ValueError(f"Unknown prefetch strategy: {strategy}")

        self.strategy = strategy
        self.num_layers = num_layers
        self.top_k = top_k
        self._oracle_sequence: List[int] = list(oracle_sequence) if oracle_sequence else []
        self._oracle_index: int = 0
        self._prefetch_threshold = prefetch_threshold

        # Transition matrix: count[i][j] = number of observed i→j transitions
        self._transition_counts: List[Dict[int, int]] = [defaultdict(int) for _ in range(num_layers)]
        self._last_layer: Optional[int] = None
        self._total_predictions: int = 0
        self._correct_predictions: int = 0

        logger.info("Prefetcher initialised (strategy=%s, top_k=%d)", strategy, top_k)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, layer_id: int) -> None:
        """Record a layer access for learning transition patterns.

        Call this on every cache hit / load so the transition matrix
        can be updated.
        """
        if self._last_layer is not None and 0 <= self._last_layer < self.num_layers:
            self._transition_counts[self._last_layer][layer_id] += 1
        self._last_layer = layer_id

    def predict_next(self, current_layer: int, history: Optional[Sequence[int]] = None) -> List[int]:
        """Return a list of predicted next-layer IDs.

        Args:
            current_layer: The layer ID most recently accessed.
            history: Past access sequence (used by some strategies).

        Returns:
            Sorted list of predicted layer IDs to prefetch.
        """
        self._total_predictions += 1

        if self.strategy == PrefetchStrategy.ORACLE:
            return self._predict_oracle()

        if self.strategy == PrefetchStrategy.TRANSITION_MATRIX:
            return self._predict_transition(current_layer)

        # Default: sequential
        return self._predict_sequential(current_layer)

    def prefetch(
        self,
        layer_ids: List[int],
        load_fn,
        cache_contains_fn,
    ) -> Tuple[int, int]:
        """Prefetch a batch of layers.

        Args:
            layer_ids: Predicted layer IDs to attempt loading.
            load_fn: Callable ``(layer_id) -> LayerMeta``, typically
                ``LayerStore.load_layer``.
            cache_contains_fn: Callable ``(layer_id) -> bool`` to check
                cache residency.

        Returns:
            ``(prefetch_attempts, prefetch_successes)``.
        """
        attempts = 0
        successes = 0
        for lid in layer_ids:
            if not (0 <= lid < self.num_layers):
                continue
            if cache_contains_fn(lid):
                continue  # already present → skip
            attempts += 1
            meta = load_fn(lid)
            successes += 1
            self._correct_predictions += 1
            logger.debug("Prefetched layer %d  (%.1f MB)", lid, meta.size_mb)

        return attempts, successes

    def get_accuracy(self) -> float:
        """Fraction of predictions that ultimately led to a prefetch hit."""
        if self._total_predictions == 0:
            return 0.0
        return self._correct_predictions / self._total_predictions

    # ------------------------------------------------------------------
    # Internal prediction methods
    # ------------------------------------------------------------------

    def _predict_sequential(self, current_layer: int) -> List[int]:
        """Predict the next N consecutive layers."""
        next_id = current_layer + 1
        if next_id >= self.num_layers:
            return []
        return list(range(next_id, min(next_id + self._prefetch_threshold, self.num_layers)))

    def _predict_transition(self, current_layer: int) -> List[int]:
        """Predict the top-K most frequent successors via the Markov chain."""
        counts = self._transition_counts[current_layer]
        if not counts:
            # Fall back to sequential when no data is available
            return self._predict_sequential(current_layer)

        sorted_successors = sorted(counts.items(), key=lambda x: -x[1])
        return [lid for lid, _ in sorted_successors[:self.top_k]]

    def _predict_oracle(self) -> List[int]:
        """Return the next layer from the pre-recorded oracle sequence."""
        if self._oracle_index >= len(self._oracle_sequence):
            return []
        next_id = self._oracle_sequence[self._oracle_index]
        self._oracle_index += 1
        # Also bring along its sequential neighbour for safety
        neighbours = [next_id]
        if next_id + 1 < self.num_layers:
            neighbours.append(next_id + 1)
        return neighbours

    def reset_oracle(self) -> None:
        """Reset the oracle sequence index (useful between test runs)."""
        self._oracle_index = 0
