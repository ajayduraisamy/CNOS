"""layer_ablation — measures the quality impact of disabling each transformer layer.

For each test query:
  1. Generate baseline response (all layers active).
  2. For each layer index, generate a response with *only that layer* disabled.
  3. Compare each ablated response against the baseline using quality metrics.
  4. Compute a Layer Impact Score for each layer.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

_PROTO_REAL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "real_inference")
)
if _PROTO_REAL not in sys.path:
    sys.path.insert(0, _PROTO_REAL)

from model_loader import load_model, ModelBundle
from routed_inference import RoutedInferenceEngine

from quality_metrics import (
    ComparisonResult,
    compare_responses,
    classify_importance,
)

logger = logging.getLogger(__name__)

# Standard benchmark queries across complexity levels
ABLATION_QUERIES: List[Dict[str, str]] = [
    {"query": "What is 2+2?", "type": "simple"},
    {"query": "What is the capital of France?", "type": "simple"},
    {"query": "Explain REST API", "type": "medium"},
    {"query": "Write Python binary search", "type": "medium"},
    {"query": "Explain transformer attention", "type": "complex"},
]

# In real mode on CPU, limit to a subset for time
_MAX_REAL_QUERIES = 2


@dataclass
class LayerImportanceResult:
    """Aggregated importance results for a single layer across all queries.

    Attributes:
        layer: Layer index.
        impact_scores: List of impact scores from each query.
        avg_impact_score: Mean impact score across queries.
        classification: ``"high"``, ``"medium"``, or ``"low"``.
        per_query: Detailed per-query comparison results.
    """
    layer: int = -1
    impact_scores: List[float] = field(default_factory=list)
    avg_impact_score: float = 0.0
    classification: str = "low"
    per_query: List[ComparisonResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "avg_impact_score": round(self.avg_impact_score, 4),
            "classification": self.classification,
            "num_queries": len(self.impact_scores),
        }


@dataclass
class AblationStudyResult:
    """Complete results of an ablation study.

    Attributes:
        model_key: Model identifier.
        num_layers: Total transformer layers.
        per_layer: List of :class:`LayerImportanceResult` per layer.
        config: Study configuration.
        total_time_s: Wall-clock time for the study.
        timestamp: ISO-8601 timestamp.
    """
    model_key: str = ""
    num_layers: int = 0
    per_layer: List[LayerImportanceResult] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    total_time_s: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "num_layers": self.num_layers,
            "config": self.config,
            "total_time_s": round(self.total_time_s, 2),
            "timestamp": self.timestamp,
            "per_layer": [l.to_dict() for l in self.per_layer],
        }

    @property
    def high_impact_layers(self) -> List[int]:
        return [l.layer for l in self.per_layer if l.classification == "high"]

    @property
    def medium_impact_layers(self) -> List[int]:
        return [l.layer for l in self.per_layer if l.classification == "medium"]

    @property
    def low_impact_layers(self) -> List[int]:
        return [l.layer for l in self.per_layer if l.classification == "low"]


class LayerAblationEngine:
    """Runs ablations layer-by-layer and collects quality metrics.

    Args:
        bundle: A :class:`ModelBundle` from :func:`model_loader.load_model`.
        max_tokens: Maximum tokens to generate per query.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        max_tokens: int = 32,
        temperature: float = 0.7,
    ) -> None:
        self.bundle = bundle
        self.num_layers = bundle.num_layers
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.engine = RoutedInferenceEngine(
            bundle=bundle,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        logger.info(
            "LayerAblationEngine ready  %s  %d layers  max_tokens=%d",
            bundle.model_name, self.num_layers, max_tokens,
        )

    @staticmethod
    def _all_but_one(num_layers: int, exclude: int) -> Set[int]:
        """Return a set of all layer indices except *exclude*."""
        return {i for i in range(num_layers) if i != exclude}

    def run_ablation_for_query(
        self,
        query: str,
    ) -> Tuple[str, List[ComparisonResult]]:
        """Run baseline + one ablation per layer for a single query.

        Args:
            query: Input prompt.

        Returns:
            ``(baseline_response, list_of_comparison_results)``.
        """
        # Baseline (all layers)
        baseline_response, bl_metrics = self.engine.generate(query)

        results: List[ComparisonResult] = []

        # Ablate one layer at a time
        for layer_idx in range(self.num_layers):
            active = self._all_but_one(self.num_layers, layer_idx)
            t0 = time.perf_counter()
            ablated_response, _ = self.engine.generate(query, active_layers=active)
            elapsed = time.perf_counter() - t0

            comp = compare_responses(
                baseline=baseline_response,
                ablated=ablated_response,
                layer=layer_idx,
                latency_s=elapsed,
                query=query,
            )
            results.append(comp)

            logger.debug(
                "  Layer %2d/%d  J=%.3f  R=%.3f  impact=%.3f  latency=%.2fs",
                layer_idx + 1, self.num_layers,
                comp.jaccard_similarity, comp.rouge_l_f1,
                comp.impact_score, comp.latency_s,
            )

        return baseline_response, results

    def run_study(
        self,
        queries: Optional[List[Dict[str, str]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> AblationStudyResult:
        """Run the full ablation study across all queries.

        Args:
            queries: List of ``{"query": ..., "type": ...}`` dicts.
            config: Optional config dict stored in the result.

        Returns:
            An :class:`AblationStudyResult` with per-layer aggregation.
        """
        if queries is None:
            queries = ABLATION_QUERIES

        t_start = time.perf_counter()
        from datetime import datetime

        # Aggregate impact scores per layer across queries
        layer_scores: Dict[int, List[float]] = {
            i: [] for i in range(self.num_layers)
        }
        layer_details: Dict[int, List[ComparisonResult]] = {
            i: [] for i in range(self.num_layers)
        }

        for i, qd in enumerate(queries):
            query = qd["query"]
            logger.info(
                "Query %d/%d [%s]: %s",
                i + 1, len(queries), qd.get("type", "unknown"), query,
            )
            try:
                _, comps = self.run_ablation_for_query(query)
                for comp in comps:
                    layer_scores[comp.ablated_layer].append(comp.impact_score)
                    layer_details[comp.ablated_layer].append(comp)
                logger.info(
                    "  Done  %d ablations  avg impact=%.3f",
                    len(comps),
                    sum(c.impact_score for c in comps) / max(len(comps), 1),
                )
            except Exception as exc:
                logger.error("  Query failed: %s", exc)
                continue

        total_time = time.perf_counter() - t_start

        per_layer: List[LayerImportanceResult] = []
        for lid in range(self.num_layers):
            scores = layer_scores[lid]
            avg = sum(scores) / max(len(scores), 1)
            per_layer.append(LayerImportanceResult(
                layer=lid,
                impact_scores=scores,
                avg_impact_score=avg,
                classification=classify_importance(avg),
                per_query=layer_details[lid],
            ))

        return AblationStudyResult(
            model_key=self.bundle.model_name,
            num_layers=self.num_layers,
            per_layer=per_layer,
            config=config or {},
            total_time_s=total_time,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    def cleanup(self) -> None:
        self.engine.cleanup()
        logger.info("LayerAblationEngine cleaned up")
