"""Metrics — cumulative statistics for the Dynamic Layer Router.

Tracks per-query and aggregate statistics about layer selection
decisions, compute reduction, and estimated resource savings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from layer_selector import SelectionResult

logger = logging.getLogger(__name__)


@dataclass
class CumulativeMetrics:
    """Aggregate metrics accumulated over multiple routing decisions.

    Attributes:
        total_queries: Number of queries processed.
        total_layers_available: Sum of num_layers across all queries.
        total_layers_selected: Sum of selected layers across all queries.
        total_layers_skipped: Sum of skipped layers.
        compute_reduction_pct: Percentage of layers skipped overall.
        memory_savings_estimate_mb: Estimated MB saved (approximate).
        query_type_counts: Breakdown by complexity class.
        policy_name: Name of the active policy.
        per_type_metrics: Per-class breakdown.
    """

    total_queries: int = 0
    total_layers_available: int = 0
    total_layers_selected: int = 0
    total_layers_skipped: int = 0
    compute_reduction_pct: float = 0.0
    memory_savings_estimate_mb: float = 0.0
    query_type_counts: Dict[str, int] = field(default_factory=lambda: {"simple": 0, "medium": 0, "complex": 0})
    policy_name: str = ""
    per_type_metrics: Dict[str, "TypeMetrics"] = field(default_factory=dict)

    ESTIMATED_MB_PER_LAYER: float = 150.0  # approximate memory per layer

    def update(self, result: SelectionResult) -> None:
        """Record a single selection result into the aggregate."""
        self.total_queries += 1
        self.total_layers_available += result.complexity.complexity_score  # not used directly
        self.total_layers_selected += len(result.selected_layers)
        self.total_layers_skipped += result.num_skipped
        self.policy_name = result.policy_name

        qt = result.complexity.query_type
        self.query_type_counts[qt] = self.query_type_counts.get(qt, 0) + 1

        # Per-type tracking
        if qt not in self.per_type_metrics:
            self.per_type_metrics[qt] = TypeMetrics()
        self.per_type_metrics[qt].update(result)

        # Recompute aggregate percentages
        total = self.total_layers_selected + self.total_layers_skipped
        if total > 0:
            self.compute_reduction_pct = (self.total_layers_skipped / total) * 100.0
            self.memory_savings_estimate_mb = (
                self.total_layers_skipped * self.ESTIMATED_MB_PER_LAYER
            )

    def summary(self) -> Dict[str, object]:
        """Return a dictionary suitable for display or serialisation."""
        return {
            "policy": self.policy_name,
            "total_queries": self.total_queries,
            "total_layers_selected": self.total_layers_selected,
            "total_layers_skipped": self.total_layers_skipped,
            "compute_reduction_pct": round(self.compute_reduction_pct, 2),
            "memory_savings_estimate_mb": round(self.memory_savings_estimate_mb, 1),
            "query_type_distribution": dict(self.query_type_counts),
            "per_type": {
                k: v.summary() for k, v in self.per_type_metrics.items()
            },
            "avg_layers_selected": (
                round(self.total_layers_selected / self.total_queries, 2)
                if self.total_queries > 0 else 0
            ),
            "avg_layers_skipped": (
                round(self.total_layers_skipped / self.total_queries, 2)
                if self.total_queries > 0 else 0
            ),
        }

    def print_report(self) -> None:
        """Print a human-readable summary report to stdout."""
        s = self.summary()
        print("\n" + "=" * 60)
        print("  Dynamic Layer Router — Metrics Report")
        print("=" * 60)
        print(f"  Policy:                  {s['policy']}")
        print(f"  Total queries:           {s['total_queries']}")
        print(f"  Query types:             {s['query_type_distribution']}")
        print(f"  Total layers selected:   {s['total_layers_selected']}")
        print(f"  Total layers skipped:    {s['total_layers_skipped']}")
        print(f"  Avg layers / query:      {s['avg_layers_selected']}")
        print(f"  Avg skipped / query:     {s['avg_layers_skipped']}")
        print(f"  Compute reduction:       {s['compute_reduction_pct']}%")
        print(f"  Est. memory savings:     {s['memory_savings_estimate_mb']:.0f} MB")
        print("-" * 60)

        for qtype, data in s["per_type"].items():
            print(f"  [{qtype}]  avg={data['avg_layers']} layers  "
                  f"skipped={data['avg_skipped']}  "
                  f"reduction={data['reduction_pct']}%")
        print("=" * 60)


@dataclass
class TypeMetrics:
    """Per-type breakdown."""

    count: int = 0
    total_selected: int = 0
    total_skipped: int = 0

    def update(self, result: SelectionResult) -> None:
        self.count += 1
        self.total_selected += len(result.selected_layers)
        self.total_skipped += result.num_skipped

    def summary(self) -> Dict[str, object]:
        avg_sel = round(self.total_selected / max(self.count, 1), 2)
        avg_skip = round(self.total_skipped / max(self.count, 1), 2)
        total = self.total_selected + self.total_skipped
        reduction = round((self.total_skipped / total) * 100.0, 2) if total > 0 else 0.0
        return {
            "count": self.count,
            "avg_layers": avg_sel,
            "avg_skipped": avg_skip,
            "reduction_pct": reduction,
        }
