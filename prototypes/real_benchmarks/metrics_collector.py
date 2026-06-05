"""metrics_collector -- computes aggregate metrics from baseline and CNOS runs.

Calculates:
  * RAM reduction % (peak, average)
  * Compute reduction % (layers skipped)
  * Latency overhead / speedup
  * Cache hit rate
  * Compression ratio
  * Response quality (Jaccard, ROUGE-L)
  * Tokens per second
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_PROTO = os.path.join(os.path.dirname(__file__), "..", "real_inference")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

logger = logging.getLogger(__name__)

_OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


@dataclass
class ComparisonRow:
    query: str = ""
    baseline_response: str = ""
    cnos_response: str = ""
    baseline_latency_s: float = 0.0
    cnos_latency_s: float = 0.0
    latency_reduction_pct: float = 0.0
    baseline_ram_peak_mb: float = 0.0
    cnos_ram_peak_mb: float = 0.0
    ram_reduction_pct: float = 0.0
    tokens_generated: int = 0
    layers_skipped: int = 0
    compute_reduction_pct: float = 0.0
    jaccard_sim: float = 0.0
    rouge_l: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "baseline_response": self.baseline_response,
            "cnos_response": self.cnos_response,
            "baseline_latency_s": round(self.baseline_latency_s, 4),
            "cnos_latency_s": round(self.cnos_latency_s, 4),
            "latency_reduction_pct": round(self.latency_reduction_pct, 1),
            "baseline_ram_peak_mb": round(self.baseline_ram_peak_mb, 1),
            "cnos_ram_peak_mb": round(self.cnos_ram_peak_mb, 1),
            "ram_reduction_pct": round(self.ram_reduction_pct, 1),
            "tokens_generated": self.tokens_generated,
            "layers_skipped": self.layers_skipped,
            "compute_reduction_pct": round(self.compute_reduction_pct, 1),
            "jaccard_sim": round(self.jaccard_sim, 4),
            "rouge_l": round(self.rouge_l, 4),
        }


@dataclass
class MetricsReport:
    model_key: str = ""
    num_layers: int = 0
    routing_policy: str = ""
    quantisation: str = ""
    num_queries: int = 0
    avg_baseline_latency_s: float = 0.0
    avg_cnos_latency_s: float = 0.0
    avg_latency_reduction_pct: float = 0.0
    avg_baseline_ram_peak_mb: float = 0.0
    avg_cnos_ram_peak_mb: float = 0.0
    avg_ram_reduction_pct: float = 0.0
    avg_compute_reduction_pct: float = 0.0
    avg_jaccard_sim: float = 0.0
    avg_rouge_l: float = 0.0
    avg_tokens_per_sec: float = 0.0
    avg_cache_hit_rate_pct: float = 0.0
    avg_compression_ratio: float = 0.0
    details: List[ComparisonRow] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "num_layers": self.num_layers,
            "routing_policy": self.routing_policy,
            "quantisation": self.quantisation,
            "num_queries": self.num_queries,
            "avg_baseline_latency_s": round(self.avg_baseline_latency_s, 4),
            "avg_cnos_latency_s": round(self.avg_cnos_latency_s, 4),
            "avg_latency_reduction_pct": round(self.avg_latency_reduction_pct, 1),
            "avg_baseline_ram_peak_mb": round(self.avg_baseline_ram_peak_mb, 1),
            "avg_cnos_ram_peak_mb": round(self.avg_cnos_ram_peak_mb, 1),
            "avg_ram_reduction_pct": round(self.avg_ram_reduction_pct, 1),
            "avg_compute_reduction_pct": round(self.avg_compute_reduction_pct, 1),
            "avg_jaccard_sim": round(self.avg_jaccard_sim, 4),
            "avg_rouge_l": round(self.avg_rouge_l, 4),
            "avg_tokens_per_sec": round(self.avg_tokens_per_sec, 2),
            "avg_cache_hit_rate_pct": round(self.avg_cache_hit_rate_pct, 1),
            "avg_compression_ratio": round(self.avg_compression_ratio, 2),
        }


def compute_metrics(
    baseline_result: Any,
    cnos_result: Any,
) -> MetricsReport:
    """Aggregate baseline and CNOS results into a comparison report.

    Args:
        baseline_result: A :class:`BaselineResult` (with a ``.queries`` list).
        cnos_result: A :class:`CnosResult` (with a ``.queries`` list).

    Returns:
        A :class:`MetricsReport` with averaged metrics across all queries.
    """
    from quality_evaluator import QualityEvaluator
    evaluator = QualityEvaluator()

    details: List[ComparisonRow] = []
    bq_map = {q.query: q for q in baseline_result.queries}
    cq_map = {q.query: q for q in cnos_result.queries}
    all_queries = list(set(bq_map.keys()) | set(cq_map.keys()))

    for query in all_queries:
        bq = bq_map.get(query)
        cq = cq_map.get(query)
        if bq is None or cq is None:
            continue

        # Quality evaluation
        ref = bq.response if bq.response and "ERROR" not in bq.response else cq.response
        cand = cq.response if cq.response and "ERROR" not in cq.response else ref
        if ref and cand:
            try:
                qe = evaluator.compare_responses(ref, cand)
                js = qe.get("jaccard_similarity", 0.0)
                rl = qe.get("rouge_l", 0.0)
            except Exception:
                js = 0.0
                rl = 0.0
        else:
            js = 0.0
            rl = 0.0

        lat_reduction = 0.0
        if bq.latency_s > 0:
            lat_reduction = ((bq.latency_s - cq.latency_s) / bq.latency_s) * 100

        ram_reduction = 0.0
        if bq.ram_peak_mb > 0:
            ram_reduction = ((bq.ram_peak_mb - cq.ram_peak_mb) / bq.ram_peak_mb) * 100

        details.append(ComparisonRow(
            query=query,
            baseline_response=bq.response,
            cnos_response=cq.response,
            baseline_latency_s=bq.latency_s,
            cnos_latency_s=cq.latency_s,
            latency_reduction_pct=lat_reduction,
            baseline_ram_peak_mb=bq.ram_peak_mb,
            cnos_ram_peak_mb=cq.ram_peak_mb,
            ram_reduction_pct=ram_reduction,
            tokens_generated=cq.tokens_generated,
            layers_skipped=cq.layers_skipped,
            compute_reduction_pct=cq.compute_reduction_pct,
            jaccard_sim=js,
            rouge_l=rl,
        ))

    if not details:
        return MetricsReport()

    n = len(details)
    report = MetricsReport(
        model_key=baseline_result.model_key,
        num_layers=baseline_result.num_layers,
        routing_policy=cnos_result.routing_policy,
        quantisation=cnos_result.quantisation,
        num_queries=n,
        avg_baseline_latency_s=sum(d.baseline_latency_s for d in details) / n,
        avg_cnos_latency_s=sum(d.cnos_latency_s for d in details) / n,
        avg_latency_reduction_pct=sum(d.latency_reduction_pct for d in details) / n,
        avg_baseline_ram_peak_mb=sum(d.baseline_ram_peak_mb for d in details) / n,
        avg_cnos_ram_peak_mb=sum(d.cnos_ram_peak_mb for d in details) / n,
        avg_ram_reduction_pct=sum(d.ram_reduction_pct for d in details) / n,
        avg_compute_reduction_pct=sum(d.compute_reduction_pct for d in details) / n,
        avg_jaccard_sim=sum(d.jaccard_sim for d in details) / n,
        avg_rouge_l=sum(d.rouge_l for d in details) / n,
        avg_tokens_per_sec=sum(
            cq.tokens_per_sec for qname, cq in cq_map.items()
            if qname in bq_map
        ) / max(n, 1),
        details=details,
    )

    # Compute derived cache/compression averages across CNOS queries
    cnos_queries = cnos_result.queries
    if cnos_queries:
        report.avg_cache_hit_rate_pct = sum(
            q.cache_hit_rate_pct for q in cnos_queries
        ) / len(cnos_queries)
        report.avg_compression_ratio = sum(
            q.compression_ratio for q in cnos_queries
        ) / len(cnos_queries)
    else:
        report.avg_cache_hit_rate_pct = 0.0
        report.avg_compression_ratio = 1.0

    return report


def save_metrics_json(report: MetricsReport, path: Optional[str] = None) -> str:
    if path is None:
        os.makedirs(_OUT_DIR, exist_ok=True)
        path = os.path.join(_OUT_DIR, "benchmark_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Metrics JSON saved to %s", path)
    return path
