"""quality_evaluator — measures output degradation when using selective layers.

Compares the responses from full-model (baseline) and routed inference
using several automatic metrics:
    - **Exact match** — for short / factual queries.
    - **Token overlap** — Jaccard similarity of decoded token sets.
    - **ROUGE-L** — longest common subsequence F1 score.
    - **Latency reduction** — speedup from layer skipping.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from routed_inference import InferenceMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ROUGE-L helpers (lightweight, no external dependency)
# ---------------------------------------------------------------------------


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Length of the longest common subsequence (dynamic programming)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score between two strings (word-level)."""
    if not reference or not hypothesis:
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Token overlap
# ---------------------------------------------------------------------------


def token_jaccard(reference: str, hypothesis: str) -> float:
    """Jaccard similarity of token sets."""
    ref_set = set(reference.lower().split())
    hyp_set = set(hypothesis.lower().split())
    if not ref_set and not hyp_set:
        return 1.0
    intersection = ref_set & hyp_set
    union = ref_set | hyp_set
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    """Aggregate quality comparison between baseline and routed outputs.

    Attributes:
        query: The original input.
        baseline_response: Full-model output.
        routed_response: Routed-model output.
        exact_match: Whether the responses are identical.
        token_jaccard: Jaccard similarity of token sets.
        rouge_l_f1: ROUGE-L F1 (word-level).
        baseline_metrics: Full inference metrics.
        routed_metrics: Routed inference metrics.
        latency_reduction_pct: Speedup percentage.
    """

    query: str = ""
    baseline_response: str = ""
    routed_response: str = ""
    exact_match: bool = False
    token_jaccard: float = 0.0
    rouge_l_f1: float = 0.0
    baseline_metrics: Optional[InferenceMetrics] = None
    routed_metrics: Optional[InferenceMetrics] = None
    latency_reduction_pct: float = 0.0

    def summary(self) -> Dict[str, object]:
        """Return a dictionary for display or serialisation."""
        return {
            "query": self.query[:80],
            "exact_match": self.exact_match,
            "token_jaccard": round(self.token_jaccard, 4),
            "rouge_l_f1": round(self.rouge_l_f1, 4),
            "latency_reduction_pct": round(self.latency_reduction_pct, 2),
            "baseline_layers": self.baseline_metrics.baseline_layers if self.baseline_metrics else 0,
            "routed_layers": self.routed_metrics.routed_layers if self.routed_metrics else 0,
            "baseline_latency_s": round(self.baseline_metrics.baseline_latency_s, 4) if self.baseline_metrics else 0,
            "routed_latency_s": round(self.routed_metrics.routed_latency_s, 4) if self.routed_metrics else 0,
        }

    def print_report(self) -> None:
        """Print a human-readable quality report."""
        s = self.summary()
        print("\n" + "=" * 60)
        print("  Quality Report — Baseline vs. Routed")
        print("=" * 60)
        print(f"  Query:                 {s['query']}")
        print(f"  Exact match:           {s['exact_match']}")
        print(f"  Token Jaccard:         {s['token_jaccard']:.2%}")
        print(f"  ROUGE-L F1:            {s['rouge_l_f1']:.2%}")
        print(f"  Latency reduction:     {s['latency_reduction_pct']:.1f}%")
        print(f"  Layers:                {s['baseline_layers']} -> {s['routed_layers']}")
        print(f"  Latency:               {s['baseline_latency_s']:.3f}s -> {s['routed_latency_s']:.3f}s")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class QualityEvaluator:
    """Compares baseline and routed inference outputs.

    Args:
        verbose: Print per-query reports during evaluation.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.history: List[QualityReport] = []

    def evaluate(
        self,
        query: str,
        baseline_metrics: InferenceMetrics,
        routed_metrics: InferenceMetrics,
    ) -> QualityReport:
        """Compare two inference runs and produce a :class:`QualityReport`.

        Args:
            query: The original input query.
            baseline_metrics: Metrics from full-model inference.
            routed_metrics: Metrics from routed inference.

        Returns:
            A :class:`QualityReport` with all similarity metrics.
        """
        baseline_resp = baseline_metrics.response
        routed_resp = routed_metrics.response

        # Normalise for comparison
        b_norm = baseline_resp.strip().lower()
        r_norm = routed_resp.strip().lower()

        exact = b_norm == r_norm
        jaccard = token_jaccard(baseline_resp, routed_resp)
        rouge = rouge_l_f1(baseline_resp, routed_resp)

        bl = baseline_metrics.baseline_latency_s
        rl = routed_metrics.routed_latency_s
        latency_reduction = ((bl - rl) / max(bl, 0.001)) * 100.0 if bl > 0 else 0.0

        report = QualityReport(
            query=query,
            baseline_response=baseline_resp,
            routed_response=routed_resp,
            exact_match=exact,
            token_jaccard=jaccard,
            rouge_l_f1=rouge,
            baseline_metrics=baseline_metrics,
            routed_metrics=routed_metrics,
            latency_reduction_pct=round(latency_reduction, 2),
        )

        self.history.append(report)

        if self.verbose:
            report.print_report()

        return report

    def aggregate(self) -> Dict[str, object]:
        """Average metrics across all evaluated queries."""
        if not self.history:
            return {}

        n = len(self.history)
        avg_jaccard = sum(r.token_jaccard for r in self.history) / n
        avg_rouge = sum(r.rouge_l_f1 for r in self.history) / n
        avg_latency_red = sum(r.latency_reduction_pct for r in self.history) / n
        exact_matches = sum(1 for r in self.history if r.exact_match)

        return {
            "num_queries": n,
            "avg_jaccard": round(avg_jaccard, 4),
            "avg_rouge_l": round(avg_rouge, 4),
            "avg_latency_reduction_pct": round(avg_latency_red, 2),
            "exact_match_rate": round(exact_matches / n, 4),
        }

    def print_aggregate(self) -> None:
        """Print aggregate results across all evaluations."""
        agg = self.aggregate()
        if not agg:
            print("No evaluations in history.")
            return
        print("\n" + "=" * 60)
        print("  Aggregate Quality Report")
        print("=" * 60)
        print(f"  Queries evaluated:     {agg['num_queries']}")
        print(f"  Avg token Jaccard:     {agg['avg_jaccard']:.2%}")
        print(f"  Avg ROUGE-L F1:        {agg['avg_rouge_l']:.2%}")
        print(f"  Exact match rate:      {agg['exact_match_rate']:.2%}")
        print(f"  Avg latency reduction: {agg['avg_latency_reduction_pct']:.1f}%")
        print("=" * 60)
