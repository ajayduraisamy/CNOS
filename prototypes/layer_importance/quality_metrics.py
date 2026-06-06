"""quality_metrics — compares two generated responses and computes similarity metrics.

Metrics:
  * Token overlap (Jaccard similarity on tokenized output)
  * Response length difference
  * ROUGE-L (longest common subsequence based)
  * Composite Layer Impact Score (0–1, higher = more important layer)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Quality comparison between a baseline and an ablated response.

    Attributes:
        query: The input query.
        baseline_response: Full-model response text.
        ablated_response: Response with one layer disabled.
        ablated_layer: The layer index that was disabled.
        jaccard_similarity: Token-set overlap (0–1).
        rouge_l_f1: ROUGE-L F1 score (0–1).
        length_ratio: ``len(ablated) / len(baseline)`` (capped at 2.0).
        latency_s: Inference latency for the ablated run.
        impact_score: Composite importance score (0–1).
    """
    query: str = ""
    baseline_response: str = ""
    ablated_response: str = ""
    ablated_layer: int = -1
    jaccard_similarity: float = 0.0
    rouge_l_f1: float = 0.0
    length_ratio: float = 1.0
    latency_s: float = 0.0
    impact_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "layer": self.ablated_layer,
            "query": self.query,
            "jaccard_similarity": round(self.jaccard_similarity, 4),
            "rouge_l_f1": round(self.rouge_l_f1, 4),
            "length_ratio": round(self.length_ratio, 4),
            "latency_s": round(self.latency_s, 4),
            "impact_score": round(self.impact_score, 4),
        }


def _tokenize(text: str) -> Set[str]:
    """Split text into whitespace-delimited tokens, lowercased."""
    return set(re.findall(r'\S+', text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity on token sets (0 = disjoint, 1 = identical)."""
    if not a and not b:
        return 1.0
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    union = tokens_a | tokens_b
    if not union:
        return 1.0
    return len(tokens_a & tokens_b) / len(union)


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Length of the longest common subsequence (dynamic programming)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_f1(a: str, b: str) -> float:
    """ROUGE-L F1 score on token sequences (0–1)."""
    x = _tokenize(a)
    y = _tokenize(b)
    if not x and not y:
        return 1.0
    lcs = _lcs_length(sorted(x), sorted(y))
    precision = lcs / max(len(x), 1)
    recall = lcs / max(len(y), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compare_responses(
    baseline: str,
    ablated: str,
    layer: int,
    latency_s: float = 0.0,
    query: str = "",
) -> ComparisonResult:
    """Compare a baseline response to an ablated response.

    Args:
        baseline: Response from full model.
        ablated: Response with one layer disabled.
        layer: The disabled layer index.
        latency_s: Inference latency for the ablated run.
        query: Original input query.

    Returns:
        A :class:`ComparisonResult` with all metrics and the composite impact score.
    """
    js = jaccard_similarity(baseline, ablated)
    rl = rouge_l_f1(baseline, ablated)

    blen = max(len(baseline.split()), 1)
    alen = max(len(ablated.split()), 1)
    lr = min(alen / blen, 2.0)

    # Composite impact score: 1 - similarity, weighted across metrics.
    # Quality loss = how much the response diverged from baseline.
    quality_loss = 1.0 - (0.5 * js + 0.5 * rl)
    impact_score = min(quality_loss, 1.0)

    return ComparisonResult(
        query=query,
        baseline_response=baseline,
        ablated_response=ablated,
        ablated_layer=layer,
        jaccard_similarity=js,
        rouge_l_f1=rl,
        length_ratio=lr,
        latency_s=latency_s,
        impact_score=impact_score,
    )


def classify_importance(impact_score: float) -> str:
    """Classify a layer's importance based on its impact score.

    Args:
        impact_score: Value in [0, 1].

    Returns:
        ``"high"``, ``"medium"``, or ``"low"``.
    """
    if impact_score >= 0.30:
        return "high"
    elif impact_score >= 0.10:
        return "medium"
    return "low"
