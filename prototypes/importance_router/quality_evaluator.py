"""quality_evaluator — evaluates the quality of importance-routed responses.

Metrics:
  * Jaccard similarity (token-set overlap with baseline).
  * ROUGE-L F1 (longest common subsequence).
  * Composite quality score.
  * Similarity score.
  * Quality loss.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality comparison between baseline and routed responses.

    Attributes:
        query: Input query.
        baseline_response: Full-model response.
        routed_response: Routed (layer-skipped) response.
        mode: Routing mode used.
        jaccard_similarity: Token-set overlap (0-1).
        rouge_l_f1: ROUGE-L F1 score (0-1).
        quality_score: Composite quality preserved (0-1, higher = better).
        similarity_score: Synonym for Jaccard for clarity.
        quality_loss: 1 - quality_score (0-1, higher = worse).
        latency_s: Inference latency for the routed run.
        num_layers_skipped: Number of layers skipped.
        compute_reduction_pct: Percentage of layers skipped.
    """
    query: str = ""
    baseline_response: str = ""
    routed_response: str = ""
    mode: str = ""
    jaccard_similarity: float = 0.0
    rouge_l_f1: float = 0.0
    quality_score: float = 0.0
    similarity_score: float = 0.0
    quality_loss: float = 0.0
    latency_s: float = 0.0
    num_layers_skipped: int = 0
    compute_reduction_pct: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "jaccard_similarity": round(self.jaccard_similarity, 4),
            "rouge_l_f1": round(self.rouge_l_f1, 4),
            "quality_score": round(self.quality_score, 4),
            "similarity_score": round(self.similarity_score, 4),
            "quality_loss": round(self.quality_loss, 4),
            "latency_s": round(self.latency_s, 4),
            "num_layers_skipped": self.num_layers_skipped,
            "compute_reduction_pct": round(self.compute_reduction_pct, 2),
        }


def _tokenize(text: str) -> set:
    return set(re.findall(r'\S+', text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    union = tokens_a | tokens_b
    if not union:
        return 1.0
    return len(tokens_a & tokens_b) / len(union)


def _lcs_length(x: list, y: list) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_f1(a: str, b: str) -> float:
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


def evaluate(
    baseline: str,
    routed: str,
    query: str = "",
    mode: str = "",
    latency_s: float = 0.0,
    num_layers_skipped: int = 0,
    compute_reduction_pct: float = 0.0,
) -> QualityMetrics:
    """Compare a baseline response to a routed (layer-skipped) response.

    Returns:
        A :class:`QualityMetrics` with all scores computed.
    """
    js = jaccard_similarity(baseline, routed)
    rl = rouge_l_f1(baseline, routed)

    # Quality score: composite of Jaccard + ROUGE-L, higher = better
    quality_score = 0.5 * js + 0.5 * rl
    quality_loss = 1.0 - quality_score

    return QualityMetrics(
        query=query,
        baseline_response=baseline,
        routed_response=routed,
        mode=mode,
        jaccard_similarity=js,
        rouge_l_f1=rl,
        quality_score=quality_score,
        similarity_score=js,
        quality_loss=quality_loss,
        latency_s=latency_s,
        num_layers_skipped=num_layers_skipped,
        compute_reduction_pct=compute_reduction_pct,
    )
