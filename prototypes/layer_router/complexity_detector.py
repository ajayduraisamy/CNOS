"""ComplexityDetector — classifies query difficulty for adaptive routing.

Analyses a user query by examining lexical features, structural cues,
and keyword signals to estimate the reasoning depth required.
Outputs a normalised complexity score together with a categorical
label and a human-readable depth estimate.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


@dataclass
class ComplexityResult:
    """Result of analysing a single query.

    Attributes:
        query: The original input text.
        complexity_score: Float in [0.0, 1.0] — 0 is trivial, 1 is extremely hard.
        query_type: One of ``"simple"``, ``"medium"``, ``"complex"``.
        reasoning_depth: Human-readable depth label.
        confidence: How confident the detector is in its classification (0.0–1.0).
        features: Raw feature values used for the computation.
    """

    query: str
    complexity_score: float
    query_type: str
    reasoning_depth: str
    confidence: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ComplexityDetector
# ---------------------------------------------------------------------------


class ComplexityDetector:
    """Rule-based query complexity analyser.

    Heuristics used:
        * **Length score** — longer queries tend to be more complex.
        * **Keyword score** — domain-specific terms signal technical depth.
        * **Structural score** — lists, multi-sentence structure, code blocks.
        * **Instruction score** — verbs like *design*, *architect*, *analyse*.

    Args:
        num_layers: Total transformer layers (used to map score to depth).
    """

    # Keywords that raise complexity, grouped by severity
    HIGH_SIGNAL_KEYWORDS: List[str] = [
        "distributed", "microservice", "architecture", "design pattern",
        "kubernetes", "infrastructure", "scalability", "fault tolerant",
        "consensus", "blockchain", "cryptographic", "compiler",
        "optimisation", "concurrent", "parallel", "transactional",
        "neural", "transformer", "reinforcement", "bayesian",
        "orchestrat", "replicat", "sharding", "partitioning",
        "multi tenant", "multi region", "multi cloud",
        "load balanc", "auto scal", "failover", "disaster recovery",
        "high frequency", "nanosecond", "real time", "streaming",
        "federated", "differential privacy", "gradient compression",
    ]

    MEDIUM_SIGNAL_KEYWORDS: List[str] = [
        "implement", "algorithm", "function", "class", "recursive",
        "database", "api", "endpoint", "middleware", "protocol",
        "pipeline", "workflow", "deployment", "monitoring",
        "asynchronous", "callback", "multithreading", "semaphore",
        "polymorphism", "inheritance", "dependency",
        "binary", "search", "sort", "merge", "stack", "queue",
        "graph", "tree", "string", "array", "matrix", "prime",
        "fibonacci", "palindrome", "permutation", "traverse",
        "complexity", "big o", "runtime", "iterate",
        "decorator", "generator", "context manager", "fixture",
        "serialisation", "middleware", "pagination",
    ]

    COMPLEX_INSTRUCTIONS: List[str] = [
        "design", "architect", "analyse", "analyze", "evaluate",
        "compare", "contrast", "synthesise", "synthesize",
        "deconstruct", "optimise", "optimize", "refactor",
        "architect", "orchestrate", "integrate",
        "write", "build", "create", "implement", "develop",
        "explain", "describe", "derive", "demonstrate",
    ]

    # Regex for code blocks, bullet lists, numbered steps
    STRUCTURAL_PATTERNS: List[re.Pattern] = [
        re.compile(r"```"),                          # code fence
        re.compile(r"(?m)^[\s]*[*-]\s"),             # bullet list
        re.compile(r"(?m)^[\s]*\d+\.\s"),            # numbered list
        re.compile(r"(?m)^[\s]*\|.*\|"),             # table
        re.compile(r"(?i)\b(?:step|phase|stage)\b"), # structured steps
    ]

    TYPE_THRESHOLDS: List[tuple[str, float]] = [
        ("complex", 0.40),
        ("medium", 0.20),
        ("simple", 0.00),
    ]

    DEPTH_LABELS: List[tuple[float, str]] = [
        (0.70, "deep-reasoning"),
        (0.35, "moderate-reasoning"),
        (0.12, "shallow-reasoning"),
        (0.00, "factual-retrieval"),
    ]

    def __init__(self, num_layers: int = 80) -> None:
        self.num_layers = num_layers
        logger.info("ComplexityDetector initialised (%d-layer model)", num_layers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, query: str) -> ComplexityResult:
        """Classify a user query and return a structured result.

        Args:
            query: The raw user input.

        Returns:
            A :class:`ComplexityResult` with score, type, and reasoning depth.
        """
        if not query or not query.strip():
            return ComplexityResult(
                query=query,
                complexity_score=0.0,
                query_type="simple",
                reasoning_depth="factual-retrieval",
                confidence=1.0,
                features={},
            )

        features = self._extract_features(query)
        score = self._compute_score(features)
        query_type = self._classify(score)
        depth = self._depth_label(score)
        confidence = self._estimate_confidence(features, score)

        logger.debug(
            "Query [%s] -> score=%.3f type=%s depth=%s confidence=%.2f",
            query[:60].replace("\n", " "),
            score, query_type, depth, confidence,
        )

        return ComplexityResult(
            query=query,
            complexity_score=round(score, 4),
            query_type=query_type,
            reasoning_depth=depth,
            confidence=round(confidence, 4),
            features=features,
        )

    # ------------------------------------------------------------------
    # Feature extraction & scoring
    # ------------------------------------------------------------------

    def _extract_features(self, query: str) -> Dict[str, float]:
        """Compute raw numeric features from the query text."""
        q = query.strip()
        words = q.split()
        sentences = re.split(r"[.!?]+", q)
        sentences = [s for s in sentences if s.strip()]

        # Length-based
        word_count = len(words)
        char_count = len(q)
        avg_word_len = char_count / max(word_count, 1)

        # Normalise for matching: collapse hyphens/dashes to spaces
        q_normalised = q.lower().replace("-", " ").replace("–", " ").replace("—", " ")

        # Keyword matches
        high_kw = sum(1 for kw in self.HIGH_SIGNAL_KEYWORDS if kw.lower() in q_normalised)
        med_kw = sum(1 for kw in self.MEDIUM_SIGNAL_KEYWORDS if kw.lower() in q_normalised)
        instr_kw = sum(1 for v in self.COMPLEX_INSTRUCTIONS if v.lower() in q_normalised)

        # Structure
        struct_hits = sum(1 for pat in self.STRUCTURAL_PATTERNS if pat.search(q))

        # Sentence count
        num_sentences = len(sentences)

        # Code block presence
        has_code = 1.0 if "```" in q or re.search(r"(?m)^(?:def |class |import |from \w+ import)", q) else 0.0

        # Numeric figures
        num_numbers = len(re.findall(r"\b\d+\b", q))

        return {
            "word_count": float(word_count),
            "char_count": float(char_count),
            "avg_word_len": round(avg_word_len, 2),
            "num_sentences": float(num_sentences),
            "high_signal_keywords": float(high_kw),
            "medium_signal_keywords": float(med_kw),
            "instruction_keywords": float(instr_kw),
            "structural_hits": float(struct_hits),
            "has_code_block": has_code,
            "num_numbers": float(num_numbers),
        }

    def _compute_score(self, f: Dict[str, float]) -> float:
        """Combine features into a single normalised complexity score."""
        # Normalise word count (sigmoid: 0 → 0, 40+ → ~0.9)
        length_score = 1.0 / (1.0 + math.exp(-0.12 * (f["word_count"] - 10)))

        # Keyword density (diminishing returns)
        high_density = 1.0 - math.exp(-0.5 * f["high_signal_keywords"])
        med_density = 1.0 - math.exp(-0.4 * f["medium_signal_keywords"])
        instr_density = 1.0 - math.exp(-0.6 * f["instruction_keywords"])

        # Structure score
        struct_score = min(f["structural_hits"] / 4.0, 1.0)

        # Code penalty
        code_score = f["has_code_block"] * 0.20

        # Weighted combination
        score = (
            0.15 * length_score
            + 0.30 * high_density
            + 0.20 * med_density
            + 0.22 * instr_density
            + 0.08 * struct_score
            + 0.05 * code_score
        )

        return min(max(score, 0.0), 1.0)

    def _classify(self, score: float) -> str:
        """Map a continuous score to a discrete type label."""
        for label, threshold in self.TYPE_THRESHOLDS:
            if score >= threshold:
                return label
        return "simple"

    def _depth_label(self, score: float) -> str:
        """Map score to a human-readable reasoning depth."""
        for threshold, label in self.DEPTH_LABELS:
            if score >= threshold:
                return label
        return "factual-retrieval"

    @staticmethod
    def _estimate_confidence(features: Dict[str, float], score: float) -> float:
        """Estimate classification confidence based on feature richness."""
        # More signal → higher confidence; ambiguity near thresholds lowers it
        signal_count = (
            features["high_signal_keywords"]
            + features["medium_signal_keywords"]
            + features["instruction_keywords"]
            + features["structural_hits"]
        )
        raw_conf = min(signal_count / 5.0, 1.0) * 0.6 + 0.2

        # Reduce confidence near classification boundaries
        boundary_penalty = 0.0
        for _, thresh in [("complex", 0.60), ("medium", 0.30)]:
            dist = abs(score - thresh)
            if dist < 0.08:
                boundary_penalty = max(boundary_penalty, (0.08 - dist) / 0.08 * 0.15)
        raw_conf -= boundary_penalty

        return min(max(raw_conf, 0.1), 1.0)
