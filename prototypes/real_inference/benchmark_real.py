#!/usr/bin/env python3
"""benchmark_real — compares baseline (all layers) vs. CNOS routed inference on real queries.

Runs a configurable set of test prompts through both full-model and
selective-layer inference, then reports latency, quality, and memory.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Dict, List, Optional, Set

import torch

from layer_hooks import set_active_layers
from model_loader import load_model
from quality_evaluator import QualityEvaluator
from routed_inference import RoutedInferenceEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

TEST_QUERIES: List[Dict[str, str | Set[int]]] = [
    # Simple factual queries → use ~30% of layers
    {
        "query": "What is the capital of France?",
        "plan": {0, 1, 2, 3, 4, 10, 11, 12, 18, 19, 20, 21},
    },
    {
        "query": "How many days are in a week?",
        "plan": {0, 1, 2, 3, 4, 10, 11, 12, 18, 19, 20, 21},
    },
    {
        "query": "What color is the sky?",
        "plan": {0, 1, 2, 3, 4, 10, 11, 12, 18, 19, 20, 21},
    },
    {
        "query": "Who wrote Romeo and Juliet?",
        "plan": {0, 1, 2, 3, 4, 10, 11, 12, 18, 19, 20, 21},
    },
    {
        "query": "What is 2 plus 2?",
        "plan": {0, 1, 2, 3, 4, 10, 11, 12, 18, 19, 20, 21},
    },
    # Medium queries → use ~60% of layers
    {
        "query": "Explain what a variable is in Python.",
        "plan": {0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
    },
    {
        "query": "What is the difference between a list and a tuple?",
        "plan": {0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
    },
    {
        "query": "Write a simple for loop in Python.",
        "plan": {0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
    },
    {
        "query": "How does binary search work?",
        "plan": {0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
    },
    {
        "query": "What is a function in programming?",
        "plan": {0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
    },
    # Complex queries → use all layers
    {
        "query": "Design a simple distributed system for a chat application.",
        "plan": set(range(22)),
    },
    {
        "query": "Explain the trade-offs between SQL and NoSQL databases.",
        "plan": set(range(22)),
    },
    {
        "query": "How would you architect a real-time notification service?",
        "plan": set(range(22)),
    },
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class RealBenchmark:
    """Runs baseline vs. routed inference across multiple queries.

    Args:
        engine: A configured :class:`RoutedInferenceEngine`.
        evaluator: A :class:`QualityEvaluator`.
    """

    def __init__(
        self,
        engine: RoutedInferenceEngine,
        evaluator: Optional[QualityEvaluator] = None,
    ) -> None:
        self.engine = engine
        self.evaluator = evaluator or QualityEvaluator(verbose=False)

    def run_all(self, queries: Optional[List[Dict]] = None) -> None:
        """Run all test queries and print a final aggregate report."""
        queries = queries or TEST_QUERIES
        n = len(queries)
        print(f"\n  Running {n} queries through baseline + routed inference...")
        print(f"  Model: {self.engine.bundle.model_name}  |  "
              f"{self.engine.num_layers} layers  |  "
              f"device={self.engine.device}")

        start = time.perf_counter()

        for i, item in enumerate(queries):
            query = item["query"]
            plan: Set[int] = item["plan"]

            print(f"\n  [{i + 1}/{n}] {query[:60]}...")

            # Run both
            baseline, routed = self.engine.compare(query, plan)
            self.evaluator.evaluate(query, baseline, routed)

        elapsed = time.perf_counter() - start

        print(f"\n  Benchmark completed in {elapsed:.2f}s")

        # Print aggregate
        self.evaluator.print_aggregate()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNOS Real Transformer Benchmark",
    )
    parser.add_argument(
        "--model",
        default="tinyllama",
        choices=["tinyllama", "qwen-1.5b", "llama-3.2-1b"],
        help="Model to benchmark",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per query",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=-1,
        help="Number of queries to run (-1 = all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        stream=sys.stdout,
        format="[%(levelname)s] %(message)s",
    )

    # Load model (this downloads from HuggingFace Hub on first run)
    print(f"\n  Loading {args.model}...")
    print(f"  This will download ~2.2 GB on first run and may take several minutes.")
    print(f"  Subsequent runs will use the cached copy.\n")
    sys.stdout.flush()
    bundle = load_model(args.model)
    print(f"  Model loaded: {bundle.num_layers} layers on {bundle.device}")

    # Build engine
    engine = RoutedInferenceEngine(
        bundle=bundle,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Run benchmark
    bench = RealBenchmark(engine)
    queries = TEST_QUERIES[:args.queries] if args.queries > 0 else TEST_QUERIES
    bench.run_all(queries)

    # Cleanup
    engine.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
