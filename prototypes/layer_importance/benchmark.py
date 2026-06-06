#!/usr/bin/env python3
"""benchmark.py — CLI entry point for the CNOS v0.8 Layer Importance Study.

Usage:
    python prototypes/layer_importance/benchmark.py --mode simulate
    python prototypes/layer_importance/benchmark.py --mode real --max-tokens 8

Examples:
    # Simulate with random responses (no model needed, for testing)
    python bench.py --mode simulate

    # Real ablation on TinyLlama (single query for speed)
    python bench.py --mode real --queries 1 --max-tokens 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import torch

# Ensure real_inference is on sys.path
_PROTO_REAL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "real_inference")
)
if _PROTO_REAL not in sys.path:
    sys.path.insert(0, _PROTO_REAL)

from model_loader import load_model
from layer_ablation import (
    LayerAblationEngine,
    ABLATION_QUERIES,
    AblationStudyResult,
    LayerImportanceResult,
    ComparisonResult,
)
from report_generator import generate_all_reports

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulated engine (no model required)
# ---------------------------------------------------------------------------


class SimulatedAblationEngine:
    """Stub engine that returns random-like responses for testing.

    Uses the real :class:`LayerAblationEngine` interface but produces
    deterministic fake responses so the reporting pipeline can be tested.
    """

    def __init__(
        self,
        bundle=None,
        max_tokens: int = 32,
        temperature: float = 0.7,
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_layers = 22

    @staticmethod
    def _all_but_one(num_layers: int, exclude: int) -> set:
        return {i for i in range(num_layers) if i != exclude}

    def run_study(
        self,
        queries=None,
        config=None,
    ) -> AblationStudyResult:
        if queries is None:
            queries = ABLATION_QUERIES

        t_start = time.perf_counter()

        layer_scores: dict = {i: [] for i in range(self.num_layers)}
        layer_details: dict = {i: [] for i in range(self.num_layers)}

        for qd in queries:
            query = qd["query"]
            # Simulate: deeper layers have slightly higher impact
            # to create a realistic-looking distribution
            for lid in range(self.num_layers):
                # Simulate varying impact
                import math
                fake_impact = 0.05 + 0.40 * (lid / (self.num_layers - 1))
                fake_impact += 0.02 * math.sin(lid * 0.7)
                fake_impact = min(max(fake_impact, 0.0), 1.0)

                layer_scores[lid].append(fake_impact)
                layer_details[lid].append(ComparisonResult(
                    query=query,
                    baseline_response=f"Baseline response for: {query}",
                    ablated_response=f"Ablated (layer {lid}) response for: {query}",
                    ablated_layer=lid,
                    jaccard_similarity=1.0 - fake_impact,
                    rouge_l_f1=1.0 - fake_impact - 0.02,
                    length_ratio=0.9 + 0.1 * (lid / self.num_layers),
                    latency_s=0.5 + 0.3 * (lid / self.num_layers),
                    impact_score=fake_impact,
                ))

        total_time = time.perf_counter() - t_start

        from quality_metrics import classify_importance

        per_layer = [
            LayerImportanceResult(
                layer=lid,
                impact_scores=layer_scores[lid],
                avg_impact_score=sum(layer_scores[lid]) / max(len(layer_scores[lid]), 1),
                classification=classify_importance(
                    sum(layer_scores[lid]) / max(len(layer_scores[lid]), 1)
                ),
                per_query=layer_details[lid],
            )
            for lid in range(self.num_layers)
        ]

        from datetime import datetime
        return AblationStudyResult(
            model_key="TinyLlama/TinyLlama-1.1B-Chat-v1.0 (simulated)",
            num_layers=self.num_layers,
            per_layer=per_layer,
            config=config or {"mode": "simulate"},
            total_time_s=total_time,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CNOS v0.8 Layer Importance Study — benchmark CLI",
    )
    p.add_argument(
        "--mode", choices=("real", "simulate"), default="simulate",
        help="Run with real model or simulated responses (default: simulate)",
    )
    p.add_argument(
        "--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID (default: TinyLlama/...)",
    )
    p.add_argument(
        "--max-tokens", type=int, default=16,
        help="Max tokens to generate per query (default: 16)",
    )
    p.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    p.add_argument(
        "--queries", type=int, default=None,
        help=(
            "Number of benchmark queries to run (default: all). "
            "Use 1 or 2 for quick CPU tests."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: prototypes/layer_importance/output/)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    queries = ABLATION_QUERIES
    if args.queries is not None and args.queries < len(queries):
        queries = queries[: args.queries]

    logger.info(
        "Layer Importance Study  mode=%s  queries=%d  max_tokens=%d",
        args.mode, len(queries), args.max_tokens,
    )

    config = {
        "mode": args.mode,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "num_queries": len(queries),
    }

    if args.mode == "real":
        logger.info("Loading model %s ...", args.model)
        t0 = time.perf_counter()
        bundle = load_model(
            model_key="tinyllama",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        load_time = time.perf_counter() - t0
        logger.info("Model loaded in %.1f s", load_time)

        engine = LayerAblationEngine(
            bundle=bundle,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        engine = SimulatedAblationEngine(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

    try:
        result = engine.run_study(queries=queries, config=config)
    finally:
        engine.cleanup()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "output"
    )
    files = generate_all_reports(result, output_dir=output_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Study complete!")
    logger.info("  High impact layers: %s", result.high_impact_layers)
    logger.info("  Medium impact layers: %s", result.medium_impact_layers)
    logger.info("  Low impact layers: %s", result.low_impact_layers)
    logger.info("  Total time: %.1f s", result.total_time_s)
    logger.info("  Output:")
    for fmt, path in files.items():
        logger.info("    %s: %s", fmt, path)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
