#!/usr/bin/env python3
"""benchmark.py — CLI entry point for the CNOS v0.8.1 Importance Router.

Runs baseline (all layers) and importance-routed inference across all
three routing modes, then generates a comparative report.

Usage:
    python prototypes/importance_router/benchmark.py --mode simulate
    python prototypes/importance_router/benchmark.py --mode real --max-tokens 16

Examples:
    # Simulated (no model)
    python bench.py --mode simulate

    # Real, full benchmark
    python bench.py --mode real --max-tokens 16 --queries 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure local imports take priority over real_inference (name collision: quality_evaluator.py)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_PROTO_REAL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "real_inference")
)
if _PROTO_REAL not in sys.path:
    # Insert after _THIS_DIR so local modules resolve first
    sys.path.insert(1, _PROTO_REAL)

from model_loader import load_model
from routed_inference import RoutedInferenceEngine

from layer_profile import LayerProfile
from importance_router import ImportanceRouter, RoutingMode, RoutingDecision
from quality_evaluator import evaluate, QualityMetrics

logger = logging.getLogger(__name__)

BENCHMARK_QUERIES: List[Dict[str, str]] = [
    {"query": "What is 2+2?", "type": "simple"},
    {"query": "What is the capital of France?", "type": "simple"},
    {"query": "Explain REST API", "type": "medium"},
    {"query": "Write Python binary search", "type": "medium"},
    {"query": "Explain transformer attention", "type": "complex"},
]


@dataclass
class BenchmarkResult:
    """Aggregated result for a single routing mode across all queries."""
    mode: str = ""
    all_metrics: List[QualityMetrics] = field(default_factory=list)
    total_time_s: float = 0.0

    @property
    def avg_quality_score(self) -> float:
        if not self.all_metrics:
            return 0.0
        return sum(m.quality_score for m in self.all_metrics) / len(self.all_metrics)

    @property
    def avg_jaccard(self) -> float:
        if not self.all_metrics:
            return 0.0
        return sum(m.jaccard_similarity for m in self.all_metrics) / len(self.all_metrics)

    @property
    def avg_rouge_l(self) -> float:
        if not self.all_metrics:
            return 0.0
        return sum(m.rouge_l_f1 for m in self.all_metrics) / len(self.all_metrics)

    @property
    def avg_latency_s(self) -> float:
        if not self.all_metrics:
            return 0.0
        return sum(m.latency_s for m in self.all_metrics) / len(self.all_metrics)

    @property
    def avg_layers_skipped(self) -> float:
        if not self.all_metrics:
            return 0.0
        return sum(m.num_layers_skipped for m in self.all_metrics) / len(self.all_metrics)

    @property
    def avg_reduction_pct(self) -> float:
        if not self.all_metrics:
            return 0.0
        return sum(m.compute_reduction_pct for m in self.all_metrics) / len(self.all_metrics)


@dataclass
class BenchmarkSuiteResult:
    """Complete benchmark results across all modes."""
    baseline_latency_avg: float = 0.0
    modes: Dict[str, BenchmarkResult] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    total_time_s: float = 0.0
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Simulated engine
# ---------------------------------------------------------------------------


class SimulatedBenchmark:
    """Stub that uses the real routers but with fake inference."""

    def __init__(self, max_tokens: int = 16, temperature: float = 0.7) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_layers = 22

    def run(self, queries: List[Dict[str, str]]) -> BenchmarkSuiteResult:
        t_start = time.perf_counter()
        profile = LayerProfile()
        router = ImportanceRouter(profile)

        modes = [RoutingMode.CONSERVATIVE, RoutingMode.BALANCED, RoutingMode.AGGRESSIVE]
        decisions = {m: router.decide(m) for m in modes}

        # Simulate responses
        baseline_latencies = []
        mode_results: Dict[str, BenchmarkResult] = {}

        for mode in modes:
            decision = decisions[mode]
            all_qm: List[QualityMetrics] = []
            for qd in queries:
                query = qd["query"]
                baseline_text = f"Baseline response for: {query}"
                routed_text = f"Routed ({mode.value}) response for: {query}"
                latency = 0.5 + 0.2 * (mode.value == "aggressive")
                qm = evaluate(
                    baseline=baseline_text,
                    routed=routed_text,
                    query=query,
                    mode=mode.value,
                    latency_s=latency,
                    num_layers_skipped=decision.num_skipped,
                    compute_reduction_pct=decision.compute_reduction_pct,
                )
                all_qm.append(qm)
                baseline_latencies.append(0.8)
            mode_results[mode.value] = BenchmarkResult(
                mode=mode.value, all_metrics=all_qm,
            )

        return BenchmarkSuiteResult(
            baseline_latency_avg=sum(baseline_latencies) / max(len(baseline_latencies), 1),
            modes=mode_results,
            config={"mode": "simulate", "max_tokens": self.max_tokens},
            total_time_s=time.perf_counter() - t_start,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )


# ---------------------------------------------------------------------------
# Real benchmark engine
# ---------------------------------------------------------------------------


class RealBenchmark:
    """Runs real inference with the importance router on TinyLlama."""

    def __init__(
        self, bundle, max_tokens: int = 16, temperature: float = 0.7,
    ) -> None:
        self.engine = RoutedInferenceEngine(
            bundle=bundle,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        self.num_layers = bundle.num_layers
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.profile = LayerProfile()
        self.router = ImportanceRouter(self.profile)

    def run(self, queries: List[Dict[str, str]]) -> BenchmarkSuiteResult:
        t_start = time.perf_counter()

        modes = [RoutingMode.CONSERVATIVE, RoutingMode.BALANCED, RoutingMode.AGGRESSIVE]
        decisions = {m: self.router.decide(m) for m in modes}

        baseline_latencies: List[float] = []
        mode_results: Dict[str, BenchmarkResult] = {}

        # Step 1: run baseline once per query (shared across all modes)
        baselines: List[tuple[str, float]] = []
        for i, qd in enumerate(queries):
            query = qd["query"]
            logger.info(
                "Baseline %d/%d [%s]: %s",
                i + 1, len(queries), qd.get("type", ""), query,
            )
            t0 = time.perf_counter()
            resp, _ = self.engine.generate(query)
            elapsed = time.perf_counter() - t0
            baselines.append((resp, elapsed))
            baseline_latencies.append(elapsed)

        # Step 2: for each mode, run routed inference and compare to baseline
        for mode in modes:
            decision = decisions[mode]
            logger.info(
                "Mode %s  skip=%s  reduction=%.1f%%",
                mode.value, sorted(decision.skip_layers),
                decision.compute_reduction_pct,
            )
            all_qm: List[QualityMetrics] = []
            for i, qd in enumerate(queries):
                query = qd["query"]
                baseline_resp, _ = baselines[i]

                t0 = time.perf_counter()
                routed_resp, _ = self.engine.generate(
                    query, active_layers=decision.active_layers,
                )
                routed_latency = time.perf_counter() - t0

                qm = evaluate(
                    baseline=baseline_resp,
                    routed=routed_resp,
                    query=query,
                    mode=mode.value,
                    latency_s=routed_latency,
                    num_layers_skipped=decision.num_skipped,
                    compute_reduction_pct=decision.compute_reduction_pct,
                )
                all_qm.append(qm)
                logger.info(
                    "    quality=%.4f  jaccard=%.4f  latency=%.2fs  skip=%d",
                    qm.quality_score, qm.jaccard_similarity,
                    qm.latency_s, qm.num_layers_skipped,
                )
            mode_results[mode.value] = BenchmarkResult(
                mode=mode.value, all_metrics=all_qm,
            )

        bl_avg = sum(baseline_latencies) / max(len(baseline_latencies), 1)
        return BenchmarkSuiteResult(
            baseline_latency_avg=bl_avg,
            modes=mode_results,
            config={
                "mode": "real",
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "num_queries": len(queries),
            },
            total_time_s=time.perf_counter() - t_start,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    def cleanup(self) -> None:
        self.engine.cleanup()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(result: BenchmarkSuiteResult, output_dir: str) -> str:
    """Write the importance_router_report.md and JSON.

    Returns the path to the Markdown report.
    """
    os.makedirs(output_dir, exist_ok=True)

    md_path = os.path.join(output_dir, "importance_router_report.md")
    json_path = os.path.join(output_dir, "importance_router_results.json")

    # Build markdown
    lines = [
        "# Importance-Based Layer Router — Benchmark Report",
        "",
        f"**Date:** {result.timestamp}",
        f"**Total time:** {result.total_time_s:.1f} s",
        f"**Baseline avg latency:** {result.baseline_latency_avg:.4f} s",
        "",
        "---",
        "",
        "## Configuration",
        "",
        f"```json",
        json.dumps(result.config, indent=2),
        f"```",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Mode | Quality Score | Similarity | ROUGE-L | Latency (s) | Layers Skipped | Compute Reduction |",
        "|------|--------------|------------|---------|-------------|----------------|-------------------|",
    ]

    for mode_name in ("conservative", "balanced", "aggressive"):
        mr = result.modes.get(mode_name)
        if not mr:
            continue
        lines.append(
            f"| {mode_name} | {mr.avg_quality_score:.4f} | "
            f"{mr.avg_jaccard:.4f} | {mr.avg_rouge_l:.4f} | "
            f"{mr.avg_latency_s:.4f} | {mr.avg_layers_skipped:.1f} | "
            f"{mr.avg_reduction_pct:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Per-Query Details",
        "",
    ])

    for mode_name in ("conservative", "balanced", "aggressive"):
        mr = result.modes.get(mode_name)
        if not mr:
            continue
        lines.append(f"### Mode: {mode_name}")
        lines.append("")
        lines.append(
            "| Query | Quality Score | Jaccard | ROUGE-L | Latency (s) | Skipped | Reduction |"
        )
        lines.append(
            "|-------|--------------|---------|---------|-------------|---------|-----------|"
        )
        for qm in mr.all_metrics:
            lines.append(
                f"| {qm.query[:30]}... | {qm.quality_score:.4f} | "
                f"{qm.jaccard_similarity:.4f} | {qm.rouge_l_f1:.4f} | "
                f"{qm.latency_s:.4f} | {qm.num_layers_skipped} | "
                f"{qm.compute_reduction_pct:.1f}% |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Analysis",
        "",
        f"- **Baseline avg latency**: {result.baseline_latency_avg:.4f} s",
    ])

    for mode_name in ("conservative", "balanced", "aggressive"):
        mr = result.modes.get(mode_name)
        if not mr:
            continue
        if result.baseline_latency_avg > 0:
            speedup_pct = (
                (result.baseline_latency_avg - mr.avg_latency_s)
                / result.baseline_latency_avg * 100
            )
        else:
            speedup_pct = 0.0
        lines.append(
            f"- **{mode_name}**: quality={mr.avg_quality_score:.4f}, "
            f"reduction={mr.avg_reduction_pct:.1f}%, "
            f"latency_speedup={speedup_pct:.1f}%"
        )

    lines.extend([
        "",
        "### Quality vs. Compute Trade-off",
        "",
        "| Mode | Quality Preserved | Compute Saved |",
        "|------|------------------|---------------|",
    ])

    for mode_name in ("conservative", "balanced", "aggressive"):
        mr = result.modes.get(mode_name)
        if not mr:
            continue
        lines.append(
            f"| {mode_name} | {mr.avg_quality_score * 100:.1f}% | "
            f"{mr.avg_reduction_pct:.1f}% |"
        )

    report = "\n".join(lines)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Wrote %s", md_path)

    # JSON
    json_data = {
        "config": result.config,
        "baseline_latency_avg": result.baseline_latency_avg,
        "total_time_s": result.total_time_s,
        "timestamp": result.timestamp,
        "modes": {
            name: {
                "avg_quality_score": mr.avg_quality_score,
                "avg_jaccard": mr.avg_jaccard,
                "avg_rouge_l": mr.avg_rouge_l,
                "avg_latency_s": mr.avg_latency_s,
                "avg_layers_skipped": mr.avg_layers_skipped,
                "avg_reduction_pct": mr.avg_reduction_pct,
                "per_query": [qm.to_dict() for qm in mr.all_metrics],
            }
            for name, mr in result.modes.items()
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", json_path)

    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CNOS v0.8.1 Importance Router — benchmark",
    )
    p.add_argument(
        "--mode", choices=("real", "simulate"), default="simulate",
    )
    p.add_argument(
        "--max-tokens", type=int, default=16,
    )
    p.add_argument(
        "--temperature", type=float, default=0.7,
    )
    p.add_argument(
        "--queries", type=int, default=None,
        help="Number of benchmark queries (default: all 5). Use 1-2 for CPU speed.",
    )
    p.add_argument(
        "--output-dir", default=None,
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    queries = BENCHMARK_QUERIES
    if args.queries is not None and args.queries < len(queries):
        queries = queries[: args.queries]

    logger.info(
        "Importance Router Benchmark  mode=%s  queries=%d  max_tokens=%d",
        args.mode, len(queries), args.max_tokens,
    )

    if args.mode == "real":
        logger.info("Loading model ...")
        t0 = time.perf_counter()
        bundle = load_model(
            model_key="tinyllama",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        logger.info("Model loaded in %.1f s", time.perf_counter() - t0)
        engine = RealBenchmark(
            bundle=bundle,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        engine = SimulatedBenchmark(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

    try:
        result = engine.run(queries)
    finally:
        if hasattr(engine, "cleanup"):
            engine.cleanup()

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "output")
    report_path = generate_report(result, output_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Importance Router Benchmark Complete!")
    for mode_name in ("conservative", "balanced", "aggressive"):
        mr = result.modes.get(mode_name)
        if mr:
            logger.info(
                "  %s: quality=%.4f  reduction=%.1f%%  latency=%.4fs",
                mode_name, mr.avg_quality_score,
                mr.avg_reduction_pct, mr.avg_latency_s,
            )
    logger.info("  Report: %s", report_path)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
