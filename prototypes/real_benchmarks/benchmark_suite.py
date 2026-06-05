"""benchmark_suite -- orchestrator that runs baseline and CNOS benchmarks on a real model.

Usage:
    python prototypes/real_benchmarks/benchmark_suite.py \\
        --model tinyllama \\
        --max-tokens 64 \\
        --routing-policy adaptive \\
        --quantisation int8 \\
        --ram-gb 4.0

Outputs:
    * output/benchmark_report.md
    * output/benchmark_results.csv
    * output/benchmark_results.json
    * output/baseline_results.json
    * output/cnos_results.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

_PROTO = os.path.dirname(__file__)
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_suite")

# Standard benchmark queries covering different complexity levels
DEFAULT_QUERIES = [
    "What is 2 + 2?",
    "Explain the concept of gravity in simple terms.",
    "Write a short poem about artificial intelligence.",
    "Describe the process of photosynthesis.",
    "What are the main differences between Python and JavaScript?",
    "Explain how neural networks work.",
    "Write a brief summary of the French Revolution.",
    "What is the capital of Australia?",
    "Explain the concept of recursion in programming.",
    "Describe the water cycle.",
]


def run_benchmark(
    model_key: str = "tinyllama",
    max_tokens: int = 64,
    routing_policy: str = "adaptive",
    quantisation: str = "int8",
    ram_gb: float = 4.0,
    device: str = "",
    queries: Optional[list] = None,
    timeout_s: float = 300.0,
    no_save: bool = False,
) -> Any:
    """Run the full benchmark flow: load -> baseline -> cnos -> metrics -> reports.

    Args:
        model_key: Model key (``"tinyllama"`` or ``"qwen-1.5b"``).
        max_tokens: Max tokens to generate per query.
        routing_policy: Layer routing policy name.
        quantisation: KV cache compression scheme.
        ram_gb: Simulated RAM in GB (for memory virtualization).
        device: Device override (``"cpu"``, ``"cuda"``, ``"mps"``).
        queries: List of query strings (default: standard 10 queries).
        timeout_s: Max seconds for model download.
        no_save: If True, skip saving results to disk.

    Returns:
        The :class:`MetricsReport`.
    """
    import benchmark_loader as ml

    if queries is None:
        queries = DEFAULT_QUERIES

    # 1. Load model
    logger.info("Loading model %s (timeout=%ds)...", model_key, timeout_s)
    t0 = time.perf_counter()

    dev = device if device else None
    load_result = ml.load_model(
        model_key=model_key,
        device=dev,
        timeout_s=timeout_s,
    )

    if not load_result.success:
        logger.error("Model load failed: %s", load_result.error)
        return None

    logger.info(
        "Model loaded in %.2fs (RAM: %.0fMB)",
        load_result.load_time_s, load_result.memory_mb,
    )

    bundle = load_result.bundle

    from baseline_runner import BaselineRunner
    from cnos_runner import CnosRunner

    # 2. Baseline
    logger.info("Running baseline (%d queries)...", len(queries))
    baseline = BaselineRunner(bundle=bundle, max_tokens=max_tokens)
    baseline_result = baseline.run_queries(queries)
    baseline.cleanup()

    if not no_save:
        baseline_result.save()

    # 3. CNOS
    logger.info("Running CNOS (%d queries)...", len(queries))
    cnos = CnosRunner(
        bundle=bundle,
        max_tokens=max_tokens,
        routing_policy=routing_policy,
        quantisation=quantisation,
        ram_gb=ram_gb,
    )
    cnos_result = cnos.run_queries(queries)
    cnos.cleanup()

    if not no_save:
        cnos_result.save()

    # 4. Compute metrics
    logger.info("Computing comparison metrics...")
    from metrics_collector import compute_metrics, save_metrics_json
    report = compute_metrics(baseline_result, cnos_result)

    if not no_save:
        save_metrics_json(report)

    # 5. Generate reports
    from report_generator import generate_all_reports
    paths = generate_all_reports(report)

    logger.info("Benchmark complete.")
    for fmt, path in paths.items():
        logger.info("  %s: %s", fmt.upper(), path)

    return report


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNOS Real Benchmark Suite",
    )
    parser.add_argument(
        "--model", default="tinyllama",
        choices=["tinyllama", "qwen-1.5b"],
        help="Model to benchmark",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--routing-policy", default="adaptive",
        help="Layer routing policy",
    )
    parser.add_argument(
        "--quantisation", default="int8",
        help="KV cache quantisation",
    )
    parser.add_argument(
        "--ram-gb", type=float, default=4.0,
        help="Simulated RAM in GB",
    )
    parser.add_argument(
        "--device", default="",
        help="Device override (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Model download timeout (s)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving results to disk",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        model_key=args.model,
        max_tokens=args.max_tokens,
        routing_policy=args.routing_policy,
        quantisation=args.quantisation,
        ram_gb=args.ram_gb,
        device=args.device or None,
        timeout_s=args.timeout,
        no_save=args.no_save,
    )
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
