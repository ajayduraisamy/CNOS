"""benchmark  compares Baseline Transformer vs CNOS Runtime.

Collects metrics across a set of test queries and produces:
  * Console comparison table
  * Markdown report (``benchmark_report.md``)
  * CSV export (``benchmark_results.csv``)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_PROTO = os.path.join(os.path.dirname(__file__), "..")
for _dir in ("neural_paging", "layer_router", "kv_cache_compression",
             "memory_virtualization", "real_inference"):
    _p = os.path.join(_PROTO, _dir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

_OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

TEST_QUERIES: List[Dict[str, str]] = [
    {"query": "What is 2+2?", "type": "simple"},
    {"query": "What is the capital of France?", "type": "simple"},
    {"query": "Define gravity", "type": "simple"},
    {"query": "Explain how photosynthesis works", "type": "medium"},
    {"query": "Write a Python function to sort a list", "type": "medium"},
    {"query": "Compare mitosis and meiosis", "type": "medium"},
    {"query": "Derive the quadratic formula step by step", "type": "complex"},
    {"query": "Write a detailed essay on the causes of World War II", "type": "complex"},
    {"query": "Explain the theory of relativity with mathematical formulation", "type": "complex"},
]


@dataclass
class BenchmarkRow:
    query: str
    query_type: str
    baseline_latency_s: float = 0.0
    cnos_latency_s: float = 0.0
    latency_reduction_pct: float = 0.0
    baseline_layers: int = 0
    cnos_layers_executed: int = 0
    cnos_layers_skipped: int = 0
    compute_reduction_pct: float = 0.0
    ram_used_gb: float = 0.0
    page_faults: int = 0
    page_hits: int = 0
    hit_rate_pct: float = 0.0
    cache_compression_ratio: float = 1.0
    tokens_generated: int = 0

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "type": self.query_type,
            "baseline_latency_s": round(self.baseline_latency_s, 4),
            "cnos_latency_s": round(self.cnos_latency_s, 4),
            "latency_reduction_pct": round(self.latency_reduction_pct, 1),
            "baseline_layers": self.baseline_layers,
            "cnos_layers_executed": self.cnos_layers_executed,
            "cnos_layers_skipped": self.cnos_layers_skipped,
            "compute_reduction_pct": round(self.compute_reduction_pct, 1),
            "ram_used_gb": round(self.ram_used_gb, 2),
            "page_faults": self.page_faults,
            "page_hits": self.page_hits,
            "hit_rate_pct": round(self.hit_rate_pct, 1),
            "cache_compression_ratio": round(self.cache_compression_ratio, 2),
            "tokens_generated": self.tokens_generated,
        }


@dataclass
class BenchmarkSummary:
    rows: List[BenchmarkRow] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    @property
    def avg_latency_reduction(self) -> float:
        if not self.rows:
            return 0.0
        return sum(r.latency_reduction_pct for r in self.rows) / len(self.rows)

    @property
    def avg_compute_reduction(self) -> float:
        if not self.rows:
            return 0.0
        return sum(r.compute_reduction_pct for r in self.rows) / len(self.rows)

    @property
    def avg_hit_rate(self) -> float:
        if not self.rows:
            return 0.0
        return sum(r.hit_rate_pct for r in self.rows) / len(self.rows)

    @property
    def total_faults(self) -> int:
        return sum(r.page_faults for r in self.rows)

    @property
    def total_hits(self) -> int:
        return sum(r.page_hits for r in self.rows)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# CNOS Benchmark Report")
        lines.append(f"")
        lines.append(f"**Generated:** {self.timestamp}")
        lines.append(f"**Config:** {json.dumps(self.config, indent=2)}")
        lines.append(f"")
        lines.append(f"## Summary")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---:|")
        lines.append(f"| Avg Latency Reduction | {self.avg_latency_reduction:.1f}% |")
        lines.append(f"| Avg Compute Reduction | {self.avg_compute_reduction:.1f}% |")
        lines.append(f"| Avg Page Hit Rate | {self.avg_hit_rate:.1f}% |")
        lines.append(f"| Total Page Faults | {self.total_faults} |")
        lines.append(f"| Total Page Hits | {self.total_hits} |")
        lines.append(f"")
        lines.append(f"## Per-Query Results")
        lines.append(f"")
        h = "| Query | Layers | Reduction | RAM | Faults/Hits | Hit% | Compress | Latency |"
        sep = "|" + "---|" * 8
        lines.append(h)
        lines.append(sep)
        for r in self.rows:
            q_short = r.query[:45]
            layers = f"{r.cnos_layers_executed}/{r.cnos_layers_executed + r.cnos_layers_skipped}"
            ram = f"{r.ram_used_gb:.2f} GB"
            fh = f"{r.page_faults}/{r.page_hits}"
            comp = f"{r.cache_compression_ratio:.2f}x"
            lines.append(
                f"| {q_short:45s} | {layers:>7s} | {r.compute_reduction_pct:5.1f}% "
                f"| {ram:>7s} | {fh:>9s} | {r.hit_rate_pct:5.1f}% "
                f"| {comp:>6s} | {r.cnos_latency_s:.3f}s |"
            )
        lines.append(f"")
        return "\n".join(lines)

    def to_csv(self) -> str:
        if not self.rows:
            return ""
        import io
        buf = io.StringIO()
        fieldnames = list(self.rows[0].to_csv_row().keys())
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for r in self.rows:
            writer.writerow(r.to_csv_row())
        return buf.getvalue()


class Benchmark:
    """Compares Baseline Transformer vs CNOS Runtime across test queries.

    Args:
        runtime: Configured :class:`CnosRuntime` instance.
        queries: List of query dicts with ``"query"`` and ``"type"`` keys.
            Defaults to :const:`TEST_QUERIES`.
        config: Config dict (``"mode"``, ``"model"``, etc.) used when *runtime* is None.
    """

    def __init__(
        self,
        runtime: Any = None,
        queries: Optional[List[Dict[str, str]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.queries = queries or TEST_QUERIES
        self._runtime = runtime
        self._config = config or {}

    def run(self) -> BenchmarkSummary:
        from runtime import CnosRuntime, RuntimeConfig

        if self._runtime is None:
            cfg = RuntimeConfig(mode=self._config.get("mode", "simulate"))
            if "model" in self._config:
                cfg.model_key = self._config["model"]
            if "max_tokens" in self._config:
                cfg.max_tokens = self._config["max_tokens"]
            if "ram_gb" in self._config:
                cfg.ram_gb = self._config["ram_gb"]
            if "routing_policy" in self._config:
                cfg.routing_policy = self._config["routing_policy"]
            if "quantisation" in self._config:
                cfg.quantisation = self._config["quantisation"]
            self._runtime = CnosRuntime(cfg)

        rt = self._runtime
        rows: List[BenchmarkRow] = []

        print(f"\n  {'=' * 70}")
        print(f"  CNOS v0.7 Integration Benchmark")
        print(f"  Model: {rt.config.model_key}  RAM: {rt.config.ram_gb}GB  "
              f"Routing: {rt.config.routing_policy}  Quant: {rt.config.quantisation}")
        print(f"  {'=' * 70}\n")

        header = (
            f"  {'Query':40s} {'Layers':>8s} {'Reduction':>10s} "
            f"{'RAM':>8s} {'Faults/Hits':>10s} {'Hit%':>6s} {'Latency':>8s}"
        )
        print(header)
        print(f"  {'-' * 90}")

        for i, qd in enumerate(self.queries):
            query = qd["query"]
            qtype = qd.get("type", "unknown")

            try:
                baseline = rt.process_baseline(query)
                cnos = rt.process(query)

                row = BenchmarkRow(
                    query=query,
                    query_type=qtype,
                    baseline_latency_s=baseline.latency_s,
                    cnos_latency_s=cnos.latency_s,
                    latency_reduction_pct=(
                        (baseline.latency_s - cnos.latency_s) / max(baseline.latency_s, 1e-9) * 100
                    ) if baseline.latency_s > 0 else 0.0,
                    baseline_layers=baseline.layers_executed,
                    cnos_layers_executed=cnos.layers_executed,
                    cnos_layers_skipped=cnos.layers_skipped,
                    compute_reduction_pct=cnos.compute_reduction_pct,
                    ram_used_gb=cnos.memory.get("ram_used_gb", 0) if cnos.memory else 0,
                    page_faults=cnos.memory.get("page_faults", 0) if cnos.memory else 0,
                    page_hits=cnos.memory.get("page_hits", 0) if cnos.memory else 0,
                    hit_rate_pct=cnos.memory.get("hit_rate_pct", 0) if cnos.memory else 0,
                    cache_compression_ratio=(
                        cnos.cache.get("compression_ratio", 1.0) if cnos.cache else 1.0
                    ),
                    tokens_generated=cnos.tokens_generated,
                )
                rows.append(row)

            except Exception as e:
                logger.error("Query %d failed: %s", i, e)
                continue

            short_q = query[:38]
            layers = f"{row.cnos_layers_executed}/{row.cnos_layers_executed + row.cnos_layers_skipped}"
            print(
                f"  [{i + 1}] {short_q:38s} {layers:>8s} "
                f"{row.compute_reduction_pct:8.1f}% "
                f"{row.ram_used_gb:6.2f}GB "
                f"{row.page_faults:4d}/{row.page_hits:4d} "
                f"{row.hit_rate_pct:5.1f}% "
                f"{row.cnos_latency_s:6.3f}s"
            )

        summary = BenchmarkSummary(
            rows=rows,
            config={
                "model": rt.config.model_key,
                "ram_gb": rt.config.ram_gb,
                "routing_policy": rt.config.routing_policy,
                "quantisation": rt.config.quantisation,
                "eviction_policy": rt.config.eviction_policy,
                "mode": rt.config.mode,
            },
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

        self._print_summary(summary)
        self._write_reports(summary)
        rt.cleanup()
        return summary

    @staticmethod
    def _print_summary(summary: BenchmarkSummary) -> None:
        print(f"\n  {'=' * 70}")
        print(f"  Benchmark Summary")
        print(f"  {'=' * 70}")
        print(f"  Avg Latency Reduction:  {summary.avg_latency_reduction:.1f}%")
        print(f"  Avg Compute Reduction:  {summary.avg_compute_reduction:.1f}%")
        print(f"  Avg Page Hit Rate:      {summary.avg_hit_rate:.1f}%")
        print(f"  Total Page Faults:      {summary.total_faults}")
        print(f"  Total Page Hits:        {summary.total_hits}")
        print(f"  Queries:                {len(summary.rows)}")
        print(f"  {'=' * 70}\n")

    @staticmethod
    def _write_reports(summary: BenchmarkSummary) -> None:
        os.makedirs(_OUT_DIR, exist_ok=True)

        md_path = os.path.join(_OUT_DIR, "benchmark_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(summary.to_markdown())
        print(f"  Markdown report: {md_path}")

        csv_path = os.path.join(_OUT_DIR, "benchmark_results.csv")
        csv_content = summary.to_csv()
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(csv_content)
        print(f"  CSV report:      {csv_path}")

        json_path = os.path.join(_OUT_DIR, "benchmark_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": summary.config,
                "summary": {
                    "avg_latency_reduction_pct": round(summary.avg_latency_reduction, 1),
                    "avg_compute_reduction_pct": round(summary.avg_compute_reduction, 1),
                    "avg_hit_rate_pct": round(summary.avg_hit_rate, 1),
                    "total_faults": summary.total_faults,
                    "total_hits": summary.total_hits,
                },
                "results": [r.to_csv_row() for r in summary.rows],
            }, f, indent=2)
        print(f"  JSON report:     {json_path}")


def parse_args(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse CLI arguments into a config dict."""
    import argparse
    p = argparse.ArgumentParser(description="CNOS Integration Benchmark")
    p.add_argument("--mode", default="simulate", choices=["simulate", "real"],
                   help="Execution mode (real requires model download)")
    p.add_argument("--model", default="tinyllama",
                   choices=["tinyllama", "qwen-1.5b", "llama-3.2-1b"],
                   help="Model key")
    p.add_argument("--max-tokens", type=int, default=32,
                   help="Max tokens to generate per query")
    p.add_argument("--ram-gb", type=float, default=4.0,
                   help="Simulated RAM in GB")
    p.add_argument("--routing-policy", default="adaptive",
                   help="Layer routing policy")
    p.add_argument("--quantisation", default="int8",
                   help="KV cache quantisation scheme")
    args = p.parse_args(argv)
    return {
        "mode": args.mode,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "ram_gb": args.ram_gb,
        "routing_policy": args.routing_policy,
        "quantisation": args.quantisation,
    }


def main() -> int:
    import sys
    config = parse_args()
    print(f"\n  CNOS Benchmark  mode={config['mode']}  model={config['model']}")
    bm = Benchmark(config=config)
    summary = bm.run()
    return 0 if len(summary.rows) > 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
    sys.exit(main())
