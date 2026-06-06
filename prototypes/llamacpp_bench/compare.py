"""compare.py — compare CNOS (PyTorch) vs llama.cpp benchmark results.

Loads two result JSON files and generates a side-by-side comparison:

  * CNOS runtime (real_inference / RoutedInferenceEngine)
  * llama.cpp (llama-cli.exe with GGUF Q4_K_M)

Metrics compared:
  * Latency per query (seconds)
  * Tokens per second
  * Response length (characters)
  * Response text (side by side)

Usage:
    python prototypes/llamacpp_bench/compare.py \\
        --cnos <cnos_results.json> \\
        --llama <llamacpp_bench_results.json> \\
        --output output/comparison_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ComparisonRow:
    """Single-query comparison between CNOS and llama.cpp.

    Attributes:
        query: Input prompt.
        cnos_latency_s: CNOS inference latency.
        cnos_response: CNOS generated text.
        cnos_tokens: CNOS tokens generated.
        llama_latency_s: llama.cpp inference latency.
        llama_response: llama.cpp generated text.
        llama_tokens: llama.cpp tokens generated.
        speedup_x: llama.cpp latency / CNOS latency.
    """
    query: str = ""
    cnos_latency_s: float = 0.0
    cnos_response: str = ""
    cnos_tokens: int = 0
    llama_latency_s: float = 0.0
    llama_response: str = ""
    llama_tokens: int = 0

    @property
    def speedup_x(self) -> Optional[float]:
        if self.cnos_latency_s > 0:
            return round(self.llama_latency_s / self.cnos_latency_s, 3)
        return None


@dataclass
class ComparisonReport:
    """Full comparison between CNOS and llama.cpp.

    Attributes:
        rows: Per-query comparisons.
        cnos_model: Model name from CNOS.
        llama_model: Model name from llama.cpp.
        cnos_avg_latency: Average CNOS latency.
        llama_avg_latency: Average llama.cpp latency.
        cnos_avg_tokens: Average CNOS output tokens.
        llama_avg_tokens: Average llama.cpp output tokens.
        avg_speedup: Average speedup factor.
    """
    rows: List[ComparisonRow] = field(default_factory=list)
    cnos_model: str = ""
    llama_model: str = ""
    cnos_avg_latency: float = 0.0
    llama_avg_latency: float = 0.0
    cnos_avg_tokens: float = 0.0
    llama_avg_tokens: float = 0.0
    avg_speedup: Optional[float] = None

    def to_markdown(self) -> str:
        lines = [
            "# CNOS vs llama.cpp — Benchmark Comparison",
            "",
            f"**Date:** {datetime.now().isoformat(timespec='seconds')}",
            f"**CNOS model:** {self.cnos_model}",
            f"**llama.cpp model:** {self.llama_model}",
            "",
            "---",
            "",
            "## Summary",
            "",
            "| Metric | CNOS (PyTorch) | llama.cpp (GGUF) |",
            "|--------|----------------|-------------------|",
        ]
        if self.cnos_avg_latency > 0:
            lines.append(
                f"| Avg latency (s) | {self.cnos_avg_latency:.4f} | "
                f"{self.llama_avg_latency:.4f} |"
            )
        if self.avg_speedup is not None:
            lines.append(
                f"| Avg speedup | — | {self.avg_speedup:.3f}x |"
            )
        if self.cnos_avg_tokens > 0:
            lines.append(
                f"| Avg tokens | {self.cnos_avg_tokens:.1f} | "
                f"{self.llama_avg_tokens:.1f} |"
            )
        lines.extend([
            "",
            "---",
            "",
            "## Per-Query Comparison",
            "",
            "| Query | CNOS Latency (s) | llama Latency (s) | Speedup | CNOS Response | llama Response |",
            "|-------|------------------|--------------------|---------|---------------|-----------------|",
        ])

        for row in self.rows:
            speedup_str = f"{row.speedup_x:.3f}x" if row.speedup_x else "N/A"
            lines.append(
                f"| {row.query[:30]}... | {row.cnos_latency_s:.4f} | "
                f"{row.llama_latency_s:.4f} | {speedup_str} | "
                f"{row.cnos_response[:50]}... | {row.llama_response[:50]}... |"
            )

        lines.append("")
        return "\n".join(lines)


def load_results(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSON results file and return the list of query results."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # Support nested format: {"modes": {"baseline": {"per_query": [...]}}}
    if isinstance(data, dict):
        for key in ("per_query", "results", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Unrecognized JSON format in {filepath}")


def build_report(
    cnos_file: str,
    llama_file: str,
    cnos_model: str = "tinyllama",
    llama_model: str = "tinyllama (GGUF)",
) -> ComparisonReport:
    """Build a ComparisonReport from CNOS and llama.cpp result JSON files."""
    cnos_data = load_results(cnos_file)
    llama_data = load_results(llama_file)

    rows: List[ComparisonRow] = []
    cnos_lats = []
    llama_lats = []
    cnos_toks = []
    llama_toks = []

    n = min(len(cnos_data), len(llama_data))
    for i in range(n):
        c = cnos_data[i]
        l = llama_data[i]
        row = ComparisonRow(
            query=c.get("query", l.get("query", f"query_{i}")),
            cnos_latency_s=c.get("latency_s", c.get("routed_latency_s", 0)),
            cnos_response=c.get("response", ""),
            cnos_tokens=c.get("num_tokens_generated", c.get("tokens_generated", 0)),
            llama_latency_s=l.get("latency_s", 0),
            llama_response=l.get("response", ""),
            llama_tokens=l.get("tokens_generated", 0),
        )
        rows.append(row)
        cnos_lats.append(row.cnos_latency_s)
        llama_lats.append(row.llama_latency_s)
        cnos_toks.append(row.cnos_tokens)
        llama_toks.append(row.llama_tokens)

    avg_cnos_lat = sum(cnos_lats) / max(len(cnos_lats), 1)
    avg_llama_lat = sum(llama_lats) / max(len(llama_lats), 1)
    avg_speedup = avg_cnos_lat / avg_llama_lat if avg_llama_lat > 0 else None

    return ComparisonReport(
        rows=rows,
        cnos_model=cnos_model,
        llama_model=llama_model,
        cnos_avg_latency=round(avg_cnos_lat, 4),
        llama_avg_latency=round(avg_llama_lat, 4),
        cnos_avg_tokens=round(sum(cnos_toks) / max(len(cnos_toks), 1), 1),
        llama_avg_tokens=round(sum(llama_toks) / max(len(llama_toks), 1), 1),
        avg_speedup=round(avg_speedup, 3) if avg_speedup else None,
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare CNOS vs llama.cpp benchmark results",
    )
    p.add_argument("--cnos", required=True, help="CNOS results JSON")
    p.add_argument("--llama", required=True, help="llama.cpp results JSON")
    p.add_argument("--output", default=None, help="Output Markdown report path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )

    if not os.path.isfile(args.cnos):
        logger.error("CNOS results file not found: %s", args.cnos)
        return 1
    if not os.path.isfile(args.llama):
        logger.error("llama.cpp results file not found: %s", args.llama)
        return 1

    report = build_report(args.cnos, args.llama)
    md = report.to_markdown()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("Wrote: %s", args.output)
    else:
        print(md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
