"""report_generator -- generates Markdown, CSV, and JSON benchmark reports.

Produces:
  * ``benchmark_report.md`` -- human-readable Markdown summary
  * ``benchmark_results.csv`` -- per-query results in CSV
  * ``benchmark_results.json`` -- full data as JSON
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def generate_markdown_report(
    report: Any,
    output_path: Optional[str] = None,
) -> str:
    """Generate a Markdown report from a MetricsReport object.

    Args:
        report: A :class:`MetricsReport` with aggregate + per-query data.
        output_path: Path for output file; defaults to ``output/benchmark_report.md``.

    Returns:
        The path to the written report.
    """
    if output_path is None:
        os.makedirs(_OUT_DIR, exist_ok=True)
        output_path = os.path.join(_OUT_DIR, "benchmark_report.md")

    lines: List[str] = []
    lines.append("# CNOS Real Benchmark Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Model:** {report.model_key} ({report.num_layers} layers)  ")
    lines.append(f"**Routing Policy:** {report.routing_policy}  ")
    lines.append(f"**Quantisation:** {report.quantisation}  ")
    lines.append(f"**Queries:** {report.num_queries}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Baseline | CNOS | Change |")
    lines.append("|--------|----------|------|--------|")
    lines.append(
        f"| Latency (s) | {report.avg_baseline_latency_s:.2f} | "
        f"{report.avg_cnos_latency_s:.2f} | "
        f"{report.avg_latency_reduction_pct:+.1f}% |"
    )
    lines.append(
        f"| RAM Peak (MB) | {report.avg_baseline_ram_peak_mb:.0f} | "
        f"{report.avg_cnos_ram_peak_mb:.0f} | "
        f"{report.avg_ram_reduction_pct:+.1f}% |"
    )
    lines.append(
        f"| Compute Reduction | -- | "
        f"{report.avg_compute_reduction_pct:.1f}% | -- |"
    )
    lines.append(
        f"| Jaccard Similarity | -- | "
        f"{report.avg_jaccard_sim:.3f} | -- |"
    )
    lines.append(
        f"| ROUGE-L | -- | "
        f"{report.avg_rouge_l:.3f} | -- |"
    )
    lines.append(
        f"| Cache Hit Rate | -- | "
        f"{report.avg_cache_hit_rate_pct:.1f}% | -- |"
    )
    lines.append(
        f"| Compression Ratio | -- | "
        f"{report.avg_compression_ratio:.2f}x | -- |"
    )
    lines.append(
        f"| Tokens/sec | {report.avg_tokens_per_sec:.1f} | -- | -- |"
    )
    lines.append("")

    lines.append("## Per-Query Details")
    lines.append("")
    lines.append(
        "| # | Latency (s) | RAM (MB) | Layers Skipped | "
        "Jaccard | ROUGE-L |"
    )
    lines.append(
        "|---|------------|----------|----------------|---------|---------|"
    )
    for i, d in enumerate(report.details, 1):
        lines.append(
            f"| {i} | {d.baseline_latency_s:.2f} / {d.cnos_latency_s:.2f} "
            f"| {d.baseline_ram_peak_mb:.0f} / {d.cnos_ram_peak_mb:.0f} "
            f"| {d.layers_skipped} / {report.num_layers} "
            f"| {d.jaccard_sim:.3f} "
            f"| {d.rouge_l:.3f} |"
        )
    lines.append("")

    lines.append("## Query Samples")
    lines.append("")
    for i, d in enumerate(report.details, 1):
        lines.append(f"### Query {i}: {d.query}")
        lines.append("")
        lines.append("**Baseline response:**  ")
        lines.append(f"> {d.baseline_response}")
        lines.append("")
        lines.append("**CNOS response:**  ")
        lines.append(f"> {d.cnos_response}")
        lines.append("")

    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Markdown report saved to %s", output_path)
    return output_path


def generate_csv_report(
    report: Any,
    output_path: Optional[str] = None,
) -> str:
    """Generate a CSV report from a MetricsReport.

    Args:
        report: A :class:`MetricsReport`.
        output_path: Path for output file.

    Returns:
        The path to the written CSV.
    """
    if output_path is None:
        os.makedirs(_OUT_DIR, exist_ok=True)
        output_path = os.path.join(_OUT_DIR, "benchmark_results.csv")

    fieldnames = [
        "query",
        "baseline_latency_s",
        "cnos_latency_s",
        "latency_reduction_pct",
        "baseline_ram_peak_mb",
        "cnos_ram_peak_mb",
        "ram_reduction_pct",
        "tokens_generated",
        "layers_skipped",
        "compute_reduction_pct",
        "jaccard_sim",
        "rouge_l",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in report.details:
            row = d.to_dict()
            # Remove long text fields
            row.pop("baseline_response", None)
            row.pop("cnos_response", None)
            writer.writerow(row)

    logger.info("CSV report saved to %s", output_path)
    return output_path


def generate_json_report(
    report: Any,
    output_path: Optional[str] = None,
) -> str:
    """Generate a JSON report from a MetricsReport.

    Args:
        report: A :class:`MetricsReport`.
        output_path: Path for output file.

    Returns:
        The path to the written JSON.
    """
    if output_path is None:
        os.makedirs(_OUT_DIR, exist_ok=True)
        output_path = os.path.join(_OUT_DIR, "benchmark_results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    logger.info("JSON report saved to %s", output_path)
    return output_path


def generate_all_reports(report: Any) -> Dict[str, str]:
    """Generate Markdown, CSV, and JSON reports.

    Args:
        report: A :class:`MetricsReport`.

    Returns:
        Dict mapping format name to file path.
    """
    paths = {}
    paths["markdown"] = generate_markdown_report(report)
    paths["csv"] = generate_csv_report(report)
    paths["json"] = generate_json_report(report)
    return paths
