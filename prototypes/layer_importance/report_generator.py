"""report_generator — produces JSON, CSV, and Markdown output from ablation results."""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from quality_metrics import classify_importance
from layer_ablation import AblationStudyResult, LayerImportanceResult

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def write_json(
    result: AblationStudyResult,
    filepath: Optional[str] = None,
) -> str:
    """Write *layer_importance.json* (full data)."""
    if filepath is None:
        filepath = os.path.join(DEFAULT_OUTPUT_DIR, "layer_importance.json")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    data = result.to_dict()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", filepath)
    return filepath


def write_csv(
    result: AblationStudyResult,
    filepath: Optional[str] = None,
) -> str:
    """Write *layer_importance.csv*."""
    if filepath is None:
        filepath = os.path.join(DEFAULT_OUTPUT_DIR, "layer_importance.csv")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fieldnames = [
        "layer",
        "avg_impact_score",
        "classification",
        "num_queries",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for layer in result.per_layer:
            writer.writerow({
                "layer": layer.layer,
                "avg_impact_score": round(layer.avg_impact_score, 4),
                "classification": layer.classification,
                "num_queries": len(layer.impact_scores),
            })
    logger.info("Wrote %s", filepath)
    return filepath


def _score_distribution_table(result: AblationStudyResult) -> str:
    """Build the Markdown table rows for per-layer impact scores."""
    lines = [
        "| Layer | Avg Impact Score | Classification | Queries |",
        "|-------|-----------------|----------------|---------|",
    ]
    for layer in result.per_layer:
        lines.append(
            f"| {layer.layer} | {layer.avg_impact_score:.4f} | "
            f"{layer.classification} | {len(layer.impact_scores)} |"
        )
    return "\n".join(lines)


def _summary_text(result: AblationStudyResult) -> str:
    high = result.high_impact_layers
    medium = result.medium_impact_layers
    low = result.low_impact_layers

    total = len(result.per_layer)
    lines = [
        f"- **Total layers analyzed:** {total}",
        f"- **High impact layers** (score ≥ 0.30): {len(high)} — {high}",
        f"- **Medium impact layers** (score 0.10–0.30): {len(medium)} — {medium}",
        f"- **Low impact layers** (score < 0.10): {len(low)} — {low}",
        "",
        "### Interpretation",
        "",
        "- **High impact** — Disabling this layer causes significant quality loss.",
        "  These layers are critical and should NOT be skipped in routing.",
        "- **Medium impact** — Some quality degradation; may be safe to skip",
        "  under aggressive routing budgets.",
        "- **Low impact** — Minimal quality loss when disabled; good candidates",
        "  for skipping during inference.",
    ]
    return "\n".join(lines)


def _per_query_details(result: AblationStudyResult) -> str:
    sections = []
    for layer in result.per_layer:
        if not layer.per_query:
            continue
        lname = classify_importance(layer.avg_impact_score).upper()
        sections.append(f"### Layer {layer.layer}  ({lname}, score={layer.avg_impact_score:.4f})")
        sections.append("")
        sections.append("| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |")
        sections.append("|-------|---------|---------|-------------|-------------|-------------|")
        for comp in layer.per_query:
            sections.append(
                f"| {comp.query[:40]}... | {comp.jaccard_similarity:.4f} | "
                f"{comp.rouge_l_f1:.4f} | {comp.length_ratio:.4f} | "
                f"{comp.latency_s:.4f} | {comp.impact_score:.4f} |"
            )
        sections.append("")
    return "\n".join(sections)


def generate_markdown_report(
    result: AblationStudyResult,
    filepath: Optional[str] = None,
) -> str:
    """Write *layer_importance_report.md*."""
    if filepath is None:
        filepath = os.path.join(DEFAULT_OUTPUT_DIR, "layer_importance_report.md")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    high_str = ", ".join(str(l) for l in result.high_impact_layers) or "none"
    medium_str = ", ".join(str(l) for l in result.medium_impact_layers) or "none"
    low_str = ", ".join(str(l) for l in result.low_impact_layers) or "none"

    report = f"""# Layer Importance Study — {result.model_key}

**Date:** {result.timestamp}  
**Total time:** {result.total_time_s:.1f} s  
**Configuration:** {json.dumps(result.config, indent=2)}

---

## Summary

{_summary_text(result)}

---

## Per-Layer Impact Scores

{_score_distribution_table(result)}

---

## Classification

| Category | Count | Layers |
|----------|-------|--------|
| High Impact | {len(result.high_impact_layers)} | {high_str} |
| Medium Impact | {len(result.medium_impact_layers)} | {medium_str} |
| Low Impact | {len(result.low_impact_layers)} | {low_str} |

---

## Per-Query Details

{_per_query_details(result)}

---

## Scoring Methodology

### Layer Impact Score

Each layer's impact is computed as the average across all queries:

```
Impact Score = 1 - (0.5 * Jaccard + 0.5 * ROUGE-L F1)
```

Where:

- **Jaccard similarity** — Token-set overlap between baseline and ablated response.
- **ROUGE-L F1** — Longest common subsequence based recall/precision F1.
- **Impact Score** — Composite quality loss (0 = no loss, 1 = complete loss).

### Classification Thresholds

| Classification | Score Range | Meaning |
|----------------|-------------|---------|
| **High** | >= 0.30 | Critical layer — severe quality loss when removed |
| **Medium** | 0.10 – 0.30 | Moderate importance — some loss |
| **Low** | < 0.10 | Low importance — safe to skip |
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Wrote %s", filepath)
    return filepath


def generate_all_reports(
    result: AblationStudyResult,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Generate all report formats.

    Args:
        result: Study results.
        output_dir: Output directory (default ``output/``).

    Returns:
        ``{"json": ..., "csv": ..., "md": ...}`` mapping format to filepath.
    """
    d = output_dir or DEFAULT_OUTPUT_DIR
    return {
        "json": write_json(result, os.path.join(d, "layer_importance.json")),
        "csv": write_csv(result, os.path.join(d, "layer_importance.csv")),
        "md": generate_markdown_report(result, os.path.join(d, "layer_importance_report.md")),
    }
