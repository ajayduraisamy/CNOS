# CNOS v0.8 — Layer Importance Study

Discover which transformer layers are important and which can be skipped with minimal quality loss.

## Overview

This module performs a **layer ablation study** on TinyLlama-1.1B-Chat-v1.0:

1. Load the model and install layer gates (CNOS LayerRouter v0.3).
2. Run baseline inference (all 22 layers active).
3. For each of the 22 layers, run inference with **only that one layer disabled**.
4. Compare each ablated response to baseline using quality metrics.
5. Compute a **Layer Impact Score** for each layer.
6. Classify layers as **high / medium / low** impact.

## Files

| File | Purpose |
|------|---------|
| `benchmark.py` | CLI entry point (`--mode real` or `--mode simulate`) |
| `layer_ablation.py` | :class:`LayerAblationEngine` — one-layer-at-a-time ablation |
| `quality_metrics.py` | Jaccard, ROUGE-L, composite impact score, classification |
| `report_generator.py` | JSON, CSV, Markdown report generation |
| `test_layer_importance.py` | Unit tests (44+) for all modules |
| `README.md` | This file |

## Quick Start

### Simulated (no model, for test)

```bash
python prototypes/layer_importance/benchmark.py --mode simulate
```

### Real ablation on TinyLlama

```bash
python prototypes/layer_importance/benchmark.py --mode real --max-tokens 8 --queries 1
```

Use `--queries 1` or `--queries 2` for a quick CPU test (22 ablations × N queries).
Use `--verbose` / `-v` for per-layer debug logging.

### Run tests

```bash
# From repo root
set PYTHONPATH=prototypes\layer_importance;%PYTHONPATH%
python -m pytest prototypes/layer_importance/test_layer_importance.py -v -x

# Skip slow integration tests
python -m pytest prototypes/layer_importance/test_layer_importance.py -v -x -m "not slow"
```

### Full study

```bash
python prototypes/layer_importance/benchmark.py --mode real --max-tokens 16 --queries 5
```

**Warning:** On CPU, 5 queries × 22 ablations = 110 inference runs at ~15s each ≈ 28 minutes.

## Output

All reports are written to `prototypes/layer_importance/output/`:

| File | Format |
|------|--------|
| `layer_importance.json` | Full per-layer scores + per-query details |
| `layer_importance.csv` | Per-layer summary (layer, score, classification) |
| `layer_importance_report.md` | Human-readable report with tables |

## Metrics

### Layer Impact Score

```
Impact = 1 - (0.5 * Jaccard + 0.5 * ROUGE-L F1)
```

| Metric | Range | Description |
|--------|-------|-------------|
| Jaccard similarity | 0–1 | Token-set overlap between baseline and ablated response |
| ROUGE-L F1 | 0–1 | Longest common subsequence F1 |
| Length ratio | 0–2 | Ablated token count / baseline token count |
| Impact score | 0–1 | Composite quality loss when layer is disabled |

### Classification Thresholds

| Classification | Score Range | Meaning |
|----------------|-------------|---------|
| **High** | ≥ 0.30 | Critical layer — must keep |
| **Medium** | 0.10 – 0.30 | Moderate — may skip under aggressive budget |
| **Low** | < 0.10 | Safe to skip |

## Dependencies

- Python 3.10+
- PyTorch 2.12+
- Transformers 5.9+
- sentencepiece 0.2+
- accelerate 1.13+
- psutil 7.0+

All via the project's existing `requirements.txt`.
