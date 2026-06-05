# CNOS v0.8.1 — Importance-Based Layer Router

Routes inference through only the important layers, skipping low/medium-impact layers while preserving critical ones.

## Architecture

```
LayerProfile (importance scores)
        |
        v
ImportanceRouter (routing decisions)
        |
        v
RoutedInferenceEngine (generation)
        |
        v
QualityEvaluator (compare vs baseline)
        |
        v
Benchmark Report (Markdown + JSON)
```

## Files

| File | Purpose |
|------|---------|
| `layer_profile.py` | Loads layer importance data, provides skip candidates |
| `importance_router.py` | Routing decisions: conservative / balanced / aggressive |
| `quality_evaluator.py` | Jaccard, ROUGE-L, quality score comparison |
| `benchmark.py` | CLI entry point + report generation |
| `test_importance_router.py` | 30+ unit/integration tests |
| `README.md` | This file |

## Routing Modes

| Mode | Skip Budget | Typical Layers Skipped |
|------|-------------|----------------------|
| Conservative | 1–2 | 19 (lowest impact) |
| Balanced | 2–3 | 19, 8 |
| Aggressive | 3–4 | 19, 8, 9 |

**Guarantee:** Critical (high impact) layers are **never skipped**.

## Quick Start

```bash
# Simulated (no model needed)
python prototypes/importance_router/benchmark.py --mode simulate

# Real TinyLlama benchmark (2 queries for speed)
python prototypes/importance_router/benchmark.py --mode real --max-tokens 16 --queries 2

# Full benchmark (all 5 queries, takes ~15 min on CPU)
python prototypes/importance_router/benchmark.py --mode real --max-tokens 16 --queries 5
```

## Tests

```bash
python -m pytest prototypes/importance_router/test_importance_router.py -v
```

## Output

`prototypes/importance_router/output/`:

| File | Format |
|------|--------|
| `importance_router_report.md` | Comparative report across all 3 modes |
| `importance_router_results.json` | Per-query metrics |

## Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| Quality Score | 0–1 | Composite (0.5 Jaccard + 0.5 ROUGE-L F1) |
| Similarity Score | 0–1 | Token-set overlap with baseline |
| Quality Loss | 0–1 | 1 − Quality Score |
| Compute Reduction | 0–100% | Percentage of layers skipped |

## Requirements

- Python 3.10+
- PyTorch 2.12+
- Transformers 5.9+
- sentencepiece 0.2+
- accelerate 1.13+
