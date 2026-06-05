# CNOS Dynamic Layer Router

> **Reduce computation by skipping unnecessary transformer layers based on query complexity.**

## Overview

Not every user query requires all 80 layers of a transformer model. Simple factual lookups can get by with a fraction of the layers, while complex reasoning tasks need the full depth. The Dynamic Layer Router analyses each query, estimates its complexity, and generates a **layer execution plan** — the minimal set of layers needed to produce a high-quality response.

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Complexity      │────▶│ Routing Policy   │────▶│ Layer Selector   │
│ Detector        │     │ (static /        │     │ (produces final  │
│                 │     │  adaptive /      │     │  execution plan) │
│ score + type    │     │  experimental)   │     │                  │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────┐
                                              │ Inference Engine │
                                              │ (CNOS Neural     │
                                              │  Paging + Cache) │
                                              └──────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `complexity_detector.py` | Analyses query text → complexity score, type, reasoning depth |
| `routing_policy.py` | Generates layer plans: `StaticPolicy`, `AdaptivePolicy`, `ExperimentalPolicy` |
| `layer_selector.py` | Unified API combining detector + policy into a single call |
| `metrics.py` | Tracks cumulative statistics (layers skipped, compute reduction, memory savings) |
| `benchmark.py` | Runs 300 queries (100/100/100) across multiple policies and prints comparison |
| `test_router.py` | Validation suite for all components |

## Policies

### Static
Hand-crafted fixed plans: simple → ~20 layers, medium → ~40 layers, complex → all 80.

### Adaptive
Starts from the static plan but grows when the model's early-exit confidence falls below a threshold. Simulates a speculative early-exit mechanism.

### Experimental

| Strategy | Description |
|----------|-------------|
| `even-odd` | Alternates between even and odd layer sets (context-cache friendly) |
| `cluster` | Groups layers into functional clusters and selects entire clusters |
| `density` | Samples densely near input/output, sparsely in the middle |
| `random-topk` | Weighted random subset (later layers weighted higher) |

## Quick Start

```bash
cd prototypes/layer_router

# Run the test suite
python test_router.py

# Run the benchmark (300 queries across 3 policies)
python benchmark.py

# Compare specific policies
python benchmark.py --policies static adaptive experimental/density experimental/cluster

# Custom query load
python benchmark.py --num-simple 50 --num-medium 50 --num-complex 50
```

## Example Output

```
  CNOS Dynamic Layer Router -- Benchmark
  300 queries (100S / 100M / 100C) | 80 layers
  Policies: static, adaptive, experimental/density

======================================================================
  Policy Comparison Summary
======================================================================
  Policy                   Layers/Query   Reduction    Time (s)   Mem Saved
  ------------------------------------------------------------------------
  static                   46.67          41.67%       0.01       4998
  adaptive                 46.67          41.67%       0.01       4998
  experimental/density     37.60          53.00%       0.02       6361
======================================================================
```

## Dependencies

Zero external dependencies — uses only the Python standard library (Python 3.11+).
