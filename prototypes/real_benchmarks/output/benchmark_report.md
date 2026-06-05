# CNOS Real Benchmark Report

**Date:** 2026-06-05 15:14:23  
**Model:** tinyllama (22 layers)  
**Routing Policy:** adaptive  
**Quantisation:** int8  
**Queries:** 1

## Summary

| Metric | Baseline | CNOS | Change |
|--------|----------|------|--------|
| Latency (s) | 1.00 | 0.80 | +20.0% |
| RAM Peak (MB) | 500 | 400 | +20.0% |
| Compute Reduction | -- | 31.8% | -- |
| Jaccard Similarity | -- | 0.850 | -- |
| ROUGE-L | -- | 0.780 | -- |
| Cache Hit Rate | -- | 90.0% | -- |
| Compression Ratio | -- | 2.00x | -- |
| Tokens/sec | 12.5 | -- | -- |

## Per-Query Details

| # | Latency (s) | RAM (MB) | Layers Skipped | Jaccard | ROUGE-L |
|---|------------|----------|----------------|---------|---------|
| 1 | 1.00 / 0.80 | 500 / 400 | 7 / 22 | 0.850 | 0.780 |

## Query Samples

### Query 1: test query

**Baseline response:**  
> baseline

**CNOS response:**  
> cnos
