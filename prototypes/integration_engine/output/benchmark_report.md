# CNOS Benchmark Report

**Generated:** 2026-06-05T15:49:49
**Config:** {
  "model": "tinyllama",
  "ram_gb": 4.0,
  "routing_policy": "adaptive",
  "quantisation": "int8",
  "eviction_policy": "lru",
  "mode": "real"
}

## Summary

| Metric | Value |
|---|---:|
| Avg Latency Reduction | 61.3% |
| Avg Compute Reduction | 54.5% |
| Avg Page Hit Rate | 38.9% |
| Total Page Faults | 30 |
| Total Page Hits | 30 |

## Per-Query Results

| Query | Layers | Reduction | RAM | Faults/Hits | Hit% | Compress | Latency |
|---|---|---|---|---|---|---|---|
| What is 2+2?                                  |   10/22 |  54.5% | 0.01 GB |      10/0 |   0.0% |  4.00x | 45.339s |
| What is the capital of France?                |   10/22 |  54.5% | 0.01 GB |     10/10 |  50.0% |  4.00x | 16.777s |
| Define gravity                                |   10/22 |  54.5% | 0.01 GB |     10/20 |  66.7% |  4.00x | 13.700s |
