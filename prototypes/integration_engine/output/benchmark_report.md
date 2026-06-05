# CNOS Benchmark Report

**Generated:** 2026-06-05T15:11:17
**Config:** {
  "model": "tinyllama",
  "ram_gb": 4.0,
  "routing_policy": "adaptive",
  "quantisation": "int8",
  "eviction_policy": "lru",
  "mode": "simulate"
}

## Summary

| Metric | Value |
|---|---:|
| Avg Latency Reduction | -110.0% |
| Avg Compute Reduction | 54.5% |
| Avg Page Hit Rate | 25.0% |
| Total Page Faults | 20 |
| Total Page Hits | 10 |

## Per-Query Results

| Query | Layers | Reduction | RAM | Faults/Hits | Hit% | Compress | Latency |
|---|---|---|---|---|---|---|---|
| test one                                      |   10/22 |  54.5% | 0.01 GB |      10/0 |   0.0% |  4.00x | 0.301s |
| test two medium complexity                    |   10/22 |  54.5% | 0.01 GB |     10/10 |  50.0% |  4.00x | 0.011s |
