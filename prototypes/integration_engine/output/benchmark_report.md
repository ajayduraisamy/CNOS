# CNOS Benchmark Report

**Generated:** 2026-06-05T14:46:58
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
| Avg Latency Reduction | -556.0% |
| Avg Compute Reduction | 52.5% |
| Avg Page Hit Rate | 64.4% |
| Total Page Faults | 120 |
| Total Page Hits | 350 |

## Per-Query Results

| Query | Layers | Reduction | RAM | Faults/Hits | Hit% | Compress | Latency |
|---|---|---|---|---|---|---|---|
| What is 2+2?                                  |   10/22 |  54.5% | 0.01 GB |      10/0 |   0.0% |  4.00x | 0.386s |
| What is the capital of France?                |   10/22 |  54.5% | 0.01 GB |     10/10 |  50.0% |  4.00x | 0.004s |
| Define gravity                                |   10/22 |  54.5% | 0.01 GB |     10/20 |  66.7% |  4.00x | 0.008s |
| Explain how photosynthesis works              |   10/22 |  54.5% | 0.01 GB |     10/30 |  75.0% |  4.00x | 0.015s |
| Write a Python function to sort a list        |   14/22 |  36.4% | 0.02 GB |     16/38 |  70.4% |  4.00x | 0.186s |
| Compare mitosis and meiosis                   |   10/22 |  54.5% | 0.02 GB |     16/48 |  75.0% |  4.00x | 0.013s |
| Derive the quadratic formula step by step     |   10/22 |  54.5% | 0.02 GB |     16/58 |  78.4% |  4.00x | 0.006s |
| Write a detailed essay on the causes of World |   10/22 |  54.5% | 0.02 GB |     16/68 |  81.0% |  4.00x | 0.016s |
| Explain the theory of relativity with mathema |   10/22 |  54.5% | 0.02 GB |     16/78 |  83.0% |  4.00x | 0.014s |
