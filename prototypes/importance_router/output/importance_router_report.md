# Importance-Based Layer Router — Benchmark Report

**Date:** 2026-06-05T17:25:35
**Total time:** 253.9 s
**Baseline avg latency:** 26.6580 s

---

## Configuration

```json
{
  "mode": "real",
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "max_tokens": 16,
  "temperature": 0.7,
  "num_queries": 5
}
```

---

## Summary

| Mode | Quality Score | Similarity | ROUGE-L | Latency (s) | Layers Skipped | Compute Reduction |
|------|--------------|------------|---------|-------------|----------------|-------------------|
| conservative | 0.2547 | 0.2395 | 0.2700 | 8.1336 | 2.0 | 9.1% |
| balanced | 0.0681 | 0.0480 | 0.0883 | 8.8712 | 3.0 | 13.6% |
| aggressive | 0.0379 | 0.0265 | 0.0493 | 7.0839 | 3.0 | 13.6% |

---

## Per-Query Details

### Mode: conservative

| Query | Quality Score | Jaccard | ROUGE-L | Latency (s) | Skipped | Reduction |
|-------|--------------|---------|---------|-------------|---------|-----------|
| What is 2+2?... | 1.0000 | 1.0000 | 1.0000 | 2.5905 | 2 | 9.1% |
| What is the capital of France?... | 0.0000 | 0.0000 | 0.0000 | 15.2238 | 2 | 9.1% |
| Explain REST API... | 0.0000 | 0.0000 | 0.0000 | 7.2459 | 2 | 9.1% |
| Write Python binary search... | 0.2103 | 0.1538 | 0.2667 | 6.6103 | 2 | 9.1% |
| Explain transformer attention... | 0.0634 | 0.0435 | 0.0833 | 8.9973 | 2 | 9.1% |

### Mode: balanced

| Query | Quality Score | Jaccard | ROUGE-L | Latency (s) | Skipped | Reduction |
|-------|--------------|---------|---------|-------------|---------|-----------|
| What is 2+2?... | 0.0000 | 0.0000 | 0.0000 | 8.0238 | 3 | 13.6% |
| What is the capital of France?... | 0.0000 | 0.0000 | 0.0000 | 10.0572 | 3 | 13.6% |
| Explain REST API... | 0.0693 | 0.0476 | 0.0909 | 9.5362 | 3 | 13.6% |
| Write Python binary search... | 0.1235 | 0.0870 | 0.1600 | 9.0531 | 3 | 13.6% |
| Explain transformer attention... | 0.1479 | 0.1053 | 0.1905 | 7.6855 | 3 | 13.6% |

### Mode: aggressive

| Query | Quality Score | Jaccard | ROUGE-L | Latency (s) | Skipped | Reduction |
|-------|--------------|---------|---------|-------------|---------|-----------|
| What is 2+2?... | 0.0000 | 0.0000 | 0.0000 | 7.7708 | 3 | 13.6% |
| What is the capital of France?... | 0.0000 | 0.0000 | 0.0000 | 6.6879 | 3 | 13.6% |
| Explain REST API... | 0.1288 | 0.0909 | 0.1667 | 7.6088 | 3 | 13.6% |
| Write Python binary search... | 0.0608 | 0.0417 | 0.0800 | 6.6600 | 3 | 13.6% |
| Explain transformer attention... | 0.0000 | 0.0000 | 0.0000 | 6.6921 | 3 | 13.6% |

---

## Analysis

- **Baseline avg latency**: 26.6580 s
- **conservative**: quality=0.2547, reduction=9.1%, latency_speedup=69.5%
- **balanced**: quality=0.0681, reduction=13.6%, latency_speedup=66.7%
- **aggressive**: quality=0.0379, reduction=13.6%, latency_speedup=73.4%

### Quality vs. Compute Trade-off

| Mode | Quality Preserved | Compute Saved |
|------|------------------|---------------|
| conservative | 25.5% | 9.1% |
| balanced | 6.8% | 13.6% |
| aggressive | 3.8% | 13.6% |