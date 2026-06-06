# Layer Importance Study — tinyllama

**Date:** 2026-06-05T16:20:15  
**Total time:** 311.5 s  
**Configuration:** {
  "mode": "real",
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "max_tokens": 16,
  "temperature": 0.7,
  "num_queries": 3
}

---

## Summary

- **Total layers analyzed:** 22
- **High impact layers** (score ≥ 0.30): 19 — [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21]
- **Medium impact layers** (score 0.10–0.30): 3 — [8, 9, 19]
- **Low impact layers** (score < 0.10): 0 — []

### Interpretation

- **High impact** — Disabling this layer causes significant quality loss.
  These layers are critical and should NOT be skipped in routing.
- **Medium impact** — Some quality degradation; may be safe to skip
  under aggressive routing budgets.
- **Low impact** — Minimal quality loss when disabled; good candidates
  for skipping during inference.

---

## Per-Layer Impact Scores

| Layer | Avg Impact Score | Classification | Queries |
|-------|-----------------|----------------|---------|
| 0 | 0.9769 | high | 3 |
| 1 | 0.6667 | high | 3 |
| 2 | 1.0000 | high | 3 |
| 3 | 0.6425 | high | 3 |
| 4 | 0.6667 | high | 3 |
| 5 | 0.6399 | high | 3 |
| 6 | 1.0000 | high | 3 |
| 7 | 0.6472 | high | 3 |
| 8 | 0.2706 | medium | 3 |
| 9 | 0.2953 | medium | 3 |
| 10 | 0.5155 | high | 3 |
| 11 | 0.6347 | high | 3 |
| 12 | 0.6486 | high | 3 |
| 13 | 1.0000 | high | 3 |
| 14 | 0.9373 | high | 3 |
| 15 | 0.3146 | high | 3 |
| 16 | 0.6255 | high | 3 |
| 17 | 0.3333 | high | 3 |
| 18 | 0.3333 | high | 3 |
| 19 | 0.2444 | medium | 3 |
| 20 | 0.6667 | high | 3 |
| 21 | 0.9421 | high | 3 |

---

## Classification

| Category | Count | Layers |
|----------|-------|--------|
| High Impact | 19 | 0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21 |
| Medium Impact | 3 | 8, 9, 19 |
| Low Impact | 0 | none |

---

## Per-Query Details

### Layer 0  (HIGH, score=0.9769)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 2.0000 | 9.1362 | 1.0000 |
| What is the capital of France?... | 0.0000 | 0.0000 | 2.0000 | 6.5063 | 1.0000 |
| Explain REST API... | 0.0476 | 0.0909 | 0.9333 | 7.1019 | 0.9307 |

### Layer 1  (HIGH, score=0.6667)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 2.0000 | 6.8891 | 1.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.1542 | 0.0000 |
| Explain REST API... | 0.0000 | 0.0000 | 0.4667 | 4.8898 | 1.0000 |

### Layer 2  (HIGH, score=1.0000)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 1.0000 | 6.6161 | 1.0000 |
| What is the capital of France?... | 0.0000 | 0.0000 | 2.0000 | 6.3854 | 1.0000 |
| Explain REST API... | 0.0000 | 0.0000 | 0.6667 | 6.6199 | 1.0000 |

### Layer 3  (HIGH, score=0.6425)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 2.0000 | 6.5050 | 1.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.0666 | 0.0000 |
| Explain REST API... | 0.0500 | 0.0952 | 0.4667 | 3.9292 | 0.9274 |

### Layer 4  (HIGH, score=0.6667)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 2.0000 | 6.5632 | 1.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.1475 | 0.0000 |
| Explain REST API... | 0.0000 | 0.0000 | 0.6667 | 7.3485 | 1.0000 |

### Layer 5  (HIGH, score=0.6399)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 2.0000 | 5.8991 | 1.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.2774 | 0.0000 |
| Explain REST API... | 0.0556 | 0.1053 | 0.3333 | 3.4239 | 0.9196 |

### Layer 6  (HIGH, score=1.0000)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 2.0000 | 5.3764 | 1.0000 |
| What is the capital of France?... | 0.0000 | 0.0000 | 2.0000 | 6.4320 | 1.0000 |
| Explain REST API... | 0.0000 | 0.0000 | 0.0667 | 1.5800 | 1.0000 |

### Layer 7  (HIGH, score=0.6472)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 1.0000 | 1.0000 | 1.0000 | 1.1312 | 0.0000 |
| What is the capital of France?... | 0.0000 | 0.0000 | 2.0000 | 6.4809 | 1.0000 |
| Explain REST API... | 0.0400 | 0.0769 | 0.8667 | 6.2003 | 0.9415 |

### Layer 8  (MEDIUM, score=0.2706)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 1.0000 | 1.0000 | 1.0000 | 1.0879 | 0.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.3200 | 0.0000 |
| Explain REST API... | 0.1364 | 0.2400 | 0.7333 | 5.8682 | 0.8118 |

### Layer 9  (MEDIUM, score=0.2953)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 1.0000 | 1.0000 | 1.0000 | 1.1008 | 0.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.2422 | 0.0000 |
| Explain REST API... | 0.0800 | 0.1481 | 0.9333 | 6.4416 | 0.8859 |

### Layer 10  (HIGH, score=0.5155)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 1.0000 | 1.0000 | 1.0000 | 1.1535 | 0.0000 |
| What is the capital of France?... | 0.0000 | 0.0000 | 2.0000 | 6.4069 | 1.0000 |
| Explain REST API... | 0.3684 | 0.5385 | 0.8000 | 6.8198 | 0.5466 |

### Layer 11  (HIGH, score=0.6347)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 2.0000 | 6.5673 | 1.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.1408 | 0.0000 |
| Explain REST API... | 0.0667 | 0.1250 | 0.1333 | 2.2837 | 0.9042 |

### Layer 12  (HIGH, score=0.6486)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 2.0000 | 6.0845 | 1.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.1505 | 0.0000 |
| Explain REST API... | 0.0370 | 0.0714 | 0.9333 | 6.2785 | 0.9458 |

### Layer 13  (HIGH, score=1.0000)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 2.0000 | 6.5254 | 1.0000 |
| What is the capital of France?... | 0.0000 | 0.0000 | 2.0000 | 6.2396 | 1.0000 |
| Explain REST API... | 0.0000 | 0.0000 | 1.0000 | 6.2420 | 1.0000 |

### Layer 14  (HIGH, score=0.9373)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 2.0000 | 4.3204 | 1.0000 |
| What is the capital of France?... | 0.0000 | 0.0000 | 2.0000 | 6.4342 | 1.0000 |
| Explain REST API... | 0.1364 | 0.2400 | 0.7333 | 7.1485 | 0.8118 |

### Layer 15  (HIGH, score=0.3146)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 1.0000 | 1.0000 | 1.0000 | 1.1289 | 0.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.2248 | 0.0000 |
| Explain REST API... | 0.0385 | 0.0741 | 0.8667 | 6.0835 | 0.9437 |

### Layer 16  (HIGH, score=0.6255)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 1.0000 | 1.0000 | 1.0000 | 1.1289 | 0.0000 |
| What is the capital of France?... | 0.0000 | 0.0000 | 2.0000 | 6.7466 | 1.0000 |
| Explain REST API... | 0.0870 | 0.1600 | 0.8000 | 7.4732 | 0.8765 |

### Layer 17  (HIGH, score=0.3333)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 1.0000 | 1.0000 | 1.0000 | 1.1316 | 0.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.0963 | 0.0000 |
| Explain REST API... | 0.0000 | 0.0000 | 0.5333 | 6.4099 | 1.0000 |

### Layer 18  (HIGH, score=0.3333)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 1.0000 | 1.0000 | 1.0000 | 1.1482 | 0.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.1432 | 0.0000 |
| Explain REST API... | 0.0000 | 0.0000 | 0.2667 | 3.2503 | 1.0000 |

### Layer 19  (MEDIUM, score=0.2444)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 1.0000 | 1.0000 | 1.0000 | 1.2948 | 0.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.1207 | 0.0000 |
| Explain REST API... | 0.2000 | 0.3333 | 0.8000 | 7.4582 | 0.7333 |

### Layer 20  (HIGH, score=0.6667)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 2.0000 | 5.0164 | 1.0000 |
| What is the capital of France?... | 1.0000 | 1.0000 | 1.0000 | 1.0849 | 0.0000 |
| Explain REST API... | 0.0000 | 0.0000 | 0.5333 | 6.7082 | 1.0000 |

### Layer 21  (HIGH, score=0.9421)

| Query | Jaccard | ROUGE-L | Length Ratio | Latency (s) | Impact Score |
|-------|---------|---------|-------------|-------------|-------------|
| What is 2+2?... | 0.0000 | 0.0000 | 2.0000 | 2.4216 | 1.0000 |
| What is the capital of France?... | 0.0000 | 0.0000 | 2.0000 | 6.5535 | 1.0000 |
| Explain REST API... | 0.1250 | 0.2222 | 0.9333 | 6.1504 | 0.8264 |


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
