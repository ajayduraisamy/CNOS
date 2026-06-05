# KV Cache Compression Engine (v0.5)

Reduces transformer KV cache memory consumption via quantisation, pruning,
and intelligent eviction.

## Architecture

```
benchmark.py     ─── runs simulations across config matrix
     │
     ▼
kv_cache.py      ─── KVCacheManager (per-layer key/value store)
     │
     ├── quantizer.py    ─── FP16 / INT8 / INT4 quantisation
     ├── pruner.py       ─── oldest-first / least-used / attention-score
     └── eviction_policy.py ─── LRU / LFU / Adaptive selection
              │
              ▼
         metrics.py  ─── CompressionRecord + CompressionMetrics (aggregation)
```

## Components

| Module | Path | Responsibility |
|--------|------|----------------|
| `kv_cache.py` | `KVCacheEntry`, `KVCacheManager` | Per-layer key/value store with position tracking and memory accounting |
| `quantizer.py` | `FP16Quantizer`, `INT8Quantizer`, `INT4Quantizer` | Tensor quantisation round-trip with scale/zero-point metadata |
| `pruner.py` | `OldestFirstPruner`, `LeastUsedPruner`, `AttentionScorePruner` | Token-position removal strategies |
| `eviction_policy.py` | `LRUPolicy`, `LFUPolicy`, `AdaptivePolicy` | Per-layer eviction selection under cache pressure |
| `metrics.py` | `CompressionRecord`, `CompressionMetrics` | Result storage and comparison table generation |
| `benchmark.py` | `run_benchmark()` | Multi-config simulation across token-length scales |
| `test_kv_cache.py` | 15+ test suites | Unit tests for all components |

## Quantisation Methods

| Scheme | Bits/elem | Compression (vs FP32) | Quality |
|--------|-----------|----------------------|---------|
| FP16   | 16        | 2×                    | Lossless (no rounding) |
| INT8   | 8         | 4×                    | ~0.1% MSE |
| INT4   | 4         | 8×                    | ~1% MSE |

## Usage

```python
from kv_cache import KVCacheManager

mgr = KVCacheManager(num_layers=22, num_heads=32, head_dim=64)
k, v = torch.randn(32, 1, 64), torch.randn(32, 1, 64)
mgr.append(layer_idx=0, key=k, value=v)
print(mgr.total_memory_mb, "MB")
```

Run benchmark:

```bash
python prototypes/kv_cache_compression/benchmark.py --tokens 1000 5000 10000 20000 --verbose
```

Run tests:

```bash
python prototypes/kv_cache_compression/test_kv_cache.py
```

## Key Design Decisions

- **Analytical memory** — memory is computed from tensor shapes and dtypes
  rather than measured via `torch.cuda.memory_allocated()`, making it
  device-independent and reproducible in CI without a GPU.
- **Synthetic data** — benchmarks use `torch.randn` key/value tensors;
  no real model is needed for cache-level measurements.
- **Separate quantiser/pruner/eviction** — the three concerns are decoupled
  so any quantisation scheme can be combined with any pruner and eviction policy.
- **INT4 bit-packing** — two INT4 values are packed per byte with sign
  extension for the [-8, 7] range, giving 8× compression vs FP32.
