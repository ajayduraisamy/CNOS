# CNOS v0.7 Integration Engine

Unified runtime that integrates all CNOS subsystems into a single processing pipeline:

- **Neural Paging**  layer-level LRU cache with prefetching
- **Dynamic Layer Router**  query complexity analysis  selective layer execution
- **KV Cache Compression**  FP16 / INT8 / INT4 quantisation with pruning and eviction
- **Memory Virtualization**  page-level memory across GPU  RAM  CompKV  SSD tiers

## Architecture

```
User Query
    |
    v
RoutingController        analyses complexity, selects layers
    |
    v
MemoryController         ensures selected layers are in fast memory
    |
    v
ModelAdapter             runs inference (real or simulated)
    |
    v
CacheController          compresses KV cache entries
    |
    v
CnosResult               structured output with all metrics
```

## Files

| File | Description |
|---|---|
| `runtime.py` | Master coordinator + `CnosRuntime` + `RuntimeConfig` + `CnosResult` |
| `model_adapter.py` | `ModelAdapter` ABC + `RealModelAdapter` + `SimulatedModelAdapter` |
| `routing_controller.py` | `RoutingController` wrapping ComplexityDetector + LayerSelector |
| `memory_controller.py` | `MemoryController` bridging NeuralPager + VirtualMemorySystem |
| `cache_controller.py` | `CacheController` wrapping KVCacheManager + quantizers |
| `benchmark.py` | `Benchmark` comparing Baseline vs CNOS; Markdown/CSV/JSON reports |
| `test_runtime.py` | 26 unit tests covering all controllers and the full pipeline |

## Quick Start

```python
from runtime import CnosRuntime, RuntimeConfig

rt = CnosRuntime(RuntimeConfig(
    model_key="tinyllama",
    ram_gb=4,
    mode="simulate",     # no model download needed
))
result = rt.process("What is the capital of France?")
print(result.response)
print(result.to_dict())
rt.cleanup()
```

## Run Benchmark

```bash
python benchmark.py
```

Reports are written to `prototypes/integration_engine/output/`:
- `benchmark_report.md`
- `benchmark_results.csv`
- `benchmark_results.json`

## Run Tests

```bash
python test_runtime.py
```

## Two Modes

### Simulate (default)

No model download required. Uses `neural_paging` components to simulate
layer execution, memory pressure, and KV cache operations. All tests
and the benchmark use this mode by default.

### Real

Requires a HuggingFace model download (TinyLlama~2.2GB or Qwen 1.5B~3GB).

```python
rt = CnosRuntime(RuntimeConfig(model_key="tinyllama", mode="real"))
```

## Supported Models

| Key | Model | Layers | Size |
|---|---|---|---|
| `tinyllama` | TinyLlama-1.1B-Chat-v1.0 | 22 | ~2.2 GB |
| `qwen-1.5b` | Qwen2.5-1.5B-Instruct | 28 | ~3.0 GB |

## Metrics Collected

Per-query:
- Latency (end-to-end, baseline vs CNOS)
- Layers executed / skipped
- Compute reduction percentage
- RAM usage (GB)
- Page faults / page hits / hit rate
- KV cache compression ratio
- Tokens generated

Aggregate:
- Average latency reduction
- Average compute reduction
- Average page hit rate
- Total faults and hits

## Requirements

- Python 3.10.0
- PyTorch 2.x (for real mode and KV cache simulation)
- Transformers 5.x (for real mode)
