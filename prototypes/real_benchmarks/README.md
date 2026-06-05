# Real Benchmark Suite

End-to-end benchmarking of CNOS optimizations on real HuggingFace models.

## Purpose

Measure actual RAM, CPU, latency, and quality improvements from CNOS's three core optimization subsystems when applied to real transformer inference:

1. **Dynamic Layer Routing** -- skip layers based on query complexity
2. **Memory Virtualization** -- page-level memory tracking and eviction
3. **KV Cache Compression** -- FP16/INT8/INT4 quantisation

## Quick Start

```bash
# Run full benchmark (TinyLlama, adaptive routing, INT8, 4 GB simulated RAM)
python prototypes/real_benchmarks/benchmark_suite.py --model tinyllama --max-tokens 64
```

## Usage

```
python prototypes/real_benchmarks/benchmark_suite.py [options]

Options:
  --model           Model key: tinyllama (default) or qwen-1.5b
  --max-tokens      Max tokens to generate per query (default: 64)
  --routing-policy  Layer routing policy (default: adaptive)
  --quantisation    KV cache quantisation (default: int8)
  --ram-gb          Simulated RAM in GB for memory virtualization (default: 4.0)
  --device          Device override: cpu, cuda, mps (default: auto)
  --timeout         Model download timeout in seconds (default: 300)
  --no-save         Skip saving results to disk
```

## Output

All results go to `prototypes/real_benchmarks/output/`:

| File | Format | Content |
|------|--------|---------|
| `benchmark_report.md` | Markdown | Human-readable summary + per-query details |
| `benchmark_results.csv` | CSV | Per-query rows for spreadsheet analysis |
| `benchmark_results.json` | JSON | Full data for programmatic consumption |
| `baseline_results.json` | JSON | Raw baseline per-query results |
| `cnos_results.json` | JSON | Raw CNOS per-query results |

## Architecture

```
benchmark_suite.py          -- CLI entry point, orchestrator
benchmark_loader.py         -- Model loading with timing & memory measurement
baseline_runner.py          -- Standard HuggingFace inference
cnos_runner.py              -- CNOS-optimized inference (routing + paging + KV)
metrics_collector.py        -- Aggregates comparison metrics
report_generator.py         -- Markdown / CSV / JSON output
test_real_benchmarks.py     -- Unit tests (no model download needed)
```

## Running Tests

```bash
python prototypes/real_benchmarks/test_real_benchmarks.py
```

Tests use `SimulatedModelAdapter` and synthetic data -- no HuggingFace download required.

## Models

| Key | HuggingFace ID | Layers | Size |
|-----|---------------|--------|------|
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 22 | ~2.2 GB |
| `qwen-1.5b` | Qwen/Qwen2.5-1.5B-Instruct | 28 | ~3.0 GB |
