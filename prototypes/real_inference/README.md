# CNOS Real Transformer Integration — v0.4

> **Connects the Dynamic Layer Router to a real HuggingFace transformer model.**

This prototype proves that selective layer execution works on a real
neural network — not just in simulation.  It wraps each decoder layer
in a `LayerGate` that can dynamically enable or disable computation,
then measures latency, memory, and output quality.

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ Complexity      │────▶│ Layer Router     │────▶│ LayerGate[]          │
│ Detector (v0.3) │     │ (v0.3 policy)    │     │ (installed in model) │
└─────────────────┘     └──────────────────┘     └──────────────────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────────┐
                                              │ Real Transformer     │
                                              │ (TinyLlama / Qwen /  │
                                              │  Llama 3.2 1B)       │
                                              └──────────────────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────────┐
                                              │ Quality Evaluator    │
                                              │ (Jaccard / ROUGE-L)  │
                                              └──────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `model_loader.py` | Loads HuggingFace models with auto device/dtype detection |
| `layer_hooks.py` | `LayerGate` wrapper + `LayerMonitor` for per-layer statistics |
| `routed_inference.py` | `RoutedInferenceEngine` — generate with selective layers + metrics |
| `quality_evaluator.py` | Compares baseline vs. routed output (Jaccard, ROUGE-L, exact match) |
| `benchmark_real.py` | Runs 12 test queries through both modes and aggregates results |
| `test_real_inference.py` | Unit tests for all components (no GPU required) |

## How It Works

### LayerGate

Each transformer decoder layer is wrapped in a `LayerGate`:

```python
gate = LayerGate(original_layer, layer_id)
```

When `gate.active = True`, the original layer runs normally.
When `gate.active = False`, the gate returns a **passthrough**:
the hidden states pass through unchanged and the KV cache is not updated.

### LayerMonitor

A `LayerMonitor` hooks into each gate to track:
- Number of calls (active vs. skipped)
- Wall-clock time per layer
- Peak CUDA memory per layer

### RoutedInferenceEngine

High-level API that accepts a set of active layer IDs:

```python
engine = RoutedInferenceEngine(bundle)
response, metrics = engine.generate(query, active_layers={0, 1, 5, 10, 15})
```

## Usage

### Prerequisites

```bash
pip install torch transformers
```

### Run the test suite (no GPU required)

```bash
cd prototypes/real_inference
python test_real_inference.py
```

### Run the benchmark (requires model download)

```bash
python benchmark_real.py --model tinyllama --max-tokens 64
```

### Compare specific queries

```python
from model_loader import load_model
from routed_inference import RoutedInferenceEngine

bundle = load_model("tinyllama")
engine = RoutedInferenceEngine(bundle)

# Baseline (all 22 layers)
baseline_resp, baseline_metrics = engine.generate_baseline("What is 2+2?")

# Routed (12 layers only)
plan = {0, 1, 2, 3, 4, 10, 11, 12, 18, 19, 20, 21}
routed_resp, routed_metrics = engine.generate_routed("What is 2+2?", plan)

engine.print_comparison(baseline_metrics, routed_metrics)
```

## Supported Models

| Model Key | HuggingFace ID | Layers |
|-----------|---------------|--------|
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 22 |
| `qwen-1.5b` | Qwen/Qwen2.5-1.5B-Instruct | 28 |
| `llama-3.2-1b` | meta-llama/Llama-3.2-1B-Instruct | 16 |

## Memory Notes

TinyLlama (1.1B parameters) requires ~2.2 GB in fp16 or ~4.4 GB in fp32.
On systems with < 8 GB RAM, use `--model tinyllama` with fp16 (default
on CUDA).  For CPU-only, the loader automatically selects fp32.

## Quality Metrics

- **Exact match** — useful for factual queries (e.g., "What is 2+2?")
- **Token Jaccard** — set overlap of decoded tokens
- **ROUGE-L F1** — longest common subsequence (word-level)
- **Latency reduction** — wall-clock speedup from layer skipping

## Dependencies

- Python 3.11+
- PyTorch (≥ 2.0)
- transformers (≥ 4.30)
- No other external dependencies
