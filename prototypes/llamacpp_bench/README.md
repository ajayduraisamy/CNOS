# llama.cpp Benchmark Harness — CNOS Research

Benchmark `llama-cli.exe` / `llama-bench.exe` (CPU) on GGUF-quantized models.
Compare results with CNOS PyTorch inference (`real_inference`).

## Environment

| Component | Detail |
|-----------|--------|
| CPU | Intel Core i3-7130U @ 2.70 GHz (2C/4T) |
| RAM | 7.9 GB total (~1.2 GB free) |
| OS | Windows 11 |
| Python | 3.10.0 (AMD64) |
| ISA | AVX2 supported (Kaby Lake) |

## Setup

### 1. Install llama.cpp

```powershell
# Clone
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CMake (requires Visual Studio Build Tools)
cmake -B build
cmake --build build --config Release

# Binaries will be in: build\bin\Release\
# Add to PATH or copy to project root:
$env:Path += ";$pwd\build\bin\Release"
```

**Requirements:**
- Git
- CMake 3.15+
- Visual Studio 2022 (or Build Tools) with C++ workload

### 2. Verify installation

```powershell
python prototypes/llamacpp_bench/setup_check.py
```

This prints a Markdown environment report confirming that `llama-cli.exe`
and `llama-bench.exe` are accessible and AVX2 is active.

### 3. Download a GGUF model

```powershell
# TinyLlama (1.1B, Q4_K_M, ~700 MB)
# Download from:
# https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
# Save as: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## Benchmark

```powershell
# Quick check (1 query, 32 tokens)
python prototypes/llamacpp_bench/benchmark.py --model tinyllama --max-tokens 32 --queries 1

# Full benchmark (5 queries, 128 tokens)
python prototypes/llamacpp_bench/benchmark.py --model tinyllama --max-tokens 128

# Custom model path
python prototypes/llamacpp_bench/benchmark.py --model-path D:\models\my-model.gguf
```

Results saved to `prototypes/llamacpp_bench/output/llamacpp_bench_results.json`.

## Compare with CNOS

```powershell
python prototypes/llamacpp_bench/compare.py ^
    --cnos prototypes/integration_engine/output/benchmark_results.json ^
    --llama prototypes/llamacpp_bench/output/llamacpp_bench_results.json ^
    --output prototypes/llamacpp_bench/output/comparison_report.md
```

## Files

| File | Purpose |
|------|---------|
| `setup_check.py` | Environment readiness report |
| `benchmark.py` | Run llama-cli.exe on CNOS test queries |
| `compare.py` | Side-by-side CNOS vs llama.cpp comparison |
| `README.md` | This file |

## Model Registry

| Key | Model | Params | GGUF File | RAM Est. |
|-----|-------|--------|-----------|----------|
| tinyllama | TinyLlama-1.1B-Chat-v1.0 | 1.1B | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | 1.5 GB |
| llama-3.2-1b | Llama-3.2-1B-Instruct | 1B | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 1.2 GB |
| qwen-2.5-1.5b | Qwen2.5-1.5B-Instruct | 1.5B | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 2.0 GB |

## Notes

- CPU-only: always use `--n-gpu-layers 0` (default).
- With only 1.2 GB free RAM, the 1.5B model (Qwen) may cause swapping.
- Q4_K_M quantization offers the best quality/speed trade-off on CPU.
- For CNOS comparison, use the same `--max-tokens` and queries as the CNOS benchmark.
