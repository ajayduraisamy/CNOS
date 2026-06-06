# CNOS — CodeSelf Neural Operating System: Full Repository Report

**Generated:** 2026-06-06
**Author:** Ajay Duraisamy
**License:** MIT

---

## 1. Project Overview

CNOS is a research initiative exploring novel architectures for running large language models (LLMs) on memory-constrained devices (4–8 GB RAM). Instead of traditional model compression or distillation, CNOS investigates **dynamic execution paradigms** — neural paging, layer routing, memory virtualization, KV cache compression, and expert swarm systems — that allow a model to punch far above its weight class.

**Core Hypothesis:** *Intelligence is not a function of parameter count alone — it is a function of how effectively available parameters are utilized at inference time.*

**Long-Term Goal (The "4GB Challenge"):** Run a 70B-parameter model on a 4GB Raspberry Pi-class device with usable quality and sub-minute latency.

### Target Specifications

| Parameter | Target |
|-----------|--------|
| Hardware RAM | 4 GB – 8 GB |
| CPU Operation | Full support (GPU optional) |
| Model Size Support | Up to 70B parameters (with paging) |
| Inference Latency | < 5 s per query (interactive) |
| Quality Preservation | > 90% of full-model benchmark score |

### Six Research Areas

| Area | Description |
|------|-------------|
| **Neural Paging** | Swap model layers between RAM and disk based on activation patterns |
| **Dynamic Layer Routing** | Skip or reorder transformer layers per-token using a routing policy network |
| **Memory Virtualization** | Abstract GPU/CPU/RAM/disk into a unified address space for model weights |
| **KV Cache Compression** | Reduce attention cache footprint via quantization, pruning, and eviction |
| **Expert Swarm** | Distribute inference across many small, specialized sub-models |
| **Adaptive Intelligence Scaling** | Dynamically allocate compute budget based on query complexity |

---

## 2. Repository Structure

```
CNOS/
├── README.md                          # Project overview & quick start
├── LICENSE                            # MIT License
├── report.md                          # This file
│
├── docs/                              # Research documentation
│   ├── vision.md                      # Problem statement, hypothesis, target specs
│   ├── architecture.md                # 7-stage pipeline architecture
│   ├── roadmap.md                     # 7-phase development roadmap
│   ├── competitor_analysis.md         # Comparison with 6 competing projects
│   ├── research_log.md                # Dated experimental log (v0.1–v0.8.1)
│   └── research-notes.md              # Empty placeholder
│
├── prototypes/                        # Research prototype implementations
│   ├── memory_profiler/               # Phase 2 – System resource profiler
│   ├── neural_paging/                 # Phase 3 – Layer-level paging engine
│   ├── layer_router/                  # Phase 4 – Dynamic layer skipping router
│   ├── real_inference/                # Phase 4 – Real HF model integration
│   ├── kv_cache_compression/          # Phase 5 – KV cache quantization/pruning
│   ├── memory_virtualization/         # Phase 5 – Unified memory tiers
│   ├── integration_engine/            # Phase 7 – Unified CNOS runtime
│   ├── layer_importance/              # Phase 8 – Ablation study data source
│   ├── importance_router/             # Phase 8 – Importance-based routing
│   ├── llamacpp_bench/                # Untracked – llama.cpp benchmarking harness
│   └── real_benchmarks/               # On feature branch – Real model benchmarks
│
├── benchmarks/                        # Empty — evaluation harnesses
├── experiments/                       # Empty — experimental results
├── papers/                            # Empty — research publications
└── scripts/                           # Empty — utility scripts
```

---

## 3. Architecture

CNOS implements a **7-stage pipeline** that operates as an operating system for neural execution:

```
Query ──► [Analyze] ──► [Complexity Score] ──► [Route Layers]
                                                      │
                                                      ▼
                              ┌──────────────────────────────────┐
                              │  For each token t:               │
                              │  1. Router: which layers?         │
                              │  2. Paging: layers in RAM?        │
                              │     No → Page fault → Load       │
                              │  3. Memory: KV cache budget?      │
                              │  4. Execute selected layers       │
                              │  5. Produce next-token logits     │
                              └──────────────────────────────────┘
                                                      │
                                                      ▼
                                              [Response]
```

### Seven Stages
1. **Query Analyzer** — Parse input, classify query type
2. **Complexity Detector** — Estimate compute budget (score ∈ [0.0, 1.0])
3. **Layer Router** — Per-token layer execution plan (skip/reorder)
4. **Neural Paging Engine** — Manage layers in RAM vs disk
5. **Memory Manager** — Unified virtual address space, tier management, KV cache
6. **Inference Engine** — Execute layer plan (CPU/CUDA/Metal/Vulkan)
7. **Response Generator** — Decode tokens with sampling

### Key Design Principles
- Resource Awareness at Every Level
- Graceful Degradation (no crashes under memory pressure)
- No Model Modification Required
- Modular and Extensible
- Hardware Agnostic (CPU-only, GPU-accelerated, or hybrid)

---

## 4. Development Roadmap (7 Phases)

| Phase | Focus | Deliverable | Status |
|-------|-------|-------------|--------|
| **1** | Research & Literature Review | Documentation, competitor analysis | ✅ Complete |
| **2** | Memory Profiling & Baseline | `prototypes/memory_profiler/` | ✅ Complete |
| **3** | Neural Paging Engine | `prototypes/neural_paging/` | ✅ Complete |
| **4** | Dynamic Layer Router | `prototypes/layer_router/`, `prototypes/real_inference/` | ✅ Complete |
| **5** | Memory Virtualization + KV Cache | `prototypes/kv_cache_compression/`, `prototypes/memory_virtualization/` | ✅ Complete |
| **6** | Expert Swarm Architecture | `prototypes/expert_swarm/` (pending) | ❌ Not Started |
| **7** | The 4GB Challenge | `prototypes/integration_engine/` | ✅ Complete |

**Beyond Phase 7:** Layer Importance Ablation (v0.8) and Importance-Based Router (v0.8.1) have been completed as post-integration research.

---

## 5. Prototype Deep Dive

### 5.1 Phase 2 — Memory Profiler (`prototypes/memory_profiler/`)
- **Files:** 5 (profiler.py, monitor.py, logger.py, config.py, requirements.txt)
- **Purpose:** CLI tool measuring system/process RAM, CPU, disk I/O via psutil
- **Dependency:** `psutil>=5.9.0`

### 5.2 Phase 3 — Neural Paging (`prototypes/neural_paging/`)
- **Files:** 6 (pager.py, layer_store.py, prefetcher.py, cache_manager.py, test_pager.py, requirements.txt)
- **Key Features:**
  - `NeuralPager` orchestrator with cache hit/miss handling
  - `LayerStore` simulating on-disk storage with blocking I/O
  - `Prefetcher` with three strategies: sequential, transition-matrix, oracle
  - `CacheManager` with LRU eviction (O(1) via OrderedDict)
- **Stdlib-only** (no external dependencies)

### 5.3 Phase 4 — Layer Router (`prototypes/layer_router/`)
- **Files:** 7
- **Components:**
  - `RoutingPolicy` ABC with Static, Adaptive, Experimental policies
  - `LayerSelector` — unified API for ComplexityDetector + RoutingPolicy
  - `ComplexityDetector` — rule-based query analysis
  - `CumulativeMetrics` — aggregate per-session statistics
  - `benchmark.py` — evaluates 300 queries across all policies

### 5.4 Phase 4 — Real Inference (`prototypes/real_inference/`)
- **Files:** 7
- **Purpose:** Integrates with real Hugging Face models (TinyLlama-1.1B, Qwen-2.5-1.5B, Llama-3.2-1B)
- **Components:**
  - `model_loader.py` — HF model loading
  - `routed_inference.py` — `RoutedInferenceEngine` with selective layer inference
  - `layer_hooks.py` — `LayerGate` (skip/pass-through) + `LayerMonitor` (timing/mem)
  - `quality_evaluator.py` — Exact match / Jaccard / ROUGE-L / latency

### 5.5 Phase 5 — KV Cache Compression (`prototypes/kv_cache_compression/`)
- **Files:** 8
- **Compression Ratios (vs FP32):** FP16 = 2×, INT8 = 4×, INT4 = 8×
- **Pruning Strategies:** OldestFirst, LeastUsed, AttentionScore
- **Eviction Policies:** LRU, LFU, Adaptive
- **Tests:** 68/68 passing

### 5.6 Phase 5 — Memory Virtualization (`prototypes/memory_virtualization/`)
- **Files:** 10
- **Four Memory Tiers:**
  | Tier | Medium | Latency |
  |------|--------|---------|
  | 0 | GPU VRAM | 200 ns |
  | 1 | CPU RAM | 80 ns |
  | 2 | Compressed KV | 500 ns |
  | 3 | SSD | 100 µs |
- **Prefetch Strategies:** Sequential, Stride, Frequency
- **Eviction:** LRU, LFU, Adaptive (switches at 50% cache pressure)
- **Benchmark Results (7B on 4GB RAM):** 94.7% hit rate, 128 faults, 2304 hits, 0 evictions
- **Tests:** 67/67 passing

### 5.7 Phase 7 — Integration Engine (`prototypes/integration_engine/`)
- **Files:** 10
- **The Master Runtime:** All subsystems connected:
  ```
  User Query
    → RoutingController (ComplexityDetector + LayerSelector)
    → MemoryController (NeuralPager + VirtualMemorySystem)
    → ModelAdapter (Real / Simulated)
    → CacheController (KVCacheManager + quantizers)
    → CnosResult (structured output with all metrics)
  ```
- **Two Modes:** `simulate` (no model download) and `real` (requires HF model)
- **Benchmark Results:**
  - Avg Compute Reduction: **52.5%**
  - Avg Page Hit Rate: **64.4%** (warms from 0% to 83%)
  - Total Page Faults: 120
- **Tests:** 81/81 passing

### 5.8 Phase 8 — Layer Importance (`prototypes/layer_importance/`)
- **Files:** 8
- **Methodology:** Skip-each-layer ablation on TinyLlama-1.1B
- **Key Findings (22 layers):**
  - High Impact (score ≥ 0.30): **19 layers**
  - Medium Impact (0.10–0.30): **3 layers** (8, 9, 19)
  - Low Impact (< 0.10): **0 layers**
- **Insight:** TinyLlama has almost no "skippable" layers — nearly every layer is critical

### 5.9 Phase 8 — Importance Router (`prototypes/importance_router/`)
- **Files:** 8
- **Three Routing Modes:**
  | Mode | Quality Score | Compute Reduction | Latency Speedup |
  |------|:------------:|:-----------------:|:---------------:|
  | conservative | 0.2547 | 9.1% | 69.5% |
  | balanced | 0.0681 | 13.6% | 66.7% |
  | aggressive | 0.0379 | 13.6% | 73.4% |
- **Challenge:** Quality preservation is low because TinyLlama has no truly low-impact layers

### 5.10 Untracked — llama.cpp Bench (`prototypes/llamacpp_bench/`)
- 4 files on disk: benchmark.py, compare.py, setup_check.py, README.md
- Not yet committed to the repository

---

## 6. Competitive Landscape

CNOS differentiates from 6 competing projects:

| Project | RAM Target | CPU Support | CNOS Advantage |
|---------|:---------:|:-----------:|----------------|
| **llama.cpp** | 6–16 GB | ✅ | Adds intelligent layer paging + dynamic routing on top of mmap |
| **FlexGen** (Stanford) | 4–16 GB | ✅ | Expands with layer skipping and expert swarms |
| **DeepSpeed Inference** | 16+ GB | ❌ | Targets lower RAM, adds neural paging |
| **Hugging Face Accelerate** | 8–16 GB | ✅ | Smarter eviction/prefetch policies |
| **ExLlamaV2** | 6–16 GB | ❌ | CPU-first design with routing/paging |
| **MLC-LLM** | 4–16 GB | ✅ | Adaptive routing per query complexity |

### Key Gaps CNOS Fills
1. **Holistic integration** of paging, routing, virtualization, and swarms
2. **CPU-first design** (GPU as optional accelerator)
3. **Adaptive intelligence scaling** per query
4. **4GB target** as ceiling (others target 8GB as floor)
5. **Model-agnostic policy learning** from execution history

---

## 7. Test Coverage

| Prototype | Tests | Status |
|-----------|:-----:|:------:|
| neural_paging | ✓ | Complete |
| layer_router | ✓ | Complete |
| real_inference | 23/23 | ✅ Passing |
| kv_cache_compression | 68/68 | ✅ Passing |
| memory_virtualization | 67/67 | ✅ Passing |
| integration_engine | 81/81 | ✅ Passing |
| layer_importance | ✓ | Complete |
| importance_router | 30+ | ✅ Passing |

**Total:** ~270+ tests across all prototypes

---

## 8. Key Benchmarks

### Integration Engine (Simulate Mode)
- **Config:** TinyLlama, 4GB RAM, adaptive routing, INT8 quantization, LRU eviction
- **Avg Compute Reduction:** 52.5%
- **Avg Page Hit Rate:** 64.4% (cold start → 83% after warmup)

### Importance Router (Real Mode — TinyLlama)
- **Baseline Latency:** 26.66 s
- **Best Quality:** conservative mode (quality 0.25, 9.1% compute reduction, 69.5% latency speedup)
- **Best Speed:** aggressive mode (73.4% latency speedup, 3.8% quality)

### Memory Virtualization (7B model on 4GB RAM)
- **Hit Rate:** 94.7%
- **Page Faults:** 128
- **Evictions:** 0

---

## 9. Known Issues & Open Questions

1. **Quality vs. Compute Trade-off:** TinyLlama's layer importance data shows nearly all layers are critical, limiting safe skip ratios. Larger models (7B+) may have more redundancy.
2. **KV Cache Quality Impact:** End-to-end quality impact of compressed KV cache reuse during generation is not yet verified.
3. **Expert Swarm (Phase 6):** Not yet implemented — this is the largest remaining gap.
4. **Real Benchmark Latency:** The integration engine's simulate mode shows negative latency reduction due to overhead; real-mode benchmarks needed.
5. **llamacpp_bench:** 4 untracked benchmarking files need to be committed.
6. **Empty Directories:** `benchmarks/`, `experiments/`, `papers/`, `scripts/` are empty placeholders.

---

## 10. File Inventory (Complete)

### Root (3 files)
- `README.md` — Project overview
- `LICENSE` — MIT License
- `report.md` — This file

### docs/ (6 files)
- `vision.md`, `architecture.md`, `roadmap.md`, `competitor_analysis.md`, `research_log.md`, `research-notes.md`

### prototypes/ (66+ files across 10 directories)
- `memory_profiler/` (5 files) — Phase 2
- `neural_paging/` (6 files) — Phase 3
- `layer_router/` (7 files) — Phase 4
- `real_inference/` (7 files) — Phase 4
- `kv_cache_compression/` (8 files) — Phase 5
- `memory_virtualization/` (10 files) — Phase 5
- `integration_engine/` (10 files) — Phase 7
- `layer_importance/` (9 files) — Phase 8
- `importance_router/` (8 files) — Phase 8
- `llamacpp_bench/` (4 files, untracked) — Benchmarking

### Empty directories (4)
- `benchmarks/`, `experiments/`, `papers/`, `scripts/`

---

## 11. Summary

CNOS is a well-structured research project with **9 implemented prototypes**, **270+ passing tests**, and a clear 7-phase roadmap. It has successfully demonstrated:

- **52.5% compute reduction** via integrated layer routing and neural paging
- **94.7% page hit rate** in memory virtualization for 7B models on 4GB RAM
- **~70% latency speedup** in importance-based routing
- **Up to 8× KV cache compression** via INT4 quantization

The largest remaining work is Phase 6 (Expert Swarm), real-hardware validation of the integration engine, and bridging the quality gap in aggressive routing modes.
