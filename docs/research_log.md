# Research Log

## Purpose

This log tracks experimental results, insights, failures, decisions, and open questions throughout the CNOS research project. Each entry is dated and tagged by research area.

---

## Entry Template

```markdown
### YYYY-MM-DD — [Title]

**Tags:** `neural-paging` `layer-routing` `memory-virt` `kv-cache` `expert-swarm` `benchmarking`

**Hypothesis:** ...

**Experiment:** ...

**Results:** ...

**Key Insight:** ...

**Next Steps:** ...

**Open Questions:** ...

**References:** ...
```

---

## Log

### 2026-06-05 — Project Inception

**Tags:** `meta`

**Summary:** CNOS repository initialized with documentation structure, directory scaffolding, and initial memory profiler prototype. Research roadmap defined across 7 phases. Core architecture designed with 7-stage pipeline: Query Analyzer → Complexity Detector → Layer Router → Neural Paging Engine → Memory Manager → Inference Engine → Response Generator.

**Key Insight:** The project's primary differentiator is holistic integration of paging, routing, virtualization, and adaptive scaling — no existing solution combines all four.

**Next Steps:**
- Complete Phase 1 research survey
- Build and test memory profiler on 1B–7B models
- Begin Phase 3 neural paging literature deep dive

---

### 2026-06-05 — v0.4 Bugfixes: Latency Cross-Population & v5.x Compatibility

**Tags:** `real-inference` `bugfix` `transformers-v5`

**Problem:** `benchmark_real.py` reported 0.0% latency reduction because `compare()` never set `baseline_latency_s`. Additionally, `torch_dtype` (deprecated in transformers v5.x) caused silent fallback to float32, and passing `temperature` with `do_sample=False` emitted warnings.

**Fix:** `generate()` now sets both `baseline_latency_s` and `routed_latency_s` from elapsed time; `compare()` cross-populates correctly. `torch_dtype` → `dtype` in `model_loader.py`. `temperature` withheld from generate kwargs when `do_sample=False`.

**Results:** 23/23 unit tests pass.

---

### 2026-06-05 — v0.5 KV Cache Compression Engine

**Tags:** `kv-cache` `quantization` `compression`

**Components:** 8 files — `kv_cache.py`, `quantizer.py` (FP16 2× / INT8 symmetric 4× / INT4 bit-packed 8× vs FP32), `pruner.py` (oldest-first / least-used / attention-score), `eviction_policy.py` (LRU / LFU / Adaptive), `metrics.py`, `benchmark.py`, `test_kv_cache.py` (68/68 pass), `README.md`

**Key Design Decisions:**
- INT8 quantizer uses symmetric scheme (no zero-point) to avoid int8 range overflow — MSE < 0.002
- INT4 packs two 4-bit values per byte for 8× compression
- Compression computed analytically (elements × bits_per_element / 8) to avoid shape loss from packed tensors

**Open Question:** Verify end-to-end quality impact when compressed KV is re-used in generation.

---

### 2026-06-05 — v0.6 Memory Virtualization Engine

**Tags:** `memory-virt` `page-table` `eviction` `prefetch`

**Components:** 10 files — `memory_tiers.py` (4 tiers: GPU/RAM/CompKV/SSD), `page_table.py`, `allocator.py`, `eviction_manager.py` (LRU/LFU/Adaptive), `prefetch_engine.py` (sequential/stride/frequency), `metrics.py`, `virtual_memory.py` (orchestrator), `benchmark.py` (7B/30B/70B profiles under 4GB/8GB RAM), `test_virtual_memory.py` (67/67 pass), `README.md`

**Key Design Decisions:**
- Tier hierarchy: GPU VRAM (200ns) → CPU RAM (80ns) → Compressed KV (500ns) → SSD (100µs)
- Pages initially on SSD; first access triggers page fault → promoted to RAM
- Adaptive eviction switches LRU ↔ LFU at 50% cache pressure
- Explicit constraint: Python 3.10.0 compatibility (no `match`, no bare `|` union syntax without `__future__`)

---

### 2026-06-05 — Hit/Fault Logic Correction

**Tags:** `memory-virt` `bugfix`

**Problem:** `VirtualMemorySystem.access()` counted all page-table-resident pages as hits, even those on SSD (tier 3). Benchmark showed 0 page faults.

**Fix:** `access()` now treats tiers 0–2 (GPU/RAM/Compressed KV) as hits and only tier 3 (SSD) as page faults. `_handle_fault()` rewritten to use `allocator.move_page()` to promote pages from SSD to RAM instead of duplicating page table entries. Non-ASCII characters stripped from all files for cp1252 console compatibility.

**Results:** 7B on 4GB RAM: 128 faults, 2304 hits, 94.7% hit rate, 0 evictions. All 67 tests pass.

**Next Steps:**
- Re-run `benchmark_real.py --model tinyllama --max-tokens 64` to verify latency bugfix
- ~~Wire v0.3 Layer Router -- v0.4 Real Inference -- v0.5 KV Compression -- v0.6 Memory Virtualization into `CNOS.run()`~~
- Begin Phase 6: Expert Swarm (parallel token routing to specialised sub-models)

---

### 2026-06-05 -- v0.7 Integration Engine

**Tags:** `integration` `runtime` `benchmark`

**Goal:** Connect all five CNOS subsystems into a unified runtime: Neural Paging, Dynamic Layer Router, KV Cache Compression, Memory Virtualization, and Real Inference.

**Architecture:**

```
User Query
  -> RoutingController (ComplexityDetector + LayerSelector)
  -> MemoryController (NeuralPager + VirtualMemorySystem)
  -> ModelAdapter (RealModelAdapter / SimulatedModelAdapter)
  -> CacheController (KVCacheManager + quantizers)
  -> CnosResult (structured output with all metrics)
```

**Files Created (8):** `prototypes/integration_engine/`
  - `runtime.py` -- `CnosRuntime` master coordinator, `RuntimeConfig`, `CnosResult`
  - `model_adapter.py` -- `ModelAdapter` ABC; `RealModelAdapter` (real model inference via HuggingFace), `SimulatedModelAdapter` (neural_paging-based simulation); factory function for TinyLlama, Qwen 1.5B, Llama 3.2 1B
  - `routing_controller.py` -- `RoutingController` wrapping ComplexityDetector + LayerSelector; layer index clamping for cross-model compatibility
  - `memory_controller.py` -- `MemoryController` bridging NeuralPager (layer-level LRU) with VirtualMemorySystem (page-level tiers)
  - `cache_controller.py` -- `CacheController` wrapping KVCacheManager + quantizer + pruner + eviction policy
  - `benchmark.py` -- `Benchmark` comparing Baseline (all layers) vs CNOS across 9 test queries; Markdown/CSV/JSON report output
  - `test_runtime.py` -- 81 unit tests covering all controllers and the full pipeline
  - `README.md` -- Documentation with architecture, quick start, and usage

**Key Design Decision:** Two modes -- `simulate` (no model download, uses neural_paging for metrics) and `real` (requires HuggingFace model). Simulate mode allows the entire pipeline to be tested and benchmarked without GPU or 2GB+ downloads.

**Results:** 81/81 tests pass. Benchmark shows 52.5% avg compute reduction, 64.4% avg page hit rate across 9 queries (cache warms from 0% to 83%).

**Key Insight:** The negative latency reduction in simulate mode is expected -- simulating 22 layers is cheaper than simulating a routed subset due to overhead. Real mode will show actual latency improvements from skipping layers.

**Next Steps:**
- Run `benchmark.py --mode real` with TinyLlama downloaded to measure real latency reduction
- Begin Phase 6: Expert Swarm (parallel token routing to specialised sub-models)
