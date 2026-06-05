# System Architecture

## Overview

CNOS is not a model — it is an **operating system for neural execution**. It wraps any transformer-based language model and provides a resource-aware runtime that dynamically manages computation and memory.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     1. QUERY ANALYZER                                │
│   • Parse input                                                    │
│   • Extract intent, entities, constraints                           │
│   • Classify query type (factual, reasoning, creative, code)        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   2. COMPLEXITY DETECTOR                             │
│   • Estimate compute budget needed                                  │
│   • Output: complexity_score ∈ [0.0, 1.0]                           │
│   • Factors: length, ambiguity, domain novelty, reasoning depth     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     3. LAYER ROUTER                                  │
│   • For each token, decide which layers to execute                  │
│   • Routing policy: small NN (≈1M params)                          │
│   • Can skip layers (confidence ≥ threshold)                        │
│   • Can reorder layers for critical vs. non-critical tokens         │
│   • Output: per-token layer execution plan                          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  4. NEURAL PAGING ENGINE                             │
│   • Manages which model layers are in RAM vs. disk                  │
│   • Page fault handler: load layer on demand                        │
│   • Prefetcher: predict and load upcoming layers                    │
│   • Eviction policy: LRU + activation-frequency hybrid              │
│   • Output: layer availability map                                  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     5. MEMORY MANAGER                                │
│   • Unified virtual address space for all model weights             │
│   • Tier management: GPU VRAM → RAM → SSD mmap                     │
│   • Automatic promotion/demotion across tiers                       │
│   • KV cache allocation & compression (quantize, prune, evict)     │
│   • OOM prevention with graceful degradation                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     6. INFERENCE ENGINE                              │
│   • Execute the layer plan produced by Layer Router                 │
│   • Use layers paged in by Neural Paging Engine                     │
│   • Leverage KV cache from Memory Manager                           │
│   • Support for CPU (llama.cpp backend), CUDA, Metal, Vulkan        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     7. RESPONSE GENERATOR                            │
│   • Decode tokens with temperature, top-k, top-p sampling           │
│   • Streaming output via generator                                  │
│   • Post-processing (formatting, safety filtering)                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

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

## Key Design Principles

1. **Resource Awareness at Every Level** — Every component knows the current memory and compute budget and adapts accordingly.
2. **Graceful Degradation** — Under memory pressure, the system degrades quality smoothly rather than crashing.
3. **No Model Modification Required** — CNOS operates on pre-trained models without fine-tuning or architectural changes.
4. **Modular and Extensible** — Each subsystem has a clean interface, allowing independent research and replacement.
5. **Hardware Agnostic** — The same architecture works CPU-only, GPU-accelerated, or on hybrid setups.
