# Memory Virtualization Engine (v0.6)

Unified virtual memory manager for large language model execution,
virtualising SSD, RAM, compressed KV cache, and model layers into one
logical memory space.

## Architecture

```
virtual_memory.py        Orchestrator — unified address space, access, faults
    │
    ├── memory_tiers.py   Tier definitions, capacity tracking, transfer costs
    ├── page_table.py     Virtual → physical mapping, access tracking
    ├── allocator.py      Page allocation, promotion, demotion across tiers
    ├── eviction_manager.py  LRU / LFU / Adaptive eviction policies
    ├── prefetch_engine.py   Sequential / stride / frequency prefetching
    └── metrics.py        Performance counters, hit rate, report generation
```

## Tiers

| Tier | Name | Typical Size | Latency | Bandwidth |
|------|------|-------------|---------|-----------|
| 0 | GPU VRAM | 24 GB | 200 ns | 900 GB/s |
| 1 | CPU RAM | Configurable (4/8/32 GB) | 80 ns | 50 GB/s |
| 2 | Compressed KV | 4 GB (INT4) | 500 ns | 25 GB/s |
| 3 | SSD | 500 GB | 100 µs | 3.5 GB/s |

## Quick Start

```python
from virtual_memory import VirtualMemorySystem

vm = VirtualMemorySystem(ram_gb=8)

# Create a model layer (virtual component)
layer = vm.create_virtual_component(name="layer_0", num_pages=64)

# Access pages as if they were in a flat address space
latency_ns = vm.access(layer.virtual_id, page_index=5)
print(f"Access latency: {latency_ns:.0f} ns")

# Generate report
report = vm.metrics.produce_report(model_name="7B", ram_gb=8.0, ...)
vm.metrics.print_report(report)
```

## Benchmark

Simulate 7B, 30B, and 70B models under 4GB and 8GB RAM constraints:

```bash
python prototypes/memory_virtualization/benchmark.py --models 7B 30B 70B --ram 4 8 --verbose
```

Run all eviction policies for comparison:

```bash
python prototypes/memory_virtualization/benchmark.py --policy lru lfu adaptive
```

## Tests

```bash
python prototypes/memory_virtualization/test_virtual_memory.py
```

## Key Design Decisions

- **Page-level virtualisation** — each model layer is divided into 1 MB
  pages; the page table maps `(virtual_id, page_index) → (tier, offset)`.
- **Cold start on SSD** — all layer parameters are initially allocated on
  SSD; the first access triggers a page fault that loads into RAM.
- **Three predictors** — sequential (next-page), stride (constant-offset),
  and frequency (hot KV cache pages) are combined for prefetching.
- **Adaptive eviction** — switches between LRU (safe, oldest-first under
  pressure) and LFU (targets genuinely unused pages at low pressure).
- **Analytical memory** — all sizes are computed from model config
  (layers × heads × dim × bytes) rather than measured at runtime.
