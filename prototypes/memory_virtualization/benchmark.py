#!/usr/bin/env python3
"""benchmark  simulates LLM memory access patterns under RAM constraints.

Models:
    * 7B    32 layers, ~14 GB params, ~13 GB KV cache
    * 30B   60 layers, ~60 GB params, ~60 GB KV cache
    * 70B   80 layers, ~140 GB params, ~80 GB KV cache

Configurations:
    * 4 GB RAM
    * 8 GB RAM

Each simulation runs a forward pass (sequential layer access) followed
by a generation phase (attention-based KV cache access), measuring page
faults, hit rates, evictions, and prefetch accuracy.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from typing import Dict, List, Optional, Tuple

from virtual_memory import VirtualMemorySystem
from memory_tiers import TierManager
from metrics import VirtualMemoryMetrics, VirtualMemoryReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------

MODEL_PROFILES: Dict[str, Dict] = {
    "7B": {
        "name": "7B",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "params_per_layer_mb": 400.0,    # MB per layer (FP16)
        "total_params_gb": 14.0,
    },
    "30B": {
        "name": "30B",
        "num_layers": 60,
        "hidden_size": 7168,
        "num_heads": 56,
        "head_dim": 128,
        "params_per_layer_mb": 1024.0,   # ~1 GB per layer
        "total_params_gb": 60.0,
    },
    "70B": {
        "name": "70B",
        "num_layers": 80,
        "hidden_size": 8192,
        "num_heads": 64,
        "head_dim": 128,
        "params_per_layer_mb": 1792.0,   # ~1.75 GB per layer
        "total_params_gb": 140.0,
    },
}

# KV cache: num_layers * num_heads * head_dim * 2 * seq_len * 2 bytes (FP16)
# At 4096 tokens: KV cache size  num_layers * num_heads * head_dim * 2 * 4096 * 2


def kv_cache_gb(model_key: str, seq_len: int = 4096) -> float:
    p = MODEL_PROFILES[model_key]
    bytes_per_token = p["num_layers"] * p["num_heads"] * p["head_dim"] * 2 * 2  # K+V, FP16
    total_bytes = bytes_per_token * seq_len
    return total_bytes / (1024 ** 3)


def params_gb(model_key: str) -> float:
    return MODEL_PROFILES[model_key]["total_params_gb"]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_model(
    model_key: str = "7B",
    ram_gb: float = 8.0,
    seq_len: int = 4096,
    num_generate_tokens: int = 256,
    eviction_policy: str = "lru",
    prefetch_enabled: bool = True,
    verbose: bool = False,
) -> VirtualMemoryReport:
    """Run a full simulation for one model/RAM configuration.

    Phases:
        1. **Allocation**  create virtual components for each layer's
           parameters and KV cache.
        2. **Forward pass**  access each layer sequentially (simulating
           a forward pass).
        3. **Generation**  access KV cache with attention-like patterns
           (recent tokens receive more accesses).
        4. **Report**  produce a :class:`VirtualMemoryReport`.

    Returns:
        A :class:`VirtualMemoryReport` with all performance metrics.
    """
    profile = MODEL_PROFILES[model_key]
    params_gb_total = params_gb(model_key)
    kv_gb_total = kv_cache_gb(model_key, seq_len)
    total_footprint = params_gb_total + kv_gb_total

    if verbose:
        print(f"\n  Simulating {model_key} @ {ram_gb}GB RAM "
              f"(footprint={total_footprint:.1f}GB, "
              f"policy={eviction_policy}, prefetch={prefetch_enabled})")

    # Build system
    vm = VirtualMemorySystem(
        ram_gb=ram_gb,
        eviction_policy=eviction_policy,
        prefetch_enabled=prefetch_enabled,
        page_size=1024 * 1024,  # 1 MB pages
    )

    # ------------------------------------------------------------------
    # Phase 1: Allocate model layers
    # ------------------------------------------------------------------
    n_layers = profile["num_layers"]
    mpml = profile["params_per_layer_mb"]
    pages_per_layer = max(1, int(mpml))  # ~400 pages for 7B

    layer_components: List[int] = []
    for layer_idx in range(n_layers):
        comp = vm.create_virtual_component(
            name=f"layer_{layer_idx}",
            num_pages=pages_per_layer,
            preferred_tier=3,  # Initially on SSD
        )
        layer_components.append(comp.virtual_id)

    # Allocate KV cache pages
    kv_pages_per_layer = max(1, int(kv_gb_total * 1024 / n_layers))
    kv_components: List[int] = []
    for layer_idx in range(n_layers):
        comp = vm.create_virtual_component(
            name=f"kv_layer_{layer_idx}",
            num_pages=kv_pages_per_layer // max(1, n_layers // 4),
            preferred_tier=2,  # Compressed KV tier
        )
        kv_components.append(comp.virtual_id)

    initial_ram = vm.tier_manager[1].used / (1024 ** 3)
    initial_ssd = vm.tier_manager[3].used / (1024 ** 3)
    if verbose:
        print(f"  Allocated: {n_layers} layers + KV, "
              f"RAM={initial_ram:.2f}GB  SSD={initial_ssd:.2f}GB")

    # ------------------------------------------------------------------
    # Phase 2: Forward pass  sequential layer access
    # ------------------------------------------------------------------
    n_forward_steps = n_layers * 3  # Multiple forward passes

    for step in range(n_forward_steps):
        layer_idx = step % n_layers
        for page in range(min(4, pages_per_layer)):
            latency = vm.access(layer_components[layer_idx], page)
            if verbose and step < 5:
                print(f"    Forward {step}: layer {layer_idx} page {page} "
                      f"lat={latency:.0f}ns "
                      f"faults={vm.metrics.page_faults}")

    # ------------------------------------------------------------------
    # Phase 3: Generation  KV cache with attention pattern
    # ------------------------------------------------------------------
    # Attention pattern: recent tokens get more accesses, simulating
    # causal self-attention where each new token attends to all previous.

    for gen_step in range(num_generate_tokens):
        for layer_idx in range(min(4, n_layers)):  # Sample layers
            # Access KV pages with recency bias
            kv_vid = kv_components[layer_idx]
            num_kv_pages = kv_pages_per_layer // max(1, n_layers // 4)
            for kv_page in range(min(2, num_kv_pages)):
                latency = vm.access(kv_vid, kv_page)
                if gen_step == 0 and verbose:
                    print(f"    Gen init: layer {layer_idx} KV page {kv_page} "
                          f"lat={latency:.0f}ns")

    # ------------------------------------------------------------------
    # Phase 4: Report
    # ------------------------------------------------------------------
    ram_used = vm.tier_manager[1].used / (1024 ** 3)
    ssd_used = vm.tier_manager[3].used / (1024 ** 3)
    total_page_table_entries = vm.page_table.total_pages

    report = vm.metrics.produce_report(
        model_name=model_key,
        ram_gb=ram_gb,
        model_params_gb=params_gb_total,
        kv_cache_gb=kv_gb_total,
        ram_used_gb=ram_used,
        ssd_used_gb=ssd_used,
        total_prefetches=vm.prefetcher.total_prefetches,
        prefetch_accuracy=vm.prefetcher.prefetch_accuracy,
    )

    if verbose:
        vm.metrics.print_report(report)
        print(f"  Page table entries: {total_page_table_entries}")
        print(f"  RAM/SSD: {ram_used:.2f}/{ssd_used:.2f} GB")

    return report


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    models: Optional[List[str]] = None,
    ram_configs: Optional[List[float]] = None,
    policies: Optional[List[str]] = None,
    prefetch_options: Optional[List[bool]] = None,
    verbose: bool = False,
) -> VirtualMemoryMetrics:
    """Run simulation across all model/RAM/policy combinations."""
    models = models or ["7B", "30B", "70B"]
    ram_configs = ram_configs or [4.0, 8.0]
    policies = policies or ["lru", "lfu", "adaptive"]
    prefetch_options = prefetch_options or [True, False]

    metrics = VirtualMemoryMetrics()
    total = (
        len(models) * len(ram_configs) * len(policies) * len(prefetch_options)
    )
    done = 0

    t_start = time.perf_counter()

    for model_key in models:
        for ram in ram_configs:
            for policy in policies:
                for prefetch in prefetch_options:
                    done += 1
                    label = f"[{done}/{total}]"
                    if verbose:
                        print(f"\n  {label} {model_key} @ {ram}GB  "
                              f"policy={policy}  prefetch={prefetch}")
                    else:
                        print(f"\r  {label}  {model_key}  {ram}GB  "
                              f"{policy}  prefetch={'on' if prefetch else 'off'}  ",
                              end="", flush=True)

                    t0 = time.perf_counter()
                    try:
                        report = simulate_model(
                            model_key=model_key,
                            ram_gb=ram,
                            eviction_policy=policy,
                            prefetch_enabled=prefetch,
                            verbose=verbose,
                        )
                        metrics.reports.append(report)
                    except Exception as exc:
                        logger.error("Failed %s @ %sGB: %s", model_key, ram, exc)
                        if verbose:
                            import traceback
                            traceback.print_exc()

                    elapsed = time.perf_counter() - t0
                    if verbose:
                        print(f"    ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t_start
    print(f"\n  Benchmark complete: {done} configs in {total_time:.1f}s")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNOS Virtual Memory Benchmark",
    )
    parser.add_argument("--models", type=str, nargs="+",
                        default=["7B", "30B", "70B"],
                        choices=["7B", "30B", "70B"],
                        help="Models to simulate")
    parser.add_argument("--ram", type=float, nargs="+",
                        default=[4.0, 8.0],
                        help="RAM sizes in GB")
    parser.add_argument("--policy", type=str, nargs="+",
                        default=["lru"],
                        choices=["lru", "lfu", "adaptive"],
                        help="Eviction policies")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Disable prefetching")
    parser.add_argument("--verbose", action="store_true",
                        help="Detailed per-step output")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

    print(f"\n  CNOS Virtual Memory Benchmark")
    print(f"  Models:   {args.models}")
    print(f"  RAM:      {args.ram} GB")
    print(f"  Policies: {args.policy}")

    prefetch_options = [not args.no_prefetch]

    metrics = run_benchmark(
        models=args.models,
        ram_configs=args.ram,
        policies=args.policy,
        prefetch_options=prefetch_options,
        verbose=args.verbose,
    )

    metrics.print_comparison()

    return 0


if __name__ == "__main__":
    sys.exit(main())
