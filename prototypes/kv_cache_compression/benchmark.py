#!/usr/bin/env python3
"""benchmark — simulates KV cache compression across token-length scales.

Runs synthetic token-by-token generation through the KV cache manager
with various quantisation, pruning, and eviction configurations, then
prints a memory / latency comparison table.

Two modes:
    - **Full** (default): Runs actual tensor operations for accurate timing.
    - **Analytical** (``--analytical``): Computes compression ratios instantly
      from formulas (no tensor simulation).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Dict, Generator, List, Optional, Tuple

import torch

from kv_cache import KVCacheManager
from metrics import CompressionMetrics, CompressionRecord
from quantizer import get_quantizer
from pruner import get_pruner
from eviction_policy import get_eviction_policy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy token generator
# ---------------------------------------------------------------------------


def token_generator(
    num_tokens: int,
    num_layers: int = 22,
    num_heads: int = 32,
    head_dim: int = 64,
    device: Optional[torch.device] = None,
) -> Generator[List[Tuple[torch.Tensor, torch.Tensor]], None, None]:
    """Yield one token step at a time: list of ``(key, value)`` per layer."""
    dev = device or torch.device("cpu")
    scale = 0.02
    for _ in range(num_tokens):
        step: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num_layers):
            k = torch.randn(num_heads, 1, head_dim, device=dev) * scale
            v = torch.randn(num_heads, 1, head_dim, device=dev) * scale
            step.append((k, v))
        yield step


# ---------------------------------------------------------------------------
# Analytical computation (no tensor ops)
# ---------------------------------------------------------------------------


def _elems_per_token(num_layers: int, num_heads: int, head_dim: int) -> int:
    return num_layers * num_heads * head_dim * 2  # K + V


def run_analytical(
    num_tokens: int,
    num_layers: int = 22,
    num_heads: int = 32,
    head_dim: int = 64,
    quantisation: str = "fp16",
    max_cache_len: int = 4096,
) -> CompressionRecord:
    """Compute compression metrics analytically (instant, no tensor ops)."""
    baseline = min(num_tokens, max_cache_len)
    ept = _elems_per_token(num_layers, num_heads, head_dim)
    original_mb = (baseline * ept * 4) / (1024 ** 2)

    quantizer = get_quantizer(quantisation)
    bits = quantizer.bits_per_element
    compressed_mb = (baseline * ept * bits) / (8 * 1024 ** 2)
    saved = original_mb - compressed_mb
    ratio = original_mb / max(compressed_mb, 0.001)

    return CompressionRecord(
        num_tokens=num_tokens,
        quantisation=quantisation,
        pruner="none",
        eviction_policy="none",
        original_memory_mb=round(original_mb, 2),
        compressed_memory_mb=round(compressed_mb, 2),
        compression_ratio=round(ratio, 2),
        memory_saved_mb=round(saved, 2),
        quantize_time_ms=0.0,
        dequantize_time_ms=0.0,
        quality_score=1.0,
    )


# ---------------------------------------------------------------------------
# Full simulation benchmark (single config)
# ---------------------------------------------------------------------------


def run_config(
    num_tokens: int,
    num_layers: int = 22,
    num_heads: int = 32,
    head_dim: int = 64,
    quantisation: str = "fp16",
    pruner_name: str = "oldest_first",
    eviction_name: str = "lru",
    max_cache_len: int = 4096,
    device: Optional[torch.device] = None,
) -> CompressionRecord:
    """Run a single simulation (generates tensors, times quantisation)."""

    device = device or torch.device("cpu")
    baseline_tokens = min(num_tokens, max_cache_len)
    ept = _elems_per_token(num_layers, num_heads, head_dim)

    # --- Fill cache ---
    manager = KVCacheManager(
        num_layers=num_layers, num_heads=num_heads, head_dim=head_dim,
        max_cache_len=max_cache_len, device=device, quantisation=quantisation,
    )
    for step, step_data in enumerate(token_generator(
        baseline_tokens, num_layers, num_heads, head_dim, device,
    )):
        for layer_idx, (k, v) in enumerate(step_data):
            manager.append(layer_idx, k, v, position=step)

    # --- Original memory (FP32 baseline) ---
    original_memory_mb = (baseline_tokens * ept * 4) / (1024 ** 2)

    # --- Quantisation timing ---
    quantizer = get_quantizer(quantisation)
    total_quantize_s = 0.0
    for entry in manager.entries:
        if entry.seq_len > 0:
            t0 = time.perf_counter()
            quantizer.quantize(entry.keys)
            quantizer.quantize(entry.values)
            total_quantize_s += time.perf_counter() - t0

    # --- Prune + evict if over limit ---
    total_pruned = 0
    total_policy_s = 0.0
    pruner = get_pruner(pruner_name) if pruner_name != "none" else None
    policy = get_eviction_policy(eviction_name, pruner=pruner) if eviction_name != "none" else None

    if num_tokens > max_cache_len and policy is not None:
        tokens_to_free = baseline_tokens // 4
        t0 = time.perf_counter()
        candidates = policy.select_eviction_candidates(manager, tokens_to_free)
        total_policy_s += time.perf_counter() - t0
        for drop_indices in candidates.values():
            total_pruned += len(drop_indices)

    # --- Compressed memory (analytical) ---
    bits = quantizer.bits_per_element
    remaining = baseline_tokens - total_pruned
    compressed_mb = (remaining * ept * bits) / (8 * 1024 ** 2)
    ratio = original_memory_mb / max(compressed_mb, 0.001)

    return CompressionRecord(
        num_tokens=num_tokens,
        quantisation=quantisation,
        pruner=pruner_name,
        eviction_policy=eviction_name,
        original_memory_mb=round(original_memory_mb, 2),
        compressed_memory_mb=round(compressed_mb, 2),
        compression_ratio=round(ratio, 2),
        memory_saved_mb=round(original_memory_mb - compressed_mb, 2),
        quantize_time_ms=round(total_quantize_s * 1000, 3),
        dequantize_time_ms=round(total_policy_s * 1000, 3),
        total_cache_time_ms=round((total_quantize_s + total_policy_s) * 1000, 3),
        quality_score=1.0,
        num_tokens_pruned=total_pruned,
    )


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

TOKEN_COUNTS = [1000, 5000, 10000, 20000]
QUANTISATIONS = ["fp16", "int8", "int4"]
PRUNERS = ["none", "oldest_first"]
EVICTIONS = ["none", "lru"]


def run_benchmark(
    token_counts: Optional[List[int]] = None,
    quantisations: Optional[List[str]] = None,
    pruners: Optional[List[str]] = None,
    evictions: Optional[List[str]] = None,
    num_layers: int = 22,
    num_heads: int = 32,
    head_dim: int = 64,
    max_cache_len: int = 4096,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    analytical: bool = False,
) -> CompressionMetrics:
    """Run the full benchmark.  Use ``analytical=True`` for instant results."""
    token_counts = token_counts or TOKEN_COUNTS
    quantisations = quantisations or QUANTISATIONS
    pruners = pruners or PRUNERS
    evictions = evictions or EVICTIONS

    metrics = CompressionMetrics()
    total = len(token_counts) * len(quantisations) * len(pruners) * len(evictions)
    done = 0

    for nt in token_counts:
        for q in quantisations:
            for p in pruners:
                for e in evictions:
                    done += 1
                    if verbose:
                        print(f"\r  [{done}/{total}]  {nt:>6} tok  "
                              f"q={q}  p={p}  e={e}  ", end="", flush=True)

                    t0 = time.perf_counter()
                    if analytical:
                        record = run_analytical(nt, num_layers, num_heads,
                                                head_dim, q, max_cache_len)
                    else:
                        record = run_config(nt, num_layers, num_heads, head_dim,
                                            q, p, e, max_cache_len, device)
                    elapsed = time.perf_counter() - t0
                    metrics.add(record)

                    if verbose:
                        s = record.summary()
                        print(f"ratio={s['compression_ratio']:.1f}x  "
                              f"mem={s['original_mb']:.0f}->{s['compressed_mb']:.0f}MB"
                              f"  ({elapsed:.1f}s)", flush=True)

    if verbose:
        print()

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNOS KV Cache Compression Benchmark",
    )
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[1000, 5000, 10000, 20000],
                        help="Token counts to simulate")
    parser.add_argument("--quantisations", type=str, nargs="+",
                        default=["fp16", "int8", "int4"],
                        choices=["fp16", "int8", "int4"],
                        help="Quantisation schemes")
    parser.add_argument("--analytical", action="store_true",
                        help="Compute ratios from formulas (no tensor ops)")
    parser.add_argument("--fast", action="store_true",
                        help="Use small model config for quick test")
    parser.add_argument("--max-cache-len", type=int, default=4096,
                        help="Max cache length per layer")
    parser.add_argument("--verbose", action="store_true",
                        help="Per-config details")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

    if args.fast:
        num_layers, num_heads, head_dim = 4, 4, 16
    else:
        num_layers, num_heads, head_dim = 22, 32, 64

    mode = "analytical" if args.analytical else "simulation"
    print(f"\n  KV Cache Compression Benchmark [{mode}]")
    print(f"  Tokens:     {args.tokens}")
    print(f"  Quant:      {args.quantisations}")
    print(f"  Model:      {num_layers}L {num_heads}H {head_dim}D")
    print(f"  Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    metrics = run_benchmark(
        token_counts=args.tokens,
        quantisations=args.quantisations,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_cache_len=args.max_cache_len,
        verbose=args.verbose,
        analytical=args.analytical,
    )

    metrics.print_table("Memory Compression Comparison")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
