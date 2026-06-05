#!/usr/bin/env python3
"""benchmark — simulates KV cache compression across token-length scales.

Runs synthetic token-by-token generation through the KV cache manager
with various quantisation, pruning, and eviction configurations, then
prints a memory / latency comparison table.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

from kv_cache import KVCacheManager
from metrics import CompressionMetrics, CompressionRecord
from quantizer import QuantMetadata, get_quantizer
from pruner import get_pruner
from eviction_policy import get_eviction_policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulated token generation
# ---------------------------------------------------------------------------


def simulate_token_stream(
    num_tokens: int,
    num_layers: int = 22,
    num_heads: int = 32,
    head_dim: int = 64,
    device: Optional[torch.device] = None,
) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Generate synthetic key/value tensors for *num_tokens* tokens.

    Returns a list of length ``num_tokens``, each entry being a list of
    ``(key, value)`` tuples for every layer.
    """
    device = device or torch.device("cpu")
    stream: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    for _ in range(num_tokens):
        step_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num_layers):
            k = torch.randn(num_heads, 1, head_dim, device=device) * 0.02
            v = torch.randn(num_heads, 1, head_dim, device=device) * 0.02
            step_data.append((k, v))
        stream.append(step_data)
    return stream


# ---------------------------------------------------------------------------
# Single-config benchmark
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
    """Run a single benchmark configuration and return the result.

    Simulates token-by-token generation, periodically running quantisation,
    pruning, and eviction as the cache grows.
    """
    device = device or torch.device("cpu")
    manager = KVCacheManager(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_cache_len=max_cache_len,
        device=device,
        quantisation=quantisation,
    )
    quantizer = get_quantizer(quantisation)
    pruner = get_pruner(pruner_name) if pruner_name != "none" else None
    policy = get_eviction_policy(eviction_name, pruner=pruner) if eviction_name != "none" else None

    token_stream = simulate_token_stream(num_tokens, num_layers, num_heads, head_dim, device)

    total_quantize_s = 0.0
    total_dequantize_s = 0.0
    total_pruned = 0

    # Track original (FP32) memory for baseline
    fp32_bytes_per_elem = 4
    original_elems = num_tokens * num_layers * num_heads * head_dim * 2  # K + V
    original_memory_mb = (original_elems * fp32_bytes_per_elem) / (1024 ** 2)

    for step, step_data in enumerate(token_stream):
        # Append one token for each layer
        for layer_idx, (k, v) in enumerate(step_data):
            manager.append(layer_idx, k, v, position=step)

        # Periodically apply quantisation + pruning + eviction
        if step > 0 and step % 256 == 0:
            # Quantize
            for layer_idx in range(num_layers):
                entry = manager.entries[layer_idx]
                if entry.seq_len > 0:
                    t0 = time.perf_counter()
                    q_keys, meta = quantizer.quantize(entry.keys)
                    q_values, _ = quantizer.quantize(entry.values)
                    t1 = time.perf_counter()
                    total_quantize_s += (t1 - t0)

                    entry.keys = q_keys
                    entry.values = q_values

            # Prune + evict if over limit
            entry = manager.entries[0]
            if entry.seq_len > max_cache_len and policy is not None:
                tokens_to_free = entry.seq_len - max_cache_len
                t0 = time.perf_counter()
                candidates = policy.select_eviction_candidates(manager, tokens_to_free)
                t1 = time.perf_counter()
                total_dequantize_s += (t1 - t0)

                for layer_idx, drop_indices in candidates.items():
                    entry = manager.entries[layer_idx]
                    all_indices = set(range(entry.seq_len))
                    keep_indices = sorted(all_indices - set(drop_indices))
                    total_pruned += len(drop_indices)
                    entry.prune_to(keep_indices)

    # Calculate final compressed memory
    compressed_memory_mb = manager.total_memory_mb
    compression_ratio = original_memory_mb / max(compressed_memory_mb, 0.001)

    return CompressionRecord(
        num_tokens=num_tokens,
        quantisation=quantisation,
        pruner=pruner_name,
        eviction_policy=eviction_name,
        original_memory_mb=round(original_memory_mb, 2),
        compressed_memory_mb=round(compressed_memory_mb, 2),
        compression_ratio=round(compression_ratio, 2),
        memory_saved_mb=round(original_memory_mb - compressed_memory_mb, 2),
        quantize_time_ms=round(total_quantize_s * 1000, 3),
        dequantize_time_ms=round(total_dequantize_s * 1000, 3),
        total_cache_time_ms=round((total_quantize_s + total_dequantize_s) * 1000, 3),
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
) -> CompressionMetrics:
    """Run the full benchmark across all configuration combinations.

    Args:
        token_counts: List of sequence lengths to test.
        quantisations: List of quantisation schemes.
        pruners: List of pruning strategies.
        evictions: List of eviction policies.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Attention head dimension.
        max_cache_len: Maximum cache length before eviction.
        device: Torch device.
        verbose: Print per-config results.

    Returns:
        A :class:`CompressionMetrics` with all records.
    """
    token_counts = token_counts or TOKEN_COUNTS
    quantisations = quantisations or QUANTISATIONS
    pruners = pruners or PRUNERS
    evictions = evictions or EVICTIONS

    metrics = CompressionMetrics()
    total_configs = (
        len(token_counts) * len(quantisations) * len(pruners) * len(evictions)
    )
    config_num = 0

    for nt in token_counts:
        for q in quantisations:
            for p in pruners:
                for e in evictions:
                    config_num += 1
                    if verbose:
                        print(f"\n  Config [{config_num}/{total_configs}]: "
                              f"{nt} tokens  q={q}  pruner={p}  evict={e}")

                    t0 = time.perf_counter()
                    record = run_config(
                        num_tokens=nt,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        quantisation=q,
                        pruner_name=p,
                        eviction_name=e,
                        max_cache_len=max_cache_len,
                        device=device,
                    )
                    elapsed = time.perf_counter() - t0
                    metrics.add(record)

                    if verbose:
                        s = record.summary()
                        print(f"    Ratio: {s['compression_ratio']:.2f}x  "
                              f"Memory: {s['original_mb']} → {s['compressed_mb']} MB  "
                              f"({elapsed:.2f}s)")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNOS KV Cache Compression Benchmark",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1000, 5000, 10000, 20000],
        help="Token counts to simulate",
    )
    parser.add_argument(
        "--quantisations",
        type=str,
        nargs="+",
        default=["fp16", "int8", "int4"],
        choices=["fp16", "int8", "int4"],
        help="Quantisation schemes to test",
    )
    parser.add_argument(
        "--max-cache-len",
        type=int,
        default=4096,
        help="Maximum cache length per layer",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-config details",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

    print(f"\n  KV Cache Compression Benchmark")
    print(f"  ===============================")
    print(f"  Tokens:     {args.tokens}")
    print(f"  Quant:      {args.quantisations}")
    print(f"  Max cache:  {args.max_cache_len} tokens")
    print(f"  Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    metrics = run_benchmark(
        token_counts=args.tokens,
        quantisations=args.quantisations,
        max_cache_len=args.max_cache_len,
        verbose=args.verbose,
    )

    metrics.print_table("Memory Compression Comparison")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
