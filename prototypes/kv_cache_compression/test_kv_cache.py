#!/usr/bin/env python3
"""Tests for CNOS KV Cache Compression Engine (v0.5).

Covers:
    - KVCacheEntry append / prune / clear
    - KVCacheManager memory tracking
    - FP16 / INT8 / INT4 quantisation round-trips
    - OldestFirst / LeastUsed / AttentionScore pruners
    - LRU / LFU / Adaptive eviction policies
    - Metrics aggregation
"""

from __future__ import annotations

import logging
import math
import sys
import time
from typing import Any, Dict, List, Optional, Set

import torch

from kv_cache import KVCacheEntry, KVCacheManager
from quantizer import (
    FP16Quantizer,
    INT8Quantizer,
    INT4Quantizer,
    get_quantizer,
)
from pruner import (
    OldestFirstPruner,
    LeastUsedPruner,
    AttentionScorePruner,
    get_pruner,
)
from eviction_policy import (
    LRUPolicy,
    LFUPolicy,
    AdaptivePolicy,
    get_eviction_policy,
)
from metrics import CompressionMetrics, CompressionRecord

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f"  —  {detail}"
        print(msg)


def approx(a: float, b: float, eps: float = 1e-5) -> bool:
    return abs(a - b) < eps


# ===================================================================
# 1. KVCacheEntry tests
# ===================================================================


def test_kv_cache_entry() -> None:
    print("\n--- KVCacheEntry ---")

    k = torch.randn(32, 0, 64)
    v = torch.randn(32, 0, 64)
    entry = KVCacheEntry(keys=k, values=v)
    check("entry starts empty", entry.seq_len == 0)
    check("memory is zero for empty", entry.memory_bytes == 0)

    k1 = torch.randn(32, 1, 64)
    v1 = torch.randn(32, 1, 64)
    entry.append(k1, v1, position=0)
    check("seq_len=1 after append", entry.seq_len == 1)

    k2 = torch.randn(32, 1, 64)
    v2 = torch.randn(32, 1, 64)
    entry.append(k2, v2, position=1)
    check("seq_len=2 after two appends", entry.seq_len == 2)
    check("token_positions tracked", entry.token_positions == [0, 1])

    entry.prune_to([1])
    check("seq_len=1 after prune", entry.seq_len == 1)
    check("pruned to last token", entry.token_positions == [1])

    entry.clear()
    check("seq_len=0 after clear", entry.seq_len == 0)

    # Memory accounting
    k3 = torch.randn(4, 10, 8)
    v3 = torch.randn(4, 10, 8)
    e2 = KVCacheEntry(keys=k3.half(), values=v3.half())
    expected_mem = (4 * 10 * 8 * 2) * 2  # fp16 (2 bytes) * K+V
    check("memory_bytes correct for fp16", e2.memory_bytes == expected_mem)
    check("num_heads and head_dim inferred", e2.num_heads == 4 and e2.head_dim == 8)

    print(f"  KVCacheEntry: {PASS - (FAIL > 0)} passed")


# ===================================================================
# 2. KVCacheManager tests
# ===================================================================


def test_kv_cache_manager() -> None:
    print("\n--- KVCacheManager ---")

    mgr = KVCacheManager(num_layers=4, num_heads=8, head_dim=16, quantisation="fp16")
    check("all layers initialised", len(mgr.entries) == 4)
    check("all entries empty initially",
          all(e.seq_len == 0 for e in mgr.entries))

    k = torch.randn(8, 1, 16)
    v = torch.randn(8, 1, 16)
    mgr.append(0, k, v)
    mgr.append(1, k, v)
    check("layer 0 has 1 token", mgr.entries[0].seq_len == 1)
    check("layer 1 has 1 token", mgr.entries[1].seq_len == 1)
    check("total_cached_tokens == 2", mgr.total_cached_tokens == 2)

    mgr.clear()
    check("all empty after clear", all(e.seq_len == 0 for e in mgr.entries))

    comp_fp16 = mgr.compression_ratio
    check("fp16 compression ratio = 2.0", approx(comp_fp16, 2.0))

    mgr_int4 = KVCacheManager(quantisation="int4")
    check("int4 compression ratio = 8.0", approx(mgr_int4.compression_ratio, 8.0))

    s = mgr.summary()
    check("summary contains expected keys",
          all(k in s for k in ("num_layers", "total_memory_mb", "compression_ratio")))

    print(f"  KVCacheManager: {PASS - (FAIL > 0)} passed")


# ===================================================================
# 3. Quantizer tests
# ===================================================================


def test_fp16_quantizer() -> None:
    print("\n--- FP16 Quantizer ---")

    q = FP16Quantizer()
    x = torch.randn(4, 8, 16)
    encoded, meta = q.quantize(x)
    check("fp16 dtype is float16", encoded.dtype == torch.float16)
    check("fp16 scheme correct", meta.scheme == "fp16")
    restored = q.dequantize(encoded, meta)
    check("fp16 round-trip preserves values",
          torch.allclose(x, restored.float(), atol=1e-3))


def test_int8_quantizer() -> None:
    print("\n--- INT8 Quantizer ---")

    q = INT8Quantizer()
    x = torch.randn(4, 8, 16) * 5.0
    encoded, meta = q.quantize(x)
    check("int8 dtype is int8", encoded.dtype == torch.int8)
    check("int8 scheme correct", meta.scheme == "int8")
    check("meta has scale", meta.scale is not None)
    restored = q.dequantize(encoded, meta)
    # INT8 has ~0.4% relative error per element
    mse = ((restored.float() - x) ** 2).mean().item()
    check(f"int8 round-trip (mse={mse:.6f})", mse < 0.1)

    # Constant tensor
    x_const = torch.ones(4, 8, 16) * 3.0
    enc_c, meta_c = q.quantize(x_const)
    dec_c = q.dequantize(enc_c, meta_c)
    mse_c = ((dec_c.float() - x_const) ** 2).mean().item()
    check(f"int8 constant round-trip (mse={mse_c:.6f})", mse_c < 5.0)


def test_int4_quantizer() -> None:
    print("\n--- INT4 Quantizer ---")

    q = INT4Quantizer()
    x = torch.randn(4, 8, 16) * 2.0
    encoded, meta = q.quantize(x)
    check("int4 dtype is uint8", encoded.dtype == torch.uint8)
    check("int4 scheme correct", meta.scheme == "int4")

    # INT4 packs 2 values per byte, so encoded size ~= half of original
    expected_encoded_elems = math.ceil(x.numel() / 2)
    check(f"int4 packing ({encoded.numel()} ~= {expected_encoded_elems})",
          encoded.numel() == expected_encoded_elems)

    restored = q.dequantize(encoded, meta)
    mse = ((restored.float() - x) ** 2).mean().item()
    check(f"int4 round-trip (mse={mse:.6f})", mse < 2.0)

    # Zero tensor
    x_zero = torch.zeros(4, 8, 16)
    enc_z, meta_z = q.quantize(x_zero)
    dec_z = q.dequantize(enc_z, meta_z)
    check("int4 zero round-trip",
          torch.allclose(dec_z, x_zero, atol=1e-3))


def test_quantizer_registry() -> None:
    print("\n--- Quantizer Registry ---")

    fp16 = get_quantizer("fp16")
    int8 = get_quantizer("int8")
    int4 = get_quantizer("int4")
    check("fp16 quantizer loaded", isinstance(fp16, FP16Quantizer))
    check("int8 quantizer loaded", isinstance(int8, INT8Quantizer))
    check("int4 quantizer loaded", isinstance(int4, INT4Quantizer))
    check("fp16 bits=16", fp16.bits_per_element == 16)
    check("int8 bits=8", int8.bits_per_element == 8)
    check("int4 bits=4", int4.bits_per_element == 4)

    try:
        get_quantizer("fp8")
        check("unknown quantizer raises", False)
    except KeyError:
        check("unknown quantizer raises KeyError", True)

    print(f"  Quantizer Registry: {PASS - (FAIL > 0)} passed")


# ===================================================================
# 4. Pruner tests
# ===================================================================


def test_oldest_first_pruner() -> None:
    print("\n--- OldestFirst Pruner ---")

    p = OldestFirstPruner()
    k = torch.randn(4, 10, 8)
    v = torch.randn(4, 10, 8)
    entry = KVCacheEntry(keys=k, values=v)

    keep = p.prune(entry, target_tokens=4)
    check("oldest_first keeps last 4", keep == [6, 7, 8, 9])

    keep_all = p.prune(entry, target_tokens=10)
    check("oldest_first keeps all when <= target", keep_all == list(range(10)))


def test_least_used_pruner() -> None:
    print("\n--- LeastUsed Pruner ---")

    p = LeastUsedPruner()
    k = torch.randn(4, 10, 8)
    v = torch.randn(4, 10, 8)
    entry = KVCacheEntry(keys=k, values=v)

    # Without attention scores, falls back to recency
    keep = p.prune(entry, target_tokens=5)
    check("least_used retains 5 tokens (no scores)", len(keep) == 5)

    # With scores: lower score tokens should be dropped
    scores = torch.tensor([[0.5, 0.1, 0.3, 0.05, 0.7, 0.01, 0.6, 0.2, 0.4, 0.8]])
    keep2 = p.prune(entry, target_tokens=5, attention_scores=scores)
    check("least_used with scores keeps top-5", len(keep2) == 5)
    # highest scores: idx 9 (0.8), 4 (0.7), 6 (0.6), 0 (0.5), 8 (0.4)
    expected_top5 = {0, 4, 6, 8, 9}
    check("least_used picks correct tokens", set(keep2) == expected_top5)


def test_attention_score_pruner() -> None:
    print("\n--- AttentionScore Pruner ---")

    p = AttentionScorePruner()
    k = torch.randn(4, 10, 8)
    v = torch.randn(4, 10, 8)
    entry = KVCacheEntry(keys=k, values=v)

    scores = torch.tensor([[0.01, 0.02, 0.5, 0.3, 0.4, 0.1, 0.6, 0.05, 0.8, 0.2]])
    keep = p.prune(entry, target_tokens=3, attention_scores=scores)
    check("attention_score pruner keeps top 3", len(keep) == 3)
    expected = {8, 6, 2}  # scores 0.8, 0.6, 0.5
    check("attention_score picks highest scores", set(keep) == expected)

    # Without scores, falls back to oldest-first
    keep_fallback = p.prune(entry, target_tokens=3)
    check("attention_score fallback oldest_first", keep_fallback == [7, 8, 9])


def test_pruner_registry() -> None:
    print("\n--- Pruner Registry ---")

    p1 = get_pruner("oldest_first")
    p2 = get_pruner("least_used")
    p3 = get_pruner("attention_score")
    check("oldest_first in registry", isinstance(p1, OldestFirstPruner))
    check("least_used in registry", isinstance(p2, LeastUsedPruner))
    check("attention_score in registry", isinstance(p3, AttentionScorePruner))

    try:
        get_pruner("random_strategy")
        check("unknown pruner raises", False)
    except KeyError:
        check("unknown pruner raises KeyError", True)

    print(f"  Pruner Registry: {PASS - (FAIL > 0)} passed")


# ===================================================================
# 5. Eviction policy tests
# ===================================================================


def _make_manager(
    num_layers: int = 4,
    seq_lens: Optional[List[int]] = None,
) -> KVCacheManager:
    mgr = KVCacheManager(
        num_layers=num_layers,
        num_heads=4,
        head_dim=8,
        max_cache_len=100,
        quantisation="fp16",
    )
    if seq_lens:
        for layer_idx, sl in enumerate(seq_lens):
            k = torch.randn(4, sl, 8)
            v = torch.randn(4, sl, 8)
            mgr.entries[layer_idx] = KVCacheEntry(keys=k, values=v)
    return mgr


def test_lru_policy() -> None:
    print("\n--- LRU Policy ---")

    mgr = _make_manager(seq_lens=[0, 5, 10, 3])
    # Set last_access_time: layer 1 oldest, layer 2 newest
    mgr.entries[1].last_access_time = 1
    mgr.entries[2].last_access_time = 100
    mgr.entries[3].last_access_time = 50

    policy = LRUPolicy()
    candidates = policy.select_eviction_candidates(mgr, tokens_to_free=4)
    check("LRU returns candidates", len(candidates) > 0)
    # Layer 1 has oldest access time, should be evicted first
    check("LRU evicts from oldest layer", 1 in candidates)
    total_dropped = sum(len(v) for v in candidates.values())
    check(f"LRU drops ~4 tokens (dropped={total_dropped})", 3 <= total_dropped <= 5)


def test_lfu_policy() -> None:
    print("\n--- LFU Policy ---")

    mgr = _make_manager(seq_lens=[0, 5, 10, 3])
    mgr.entries[1].access_count = 1
    mgr.entries[2].access_count = 100
    mgr.entries[3].access_count = 50

    policy = LFUPolicy()
    candidates = policy.select_eviction_candidates(mgr, tokens_to_free=4)
    check("LFU returns candidates", len(candidates) > 0)
    check("LFU evicts from least-used layer", 1 in candidates)


def test_adaptive_policy() -> None:
    print("\n--- Adaptive Policy ---")

    # Low pressure (< 50%): should use LFU
    mgr_low = _make_manager(seq_lens=[10, 10, 10, 10])  # 10/100 = 10% each
    for i, entry in enumerate(mgr_low.entries):
        entry.access_count = (i + 1) * 10  # layer 0=10, 1=20, 2=30, 3=40
    mgr_low.entries[0].access_count = 1  # layer 0: least used

    policy = AdaptivePolicy(pressure_threshold=0.5)
    cand_low = policy.select_eviction_candidates(mgr_low, tokens_to_free=2)
    check("adaptive low pressure: evicts from low-access layer",
          0 in cand_low)

    # High pressure (>= 50%): should use LRU
    mgr_high = _make_manager(seq_lens=[60, 20, 10, 10])
    mgr_high.entries[0].last_access_time = 1
    mgr_high.entries[1].last_access_time = 100
    cand_high = policy.select_eviction_candidates(mgr_high, tokens_to_free=2)
    check("adaptive high pressure: evicts from oldest layer",
          0 in cand_high)


def test_eviction_registry() -> None:
    print("\n--- Eviction Registry ---")

    lru = get_eviction_policy("lru")
    lfu = get_eviction_policy("lfu")
    ada = get_eviction_policy("adaptive")
    check("lru in registry", isinstance(lru, LRUPolicy))
    check("lfu in registry", isinstance(lfu, LFUPolicy))
    check("adaptive in registry", isinstance(ada, AdaptivePolicy))

    try:
        get_eviction_policy("random_policy")
        check("unknown policy raises", False)
    except KeyError:
        check("unknown policy raises KeyError", True)

    print(f"  Eviction Registry: {PASS - (FAIL > 0)} passed")


# ===================================================================
# 6. Metrics tests
# ===================================================================


def test_metrics() -> None:
    print("\n--- CompressionMetrics ---")

    r1 = CompressionRecord(
        num_tokens=1000,
        quantisation="int8",
        original_memory_mb=100.0,
        compressed_memory_mb=25.0,
        compression_ratio=4.0,
        memory_saved_mb=75.0,
    )
    r2 = CompressionRecord(
        num_tokens=1000,
        quantisation="int4",
        original_memory_mb=100.0,
        compressed_memory_mb=12.5,
        compression_ratio=8.0,
        memory_saved_mb=87.5,
    )
    r3 = CompressionRecord(
        num_tokens=5000,
        quantisation="int8",
        original_memory_mb=500.0,
        compressed_memory_mb=125.0,
        compression_ratio=4.0,
        memory_saved_mb=375.0,
    )

    metrics = CompressionMetrics([r1, r2, r3])
    check("has 3 records", len(metrics.records) == 3)

    q_comp = metrics.compare_quantisation(1000)
    check("2 records for 1000 tokens", len(q_comp) == 2)

    best = metrics.best_compression(1000)
    check("best for 1000 is int4", best is not None and best.quantisation == "int4")

    best_5k = metrics.best_compression(5000)
    check("best for 5000 is int8", best_5k is not None and best_5k.quantisation == "int8")

    rows = metrics.table_rows()
    check("table_rows returns 3 rows", len(rows) == 3)
    check("table_rows has all keys",
          all(k in rows[0] for k in (
              "num_tokens", "quantisation", "compression_ratio", "memory_saved_mb"
          )))

    print(f"  CompressionMetrics: {PASS - (FAIL > 0)} passed")


# ===================================================================
# Run all
# ===================================================================


def main() -> int:
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    start = time.perf_counter()

    test_kv_cache_entry()
    test_kv_cache_manager()
    test_fp16_quantizer()
    test_int8_quantizer()
    test_int4_quantizer()
    test_quantizer_registry()
    test_oldest_first_pruner()
    test_least_used_pruner()
    test_attention_score_pruner()
    test_pruner_registry()
    test_lru_policy()
    test_lfu_policy()
    test_adaptive_policy()
    test_eviction_registry()
    test_metrics()

    elapsed = time.perf_counter() - start

    print(f"\n{'=' * 50}")
    print(f"  Results:  {PASS} passed  |  {FAIL} failed  |  {elapsed:.2f}s")
    print(f"{'=' * 50}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


