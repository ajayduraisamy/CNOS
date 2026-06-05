#!/usr/bin/env python3
"""Tests for CNOS Memory Virtualization Engine (v0.6).

Covers:
    - Memory tier configuration and accounting
    - Page table insertion, lookup, update, removal
    - Allocator page allocation across tiers
    - Eviction policies (LRU, LFU, Adaptive)
    - Prefetch engine predictors (sequential, stride, frequency)
    - Full VirtualMemorySystem access/fault/hit flow
    - Metrics aggregation and report generation
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict, List, Optional

from memory_tiers import TierConfig, TierManager, MemoryTier, DEFAULT_TIERS
from page_table import PageTable, PageTableEntry
from allocator import Allocator
from eviction_manager import (
    EvictionManager,
    LRUPolicy,
    LFUPolicy,
    AdaptivePolicy,
)
from prefetch_engine import (
    SequentialPredictor,
    StridePredictor,
    FrequencyPredictor,
    PrefetchEngine,
)
from metrics import VirtualMemoryMetrics, VirtualMemoryReport
from virtual_memory import VirtualMemorySystem, VirtualComponent

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
            msg += f"    {detail}"
        print(msg)


def approx(a: float, b: float, eps: float = 1e-5) -> bool:
    return abs(a - b) < eps


# ===================================================================
# 1. Memory Tier tests
# ===================================================================


def test_tier_config() -> None:
    print("\n--- TierConfig ---")
    cfg = TierConfig(
        name="TestRAM", capacity=1024**3 * 8, latency_ns=80,
        bandwidth_gbps=50.0, page_latency_ns=200, max_transfer_size=64*1024**2,
        is_volatile=True, tier_id=1,
    )
    check("capacity_gb correct", approx(cfg.capacity_gb, 8.0))
    check("capacity_mb correct", approx(cfg.capacity_mb, 8192.0))


def test_memory_tier() -> None:
    print("\n--- MemoryTier ---")
    cfg = TierConfig(name="RAM", capacity=1024**3, latency_ns=80,
                     bandwidth_gbps=50.0, page_latency_ns=200,
                     max_transfer_size=64*1024**2, tier_id=1)
    tier = MemoryTier(cfg)
    check("starts empty", tier.used == 0)
    check("free equals capacity", tier.free == cfg.capacity)

    tier.allocate(1024 * 1024)
    check("used after allocate", tier.used == 1024 * 1024)
    check("free after allocate", tier.free == cfg.capacity - 1024 * 1024)
    check("utilisation > 0", tier.utilisation_pct > 0)

    tier.free_space(512 * 1024)
    check("free after partial free", tier.used == 512 * 1024)

    tier.reset()
    check("used=0 after reset", tier.used == 0)

    try:
        tier.allocate(10 * 1024 ** 3)
        check("oversized allocate raises", False)
    except MemoryError:
        check("oversized allocate raises MemoryError", True)


def test_tier_manager() -> None:
    print("\n--- TierManager ---")
    configs = {
        1: TierConfig(name="RAM", capacity=1024**3, latency_ns=80,
                      bandwidth_gbps=50.0, page_latency_ns=200,
                      max_transfer_size=64*1024**2, tier_id=1),
        3: TierConfig(name="SSD", capacity=1024**3 * 100, latency_ns=100000,
                      bandwidth_gbps=3.5, page_latency_ns=50000,
                      max_transfer_size=1024**3, tier_id=3),
    }
    tm = TierManager(configs)
    check("two tiers", len(tm.tiers) == 2)
    check("RAM key exists", 1 in tm.tiers)
    check("SSD key exists", 3 in tm.tiers)
    check("total capacity",
          approx(tm.total_capacity_bytes, 1024**3 + 100 * 1024**3))
    s = tm.summary()
    check("summary has tier keys", "tier_1_name" in s)


# ===================================================================
# 2. Page Table tests
# ===================================================================


def test_page_table() -> None:
    print("\n--- PageTable ---")
    pt = PageTable()

    e1 = PageTableEntry(virtual_id=0, page_index=0, tier=1, offset=0, size=1024)
    pt.add(e1)
    check("1 entry after add", pt.total_pages == 1)

    e2 = PageTableEntry(virtual_id=0, page_index=1, tier=1, offset=1024, size=1024)
    pt.add(e2)
    check("2 entries after add", pt.total_pages == 2)

    got = pt.get(0, 0)
    check("get returns entry", got is not None and got.size == 1024)

    check("contains works", (0, 0) in pt)
    check("not contains", (9, 9) not in pt)

    pt.update_tier(0, 0, new_tier=3, new_offset=0)
    e1_updated = pt.get(0, 0)
    check("update tier", e1_updated is not None and e1_updated.tier == 3)

    pt.remove(0, 1)
    check("1 entry after remove", pt.total_pages == 1)

    pt.clear()
    check("0 after clear", pt.total_pages == 0)

    pt2 = PageTable()
    pt2.add(PageTableEntry(virtual_id=0, page_index=0, tier=1, offset=0, size=1024))
    pt2.mark_accessed(0, 0)
    pt2.mark_accessed(0, 0)
    e = pt2.get(0, 0)
    check("mark accessed on existing entry", e is not None)
    if e:
        check("access_count incremented", e.access_count >= 2)


# ===================================================================
# 3. Allocator tests
# ===================================================================


def _make_small_allocator(policy: str = "lru") -> Allocator:
    configs = {
        1: TierConfig(name="RAM", capacity=1024**3, latency_ns=80,
                      bandwidth_gbps=50.0, page_latency_ns=200,
                      max_transfer_size=64*1024**2, tier_id=1),
        2: TierConfig(name="CompKV", capacity=1024**3, latency_ns=500,
                      bandwidth_gbps=25.0, page_latency_ns=500,
                      max_transfer_size=16*1024**2, tier_id=2),
        3: TierConfig(name="SSD", capacity=1024**3 * 10, latency_ns=100000,
                      bandwidth_gbps=3.5, page_latency_ns=50000,
                      max_transfer_size=1024**3, tier_id=3),
    }
    tm = TierManager(configs)
    pt = PageTable()
    ev = EvictionManager(pt, tm, policy)
    al = Allocator(tm, pt, ev, page_size=64 * 1024)
    return al


def test_allocator() -> None:
    print("\n--- Allocator ---")
    al = _make_small_allocator()

    vid = al.allocate_virtual_id()
    check("first vid == 0", vid == 0)

    entries = al.allocate_pages(vid, num_pages=4, preferred_tier=1)
    check("4 pages allocated", len(entries) == 4)
    check("all in RAM", all(e.tier == 1 for e in entries))

    ram = al.tiers[1]
    check("RAM used increased", ram.used > 0)

    moved = al.move_page(0, 0, target_tier=3)
    check("move to SSD succeeds", moved)
    e = al.page_table.get(0, 0)
    if e:
        check("page now in SSD", e.tier == 3)

    promoted = al.promote(0, 0)
    check("promote works", promoted)
    e = al.page_table.get(0, 0)
    if e:
        check("promoted from SSD to KV tier", e.tier == 2)

    freed = al.free_pages(0)
    check("freed > 0", freed > 0)
    check("page table cleaned", al.page_table.get(0, 0) is None)


# ===================================================================
# 4. Eviction policy tests
# ===================================================================


def _make_evict_scenario(
    policy_name: str = "lru",
) -> Tuple[EvictionManager, PageTable, TierManager]:
    configs = {
        1: TierConfig(name="RAM", capacity=1024**3, latency_ns=80,
                      bandwidth_gbps=50.0, page_latency_ns=200,
                      max_transfer_size=64*1024**2, tier_id=1),
        2: TierConfig(name="CompKV", capacity=1024**3, latency_ns=500,
                      bandwidth_gbps=25.0, page_latency_ns=500,
                      max_transfer_size=16*1024**2, tier_id=2),
        3: TierConfig(name="SSD", capacity=1024**3 * 10, latency_ns=100000,
                      bandwidth_gbps=3.5, page_latency_ns=50000,
                      max_transfer_size=1024**3, tier_id=3),
    }
    tm = TierManager(configs)
    pt = PageTable()
    ev = EvictionManager(pt, tm, policy_name)

    for i in range(10):
        pt.add(PageTableEntry(
            virtual_id=0, page_index=i, tier=1,
            offset=i * 4096, size=4096,
            last_access_time=i,
            access_count=10 - i,
        ))
    tm[1].allocate(10 * 4096)
    return ev, pt, tm


def test_lru_eviction() -> None:
    print("\n--- LRU Eviction ---")
    ev, pt, tm = _make_evict_scenario("lru")
    victims = ev.policy.select_victims(1, bytes_needed=8192)  # type: ignore
    check("LRU returns victims", len(victims) > 0)
    # LRU: lowest last_access_time  page 0 and 1
    check("LRU victim 0 is oldest", 0 in [v[1] for v in victims])


def test_lfu_eviction() -> None:
    print("\n--- LFU Eviction ---")
    ev, pt, tm = _make_evict_scenario("lfu")
    victims = ev.policy.select_victims(1, bytes_needed=8192)  # type: ignore
    check("LFU returns victims", len(victims) > 0)
    # LFU: lowest access_count  page 9 (access_count = 10-9 = 1)
    check("LFU victim 9 is least used", 9 in [v[1] for v in victims])


def test_eviction_free_space() -> None:
    print("\n--- Eviction free_space ---")
    ev, pt, tm = _make_evict_scenario("lru")
    freed = ev.free_space(1, bytes_needed=8192)
    check("free_space frees > 0 bytes", freed >= 8192)
    check("RAM usage decreased", tm[1].used < 10 * 4096)


# ===================================================================
# 5. Prefetch predictor tests
# ===================================================================


def test_sequential_predictor() -> None:
    print("\n--- SequentialPredictor ---")
    pred = SequentialPredictor(window=3)
    for i in range(10):
        pred.record(0, i)
    next_p = pred.predict(0)
    check("sequential predicts next page", next_p == 10)

    # Non-sequential pattern
    pred2 = SequentialPredictor(window=3)
    for i in [0, 2, 5]:
        pred2.record(0, i)
    next_p2 = pred2.predict(0)
    check("non-sequential returns None", next_p2 is None)


def test_stride_predictor() -> None:
    print("\n--- StridePredictor ---")
    pred = StridePredictor(min_observations=3)
    for i in [0, 4, 8]:
        pred.record(0, i)
    next_p = pred.predict(0)
    check("stride predicts next", next_p == 12)

    # Non-uniform stride
    pred2 = StridePredictor(min_observations=3)
    for i in [0, 3, 7]:
        pred2.record(0, i)
    next_p2 = pred2.predict(0)
    check("non-uniform stride returns None", next_p2 is None)


def test_frequency_predictor() -> None:
    print("\n--- FrequencyPredictor ---")
    pred = FrequencyPredictor(top_k=3)
    for _ in range(5):
        pred.record(0, 1)
    for _ in range(3):
        pred.record(0, 2)
    for _ in range(1):
        pred.record(0, 3)

    top = pred.predict()
    check("frequency returns top pages", len(top) == 3)
    check("most frequent is first", top[0] == (0, 1))


# ===================================================================
# 6. VirtualMemorySystem tests
# ===================================================================


def test_vm_system_create() -> None:
    print("\n--- VMSystem Create ---")
    vm = VirtualMemorySystem(ram_gb=1, ssd_gb=10, page_size=64*1024)
    comp = vm.create_virtual_component(name="layer_0", num_pages=8)
    check("component created", comp is not None)
    check("virtual_id == 0", comp.virtual_id == 0)
    check("8 pages", comp.num_pages == 8)
    check("name set", comp.name == "layer_0")


def test_vm_system_access() -> None:
    print("\n--- VMSystem Access ---")
    vm = VirtualMemorySystem(ram_gb=1, ssd_gb=10, page_size=64*1024,
                             prefetch_enabled=False)
    comp = vm.create_virtual_component(name="test", num_pages=4)

    lat = vm.access(comp.virtual_id, 0)
    check("access returns latency", lat > 0)

    # Second access should be a hit
    prev_faults = vm.metrics.page_faults
    lat2 = vm.access(comp.virtual_id, 0)
    check("second access is hit", vm.metrics.page_hits > 0)


def test_vm_system_page_fault() -> None:
    print("\n--- VMSystem Page Fault ---")
    vm = VirtualMemorySystem(ram_gb=0.1, ssd_gb=1, page_size=64*1024,
                             prefetch_enabled=False)
    comp = vm.create_virtual_component(name="big", num_pages=200,
                                       preferred_tier=3)  # SSD only

    # First access should cause a page fault (page on SSD, not in RAM)
    vm.access(comp.virtual_id, 0)
    check("page fault recorded", vm.metrics.page_faults > 0)


def test_vm_summary() -> None:
    print("\n--- VMSystem Summary ---")
    vm = VirtualMemorySystem(ram_gb=1, ssd_gb=10)
    comp = vm.create_virtual_component(name="test", num_pages=2)
    vm.access(comp.virtual_id, 0)
    s = vm.summary()
    check("summary has total_entries", "total_entries" in s)
    check("summary has RAM info", "tier_1_name" in s)


# ===================================================================
# 7. Metrics tests
# ===================================================================


def test_vm_metrics() -> None:
    print("\n--- VirtualMemoryMetrics ---")
    m = VirtualMemoryMetrics()
    m.record_hit()
    m.record_hit()
    m.record_fault()
    check("hit rate 2/3", approx(m.hit_rate, 2.0 / 3.0))
    check("total accesses=3", m.total_accesses == 3)

    r = m.produce_report(
        model_name="7B", ram_gb=4.0, model_params_gb=14.0,
        kv_cache_gb=2.0, ram_used_gb=3.5, ssd_used_gb=10.0,
        total_prefetches=50, prefetch_accuracy=0.8,
    )
    check("report model name", r.model_name == "7B")
    check("report hit rate", approx(r.hit_rate_pct, 66.666, 0.1))
    check("report ram_used", approx(r.ram_used_gb, 3.5))
    check("report ram_util", approx(r.ram_util_pct, 87.5, 0.1))
    check("report prefetch acc", approx(r.prefetch_accuracy_pct, 80.0))
    check("report total accesses", r.total_accesses == 3)

    s = r.summary()
    check("summary dict has model", s["model"] == "7B")


# ===================================================================
# 8. VirtualComponent tests
# ===================================================================


def test_virtual_component() -> None:
    print("\n--- VirtualComponent ---")
    comp = VirtualComponent(
        virtual_id=5, name="layer_3", num_pages=16,
        page_size=1024 * 1024, entries=[],
    )
    check("size_bytes correct", comp.size_bytes == 16 * 1024 * 1024)
    check("size_gb correct", approx(comp.size_gb, 16.0 / 1024))


# ===================================================================
# Run all
# ===================================================================


def main() -> int:
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    start = time.perf_counter()

    test_tier_config()
    test_memory_tier()
    test_tier_manager()
    test_page_table()
    test_allocator()
    test_lru_eviction()
    test_lfu_eviction()
    test_eviction_free_space()
    test_sequential_predictor()
    test_stride_predictor()
    test_frequency_predictor()
    test_vm_system_create()
    test_vm_system_access()
    test_vm_system_page_fault()
    test_vm_summary()
    test_vm_metrics()
    test_virtual_component()

    elapsed = time.perf_counter() - start

    print(f"\n{'=' * 50}")
    print(f"  Results:  {PASS} passed  |  {FAIL} failed  |  {elapsed:.2f}s")
    print(f"{'=' * 50}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
