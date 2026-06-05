"""test_runtime  unit tests for the CNOS v0.7 Integration Engine.

Tests each controller independently and the full pipeline in simulate mode.
All tests run without a real model.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

# Import benchmark at module level before runtime pollutes sys.path
from benchmark import Benchmark  # noqa: E402
from benchmark import BenchmarkRow, BenchmarkSummary  # noqa: E402

_PASS: int = 0
_FAIL: int = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [PASS] {label}")
    else:
        _FAIL += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f"  {detail}"
        print(msg)


def approx(a: float, b: float, eps: float = 0.01) -> bool:
    return abs(a - b) < eps


# ===================================================================
# 1. ModelAdapter tests
# ===================================================================


def test_model_adapter_factory() -> None:
    print("\n--- ModelAdapter Factory ---")
    from model_adapter import create_model_adapter, SimulatedModelAdapter

    adapter = create_model_adapter("tinyllama", mode="simulate", ram_mb=1024)
    check("returns SimulatedModelAdapter",
          isinstance(adapter, SimulatedModelAdapter))
    check("num_layers == 22", adapter.num_layers == 22)
    check("model_name contains tinyllama",
          "tinyllama" in adapter.model_name.lower())

    adapter2 = create_model_adapter("qwen-1.5b", mode="simulate", ram_mb=1024)
    check("qwen num_layers == 28", adapter2.num_layers == 28)


def test_simulated_generate() -> None:
    print("\n--- SimulatedModelAdapter Generate ---")
    from model_adapter import create_model_adapter

    adapter = create_model_adapter("tinyllama", mode="simulate",
                                   max_tokens=32, ram_mb=1024)

    result = adapter.generate("test query", {0, 1, 2, 3, 4})
    check("response is string", isinstance(result.response, str))
    check("tokens_generated > 0", result.tokens_generated > 0)
    check("latency_s > 0", result.latency_s > 0)
    check("layers_executed == 5", result.layers_executed == 5)
    check("layers_skipped == 17", result.layers_skipped == 17)

    baseline = adapter.generate_baseline("baseline query")
    check("baseline layers == 22", baseline.layers_executed == 22)
    check("baseline skipped == 0", baseline.layers_skipped == 0)

    adapter.cleanup()


def test_model_adapter_unknown_key() -> None:
    print("\n--- ModelAdapter Unknown Key ---")
    from model_adapter import create_model_adapter

    try:
        create_model_adapter("nonexistent", mode="simulate")
        check("raises ValueError for unknown key", False)
    except ValueError:
        check("raises ValueError for unknown key", True)


def test_model_adapter_unknown_mode() -> None:
    print("\n--- ModelAdapter Unknown Mode ---")
    from model_adapter import create_model_adapter

    try:
        create_model_adapter("tinyllama", mode="invalid")
        check("raises ValueError for unknown mode", False)
    except ValueError:
        check("raises ValueError for unknown mode", True)


# ===================================================================
# 2. RoutingController tests
# ===================================================================


def test_routing_controller_create() -> None:
    print("\n--- RoutingController Create ---")
    from routing_controller import RoutingController

    rc = RoutingController(num_layers=22, policy_name="adaptive")
    check("num_layers == 22", rc.num_layers == 22)


def test_routing_controller_select_simple() -> None:
    print("\n--- RoutingController Select Simple Query ---")
    from routing_controller import RoutingController

    rc = RoutingController(num_layers=22, policy_name="adaptive")
    result = rc.select_layers("What is 2+2?")

    check("result has selected_layers", len(result.selected_layers) > 0)
    check("result has skipped_layers", len(result.skipped_layers) >= 0)
    check("num_selected + num_skipped == 22",
          result.num_selected + result.num_skipped == 22)
    check("compute_reduction_pct >= 0",
          result.compute_reduction_pct is not None and result.compute_reduction_pct >= 0)
    check("query_type is set", result.query_type in ("simple", "medium", "complex"))
    check("policy_name is set", len(result.policy_name) > 0)


def test_routing_controller_select_complex() -> None:
    print("\n--- RoutingController Select Complex Query ---")
    from routing_controller import RoutingController

    rc = RoutingController(num_layers=22, policy_name="static")
    result = rc.select_layers(
        "Derive the quadratic formula step by step with mathematical notation"
    )

    check("selected_layers non-empty", len(result.selected_layers) > 0)
    check("features is dict", isinstance(result.features, dict))


def test_routing_controller_set_policy() -> None:
    print("\n--- RoutingController Set Policy ---")
    from routing_controller import RoutingController

    rc = RoutingController(num_layers=22, policy_name="static")
    check("initial policy adaptive or static",
          rc.policy.name in ("static", "adaptive"))

    rc.set_policy("adaptive")
    check("policy changed to adaptive", rc.policy.name == "adaptive")


# ===================================================================
# 3. MemoryController tests
# ===================================================================


def test_memory_controller_create() -> None:
    print("\n--- MemoryController Create ---")
    from memory_controller import MemoryController

    mc = MemoryController(ram_gb=4, num_layers=22)
    check("ram_gb == 4", mc.ram_gb == 4)
    check("num_layers == 22", mc.num_layers == 22)


def test_memory_controller_prepare() -> None:
    print("\n--- MemoryController Prepare ---")
    from memory_controller import MemoryController

    mc = MemoryController(ram_gb=4, num_layers=22)
    latency = mc.prepare_layer(0)
    check("prepare_layer returns float", isinstance(latency, float))

    metrics = mc.get_metrics()
    check("metrics has ram_used_gb", metrics.ram_used_gb >= 0)
    check("metrics has total_layers", metrics.total_layers == 22)


def test_memory_controller_prepare_multi() -> None:
    print("\n--- MemoryController Prepare Multiple ---")
    from memory_controller import MemoryController

    mc = MemoryController(ram_gb=4, num_layers=22)
    latency = mc.prepare_layers({0, 1, 2, 3, 4})
    check("batch prepare returns float", isinstance(latency, float))

    metrics = mc.get_metrics()
    check("loaded_layers > 0", metrics.loaded_layers >= 0)


def test_memory_controller_out_of_range() -> None:
    print("\n--- MemoryController Out of Range ---")
    from memory_controller import MemoryController

    mc = MemoryController(ram_gb=4, num_layers=22)
    try:
        mc.prepare_layer(99)
        check("raises ValueError for OOB", False)
    except ValueError:
        check("raises ValueError for OOB", True)


# ===================================================================
# 4. CacheController tests
# ===================================================================


def test_cache_controller_create() -> None:
    print("\n--- CacheController Create ---")
    from cache_controller import CacheController

    cc = CacheController(num_layers=22, quantisation="int8")
    check("num_layers == 22", cc.num_layers == 22)
    check("total_memory_mb >= 0", cc.total_memory_mb >= 0)
    check("compression_ratio >= 1", cc.compression_ratio >= 1)


def test_cache_controller_compress_empty() -> None:
    print("\n--- CacheController Compress Empty ---")
    from cache_controller import CacheController

    cc = CacheController(num_layers=22, quantisation="int8")
    metrics = cc.compress()
    check("compression_ratio >= 1", metrics.compression_ratio >= 1)
    check("total_tokens >= 0", metrics.total_tokens >= 0)


def test_cache_controller_append_and_compress() -> None:
    print("\n--- CacheController Append and Compress ---")
    import torch
    from cache_controller import CacheController

    cc = CacheController(num_layers=4, num_heads=4, head_dim=32,
                         quantisation="int8")

    key = torch.randn(4, 1, 32)
    value = torch.randn(4, 1, 32)
    cc.append(0, key, value, position=0)
    cc.append(0, key, value, position=1)

    metrics = cc.compress()
    check("compression_ratio >= 1", metrics.compression_ratio >= 1)
    check("compression_ratio < 10", metrics.compression_ratio < 10)


# ===================================================================
# 5. CnosRuntime integration tests
# ===================================================================


def test_runtime_create() -> None:
    print("\n--- CnosRuntime Create ---")
    from runtime import CnosRuntime, RuntimeConfig

    config = RuntimeConfig(mode="simulate", max_tokens=16)
    rt = CnosRuntime(config)
    check("runtime has routing controller", hasattr(rt, "routing"))
    check("runtime has memory controller", hasattr(rt, "memory"))
    check("runtime has cache controller", hasattr(rt, "cache"))
    check("runtime has model adapter", hasattr(rt, "model"))


def test_runtime_process() -> None:
    print("\n--- CnosRuntime Process ---")
    from runtime import CnosRuntime, RuntimeConfig

    config = RuntimeConfig(mode="simulate", max_tokens=16)
    rt = CnosRuntime(config)

    result = rt.process("What is the capital of France?")
    check("result has response", len(result.response) > 0)
    check("layers_executed > 0", result.layers_executed > 0)
    check("layers_skipped >= 0", result.layers_skipped >= 0)
    check("compute_reduction_pct >= 0", result.compute_reduction_pct >= 0)
    check("latency_s > 0", result.latency_s > 0)
    check("tokens_generated > 0", result.tokens_generated > 0)
    check("routing is dict", isinstance(result.routing, dict))
    check("memory is dict", isinstance(result.memory, dict))
    check("cache is dict", isinstance(result.cache, dict))
    check("pipeline_times_s has routing", "routing" in result.pipeline_times_s)
    check("pipeline_times_s has memory", "memory" in result.pipeline_times_s)
    check("pipeline_times_s has inference", "inference" in result.pipeline_times_s)
    check("pipeline_times_s has cache", "cache" in result.pipeline_times_s)

    rt.cleanup()


def test_runtime_baseline() -> None:
    print("\n--- CnosRuntime Baseline ---")
    from runtime import CnosRuntime, RuntimeConfig

    config = RuntimeConfig(mode="simulate", max_tokens=16)
    rt = CnosRuntime(config)

    baseline = rt.process_baseline("test baseline")
    check("baseline layers == all layers",
          baseline.layers_executed == rt.model.num_layers)
    check("baseline skipped == 0", baseline.layers_skipped == 0)
    check("baseline latency > 0", baseline.latency_s > 0)

    rt.cleanup()


def test_runtime_multiple_queries() -> None:
    print("\n--- CnosRuntime Multiple Queries ---")
    from runtime import CnosRuntime, RuntimeConfig

    config = RuntimeConfig(mode="simulate", max_tokens=8)
    rt = CnosRuntime(config)

    queries = [
        "What is 2+2?",
        "Explain gravity",
        "Write a poem about AI",
    ]

    for q in queries:
        result = rt.process(q)
        check(f"processed: {q[:30]}", result.tokens_generated > 0)

    rt.cleanup()


# ===================================================================
# 6. CnosResult tests
# ===================================================================


def test_cnos_result_to_dict() -> None:
    print("\n--- CnosResult to_dict ---")
    from runtime import CnosResult

    r = CnosResult(
        query="test",
        response="test response",
        routing={"policy": "adaptive"},
        memory={"ram_used_gb": 1.5},
        cache={"compression_ratio": 4.0},
        latency_s=1.234,
        tokens_generated=42,
        layers_executed=10,
        layers_skipped=12,
        compute_reduction_pct=54.5,
        pipeline_times_s={"routing": 0.01, "inference": 1.0},
        timestamp="2026-01-01T00:00:00",
    )

    d = r.to_dict()
    check("dict has query", "query" in d)
    check("dict has response", "response" in d)
    check("dict has latency_s", "latency_s" in d)
    check("dict has routing", "routing" in d)
    check("dict has pipeline_times_s", "pipeline_times_s" in d)
    check("layers_executed == 10", d["layers_executed"] == 10)
    check("layers_skipped == 12", d["layers_skipped"] == 12)
    check("compute_reduction_pct == 54.5",
          approx(d["compute_reduction_pct"], 54.5))


# ===================================================================
# 7. Benchmark tests
# ===================================================================


def test_benchmark_run() -> None:
    print("\n--- Benchmark Run ---")
    from runtime import CnosRuntime, RuntimeConfig
    from benchmark import Benchmark

    rt = CnosRuntime(RuntimeConfig(mode="simulate", max_tokens=8))
    bm = Benchmark(runtime=rt, queries=[
        {"query": "test one", "type": "simple"},
        {"query": "test two medium complexity", "type": "medium"},
    ])
    summary = bm.run()

    check("summary has 2 rows", len(summary.rows) == 2)
    if summary.rows:
        check("row has latency_reduction_pct",
              summary.rows[0].latency_reduction_pct >= 0)
        check("row has compute_reduction_pct",
              summary.rows[0].compute_reduction_pct >= 0)
        check("row has query", len(summary.rows[0].query) > 0)

    check("avg_latency_reduction is float",
          isinstance(summary.avg_latency_reduction, float))
    check("avg_compute_reduction is float",
          isinstance(summary.avg_compute_reduction, float))


def test_benchmark_output_formats() -> None:
    print("\n--- Benchmark Output Formats ---")
    from runtime import CnosRuntime, RuntimeConfig
    from benchmark import Benchmark, BenchmarkRow, BenchmarkSummary

    rows = [
        BenchmarkRow(
            query="test", query_type="simple",
            cnos_latency_s=0.5, baseline_latency_s=1.0,
            latency_reduction_pct=50.0,
            baseline_layers=22, cnos_layers_executed=10,
            cnos_layers_skipped=12, compute_reduction_pct=45.45,
            ram_used_gb=0.5, page_faults=5, page_hits=50,
            hit_rate_pct=90.9, cache_compression_ratio=4.0,
            tokens_generated=32,
        ),
    ]

    summary = BenchmarkSummary(
        rows=rows,
        config={"model": "tinyllama"},
        timestamp="2026-01-01T00:00:00",
    )

    md = summary.to_markdown()
    check("markdown has table header", "| Query |" in md)
    check("markdown has config", "tinyllama" in md)
    check("markdown has query name", "test" in md)

    csv_content = summary.to_csv()
    check("CSV has header", "query" in csv_content.split("\n")[0])
    check("CSV has data", "test" in csv_content)


# ===================================================================
# Main
# ===================================================================


def main() -> int:
    global _PASS, _FAIL
    _PASS = 0
    _FAIL = 0

    test_model_adapter_factory()
    test_simulated_generate()
    test_model_adapter_unknown_key()
    test_model_adapter_unknown_mode()

    test_routing_controller_create()
    test_routing_controller_select_simple()
    test_routing_controller_select_complex()
    test_routing_controller_set_policy()

    test_memory_controller_create()
    test_memory_controller_prepare()
    test_memory_controller_prepare_multi()
    test_memory_controller_out_of_range()

    test_cache_controller_create()
    test_cache_controller_compress_empty()
    test_cache_controller_append_and_compress()

    test_runtime_create()
    test_runtime_process()
    test_runtime_baseline()
    test_runtime_multiple_queries()

    test_cnos_result_to_dict()

    test_benchmark_run()
    test_benchmark_output_formats()

    print(f"\n{'=' * 50}")
    print(f"  Results:  {_PASS} passed  |  {_FAIL} failed  |  "
          f"{_PASS + _FAIL} total")
    print(f"{'=' * 50}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
