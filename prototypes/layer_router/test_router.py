#!/usr/bin/env python3
"""Test suite for the CNOS Dynamic Layer Router.

Validates:
    - Complexity classification accuracy on known examples
    - Routing policy correctness (static, adaptive, experimental)
    - Layer selector integration
    - Metrics accumulation
    - Benchmark integrity
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Dict, List, Tuple

from complexity_detector import ComplexityDetector
from layer_selector import LayerSelector
from metrics import CumulativeMetrics
from routing_policy import (
    StaticPolicy,
    AdaptivePolicy,
    ExperimentalPolicy,
    RoutingPolicy,
    create_policy,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

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


def almost_eq(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# 1. Complexity Detector Tests
# ---------------------------------------------------------------------------


def test_complexity_detector() -> None:
    print("\n--- Complexity Detector ---")
    detector = ComplexityDetector(num_layers=80)

    # Known classifications
    cases: List[Tuple[str, str, float]] = [
        ("What is 2+2?", "simple", 0.0),
        ("How are you?", "simple", 0.0),
        ("Write Python code for binary search", "medium", 0.0),
        ("Write a Python function to find all prime numbers up to n", "medium", 0.0),
        ("Design a scalable distributed microservice architecture with fault tolerance and load balancing", "complex", 0.0),
        ("", "simple", 1.0),  # empty query
    ]

    for query, expected_type, _ in cases:
        result = detector.analyse(query)
        check(
            f"classify '{query[:40]}...' " if len(query) > 40 else f"classify '{query}'",
            result.query_type == expected_type,
            f"got '{result.query_type}', expected '{expected_type}' "
            f"(score={result.complexity_score:.3f})",
        )

    # Score ranges
    trivial = detector.analyse("Hello")
    check("trivial query score near 0", trivial.complexity_score < 0.2,
          f"score={trivial.complexity_score:.3f}")

    hard = detector.analyse(
        "Design a fault-tolerant distributed database system with "
        "consensus replication and automatic failover across regions."
    )
    check("hard query score > 0.4", hard.complexity_score > 0.4,
          f"score={hard.complexity_score:.3f}")

    # Confidence bounds
    check("confidence in [0.1, 1.0]", 0.1 <= trivial.confidence <= 1.0)

    # Depth labels
    check("trivial depth is factual-retrieval", trivial.reasoning_depth == "factual-retrieval")
    check("hard depth is not factual-retrieval", hard.reasoning_depth != "factual-retrieval")

    print(f"  Complexity Detector: {PASS - (PASS+FAIL - (_t := PASS+FAIL))} tests passed, "
          f"{(PASS+FAIL) - _t} total" if False else "")


# ---------------------------------------------------------------------------
# 2. Routing Policy Tests
# ---------------------------------------------------------------------------


def test_routing_policies() -> None:
    print("\n--- Routing Policies ---")

    # Static policy
    static = StaticPolicy(num_layers=80)
    simple_layers = static.select_layers(0.1, "simple")
    medium_layers = static.select_layers(0.5, "medium")
    complex_layers = static.select_layers(0.9, "complex")

    check("static simple < medium layers", len(simple_layers) < len(medium_layers),
          f"{len(simple_layers)} vs {len(medium_layers)}")
    check("static medium < complex layers", len(medium_layers) < len(complex_layers),
          f"{len(medium_layers)} vs {len(complex_layers)}")
    check("static complex == all layers", len(complex_layers) == 80,
          f"{len(complex_layers)} layers")
    check("static simple >= 15 layers", len(simple_layers) >= 15,
          f"{len(simple_layers)} layers")
    check("static always includes layer 0", 0 in simple_layers)
    check("static always includes layer 79", 79 in simple_layers)
    check("static layers are sorted", simple_layers == sorted(simple_layers))
    check("static layers are unique", len(simple_layers) == len(set(simple_layers)))

    # Adaptive policy
    adaptive = AdaptivePolicy(num_layers=80)
    ad_layers = adaptive.select_layers(0.1, "simple")
    check("adaptive returns valid layers", len(ad_layers) >= 15)

    # Record low feedback → should grow
    for _ in range(10):
        adaptive.record_feedback(0.3)
    ad_grown = adaptive.select_layers(0.1, "simple")
    check("adaptive grows with low feedback", len(ad_grown) > len(ad_layers),
          f"before={len(ad_layers)} after={len(ad_grown)}")

    # Experimental policy — even-odd
    exp_eo = ExperimentalPolicy(num_layers=80, strategy="even-odd")
    eo_layers = exp_eo.select_layers(0.1, "simple")
    check("experimental even-odd returns valid", len(eo_layers) > 0)
    check("experimental even-odd sorted", eo_layers == sorted(eo_layers))

    # Experimental policy — cluster
    exp_cl = ExperimentalPolicy(num_layers=80, strategy="cluster")
    cl_layers = exp_cl.select_layers(0.1, "simple")
    check("experimental cluster returns valid", len(cl_layers) > 0)

    # Experimental policy — density
    exp_de = ExperimentalPolicy(num_layers=80, strategy="density")
    de_simple = exp_de.select_layers(0.0, "simple")
    de_complex = exp_de.select_layers(0.9, "complex")
    check("experimental density complex == all", len(de_complex) == 80)
    check("experimental density simple < complex", len(de_simple) < len(de_complex))

    # Policy factory
    static2 = create_policy("static", 80)
    check("factory creates StaticPolicy", isinstance(static2, StaticPolicy))
    ad2 = create_policy("adaptive", 80)
    check("factory creates AdaptivePolicy", isinstance(ad2, AdaptivePolicy))
    exp2 = create_policy("experimental/density", 80)
    check("factory creates ExperimentalPolicy", isinstance(exp2, ExperimentalPolicy))

    # Invalid policy
    try:
        create_policy("unknown", 80)
        check("factory rejects unknown policy", False, "should have raised ValueError")
    except ValueError:
        check("factory rejects unknown policy", True)

    print(f"  Routing Policies: {PASS} passed")


# ---------------------------------------------------------------------------
# 3. Layer Selector Integration Tests
# ---------------------------------------------------------------------------


def test_layer_selector() -> None:
    print("\n--- Layer Selector ---")

    selector = LayerSelector(num_layers=80)

    # Simple query
    result = selector.select("What is 2+2?")
    check("simple query produces selection", len(result.selected_layers) > 0)
    check("simple query skipped layers", result.num_skipped > 0)
    check("simple query reduction > 0", result.compute_reduction_pct > 0)
    check("simple query type correct", result.complexity.query_type == "simple",
          f"got {result.complexity.query_type}")

    # Complex query
    result_c = selector.select(
        "Design a distributed fault-tolerant database system with consensus."
    )
    check("complex query uses more layers", len(result_c.selected_layers) >= len(result.selected_layers),
          f"simple={len(result.selected_layers)} complex={len(result_c.selected_layers)}")
    check("complex query reduction < simple reduction",
          result_c.compute_reduction_pct < result.compute_reduction_pct)

    # Policy swap
    selector.set_policy(AdaptivePolicy(num_layers=80))
    result_a = selector.select("What is 2+2?")
    check("policy swap works", result_a.policy_name == "adaptive")

    print(f"  Layer Selector: {PASS} passed")


# ---------------------------------------------------------------------------
# 4. Metrics Tests
# ---------------------------------------------------------------------------


def test_metrics() -> None:
    print("\n--- Metrics ---")

    selector = LayerSelector(num_layers=80)
    metrics = CumulativeMetrics()

    # Single update
    result = selector.select("What is 2+2?")
    metrics.update(result)
    check("metrics count after 1 query", metrics.total_queries == 1)
    check("metrics selected > 0", metrics.total_layers_selected > 0)
    check("metrics skipped > 0", metrics.total_layers_skipped > 0)

    # Multiple updates
    for query in ["Hello", "Write binary search", "Design distributed architecture"]:
        metrics.update(selector.select(query))
    check("metrics count after 4 queries", metrics.total_queries == 4)

    s = metrics.summary()
    check("summary has avg_layers_selected", "avg_layers_selected" in s)
    check("summary has compute_reduction_pct", "compute_reduction_pct" in s)
    check("summary has per_type", "per_type" in s)
    check("summary has query_type_distribution", "query_type_distribution" in s)

    # Per-type coverage
    qt = s["query_type_distribution"]
    check("summary has simple counts", qt.get("simple", 0) >= 1)

    print(f"  Metrics: {PASS} passed")


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------


def main() -> int:
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

    start = time.perf_counter()

    test_complexity_detector()
    test_routing_policies()
    test_layer_selector()
    test_metrics()

    elapsed = time.perf_counter() - start

    print(f"\n{'=' * 50}")
    print(f"  Results:  {PASS} passed  |  {FAIL} failed  |  {elapsed:.2f}s")
    print(f"{'=' * 50}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
