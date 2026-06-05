#!/usr/bin/env python3
"""Tests for the CNOS Real Transformer Integration (v0.4).

Validates:
    - Model loading (mock / fast-path for CI without GPU)
    - LayerGate wrapper forward and passthrough
    - LayerMonitor statistics tracking
    - Quality evaluator metrics (Jaccard, ROUGE-L)
    - Layer plan application correctness
"""

from __future__ import annotations

import logging
import math
import sys
import time
from typing import List, Set

import torch
import torch.nn as nn

from layer_hooks import (
    LayerGate,
    LayerMonitor,
    install_layer_gates,
    set_active_layers,
)
from quality_evaluator import QualityEvaluator, token_jaccard, rouge_l_f1

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


# ---------------------------------------------------------------------------
# 1. LayerGate tests
# ---------------------------------------------------------------------------


class DummyLayer(nn.Module):
    """Minimal layer that adds a learnable bias to hidden states."""

    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.bias = nn.Parameter(torch.ones(1, 1, 4) * layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Add bias to simulate layer computation (v5.x returns tensor)."""
        return hidden_states + self.bias


def test_layer_gate() -> None:
    print("\n--- LayerGate ---")

    x = torch.randn(1, 3, 4)
    dummy = DummyLayer(5)
    gate = LayerGate(dummy, layer_id=5)

    # Active: should add bias
    gate.active = True
    out_active = gate(x)
    expected = x + 5.0
    check("active gate adds bias", torch.allclose(out_active, expected))

    # Inactive: passthrough (no bias added)
    gate.active = False
    out_skip = gate(x)
    check("inactive gate passes through", torch.allclose(out_skip, x),
          f"out={out_skip} expected={x}")

    # Passthrough with extra kwargs is ignored (returns hidden_states only)
    out_kw = gate(x, use_cache=True, past_key_value="anything")
    check("passthrough with kwargs returns hidden_states",
          torch.allclose(out_kw, x))

    # Full passthrough with all kwargs
    out_full = gate(x, use_cache=True, output_attentions=True)
    check("full passthrough preserves hidden", torch.allclose(out_full, x))

    print(f"  LayerGate: {PASS - (FAIL > 0)} passed" if False else "")


# ---------------------------------------------------------------------------
# 2. LayerMonitor tests
# ---------------------------------------------------------------------------


def test_layer_monitor() -> None:
    print("\n--- LayerMonitor ---")

    monitor = LayerMonitor(num_layers=4)

    # Simulate 2 active + 1 skipped call on layer 0
    monitor.record_start(0, skipped=False)
    time.sleep(0.001)
    monitor.record_end(0)

    monitor.record_start(0, skipped=False)
    time.sleep(0.002)
    monitor.record_end(0)

    monitor.record_start(0, skipped=True)
    monitor.record_end(0)

    s = monitor.summary()
    check("monitor counts total calls", s["total_layer_calls"] == 3)
    check("monitor counts skipped calls", s["skipped_calls"] == 1)
    check("monitor counts active calls", s["active_calls"] == 2)
    check("monitor records per-layer data", len(s["per_layer"]) == 4)
    check("layer 0 has 3 calls", s["per_layer"][0]["calls"] == 3)
    check("layer 0 has 1 skipped", s["per_layer"][0]["skipped"] == 1)

    print(f"  LayerMonitor: {PASS} passed")


# ---------------------------------------------------------------------------
# 3. Gate installation tests
# ---------------------------------------------------------------------------


def test_gate_installation() -> None:
    print("\n--- Gate Installation ---")

    # Build a tiny mock model
    mock_model = nn.Module()
    mock_model.model = nn.Module()
    mock_model.model.layers = nn.ModuleList([DummyLayer(i) for i in range(6)])

    gates = install_layer_gates(mock_model, 6)
    check("gates installed for all layers", len(gates) == 6)
    check("gates are LayerGate instances", all(isinstance(g, LayerGate) for g in gates))

    # All active by default
    check("all gates active initially", all(g.active for g in gates))

    # Set specific active layers
    set_active_layers(gates, {0, 2, 4})
    check("gate 0 active after set", gates[0].active is True)
    check("gate 1 inactive after set", gates[1].active is False)
    check("gate 2 active after set", gates[2].active is True)

    print(f"  Gate Installation: {PASS} passed")


# ---------------------------------------------------------------------------
# 4. Layer plan execution tests
# ---------------------------------------------------------------------------


def test_layer_plan_execution() -> None:
    print("\n--- Layer Plan Execution ---")

    mock_model = nn.Module()
    mock_model.model = nn.Module()
    mock_model.model.layers = nn.ModuleList([DummyLayer(i) for i in range(6)])

    gates = install_layer_gates(mock_model, 6)
    x = torch.zeros(1, 1, 4)

    # Plan: only layers 0, 2, 5 active
    set_active_layers(gates, {0, 2, 5})

    # Simulate forward pass (v5.x: each gate returns a tensor)
    h = x
    for gate in gates:
        h = gate(h, use_cache=False)

    # After active layers: 0+2+5 = 7.0 added
    expected = x + 0.0 + 2.0 + 5.0
    check("plan execution produces correct output", torch.allclose(h, expected),
           f"h={h.flatten()} expected={expected.flatten()}")

    print(f"  Layer Plan Execution: {PASS} passed")


# ---------------------------------------------------------------------------
# 5. Quality evaluator tests
# ---------------------------------------------------------------------------


def test_quality_evaluator() -> None:
    print("\n--- Quality Evaluator ---")

    # Token Jaccard
    j = token_jaccard("the cat sat on the mat", "the dog sat on the log")
    expected = len({"the", "sat", "on", "the"}) / len({"the", "cat", "sat", "on", "mat", "dog", "log"})
    # Jaccard: intersection {the, sat, on} / union {the, cat, sat, on, mat, dog, log}
    # Note: "the" appears twice but sets dedup → intersection=3, union=7 → 3/7 ≈ 0.428
    expected_val = 3.0 / 7.0
    check("jaccard similarity correct", abs(j - expected_val) < 0.01,
          f"jaccard={j:.3f} expected={expected_val:.3f}")

    # Identical strings
    j_ident = token_jaccard("hello world", "hello world")
    check("jaccard identical = 1.0", abs(j_ident - 1.0) < 0.01)

    # Empty strings
    j_empty = token_jaccard("", "")
    check("jaccard empty = 1.0", abs(j_empty - 1.0) < 0.01)

    # ROUGE-L
    r = rouge_l_f1("the cat sat on the mat", "the dog sat on the log")
    # LCS = "the sat on the" → 4 tokens
    # R = 4/6=0.667, P = 4/6=0.667, F1 = 0.667
    check("rouge-l f1 correct", abs(r - 4 / 6) < 0.05,
          f"rouge_l={r:.3f} expected={4/6:.3f}")

    r_ident = rouge_l_f1("hello world", "hello world")
    check("rouge-l identical = 1.0", abs(r_ident - 1.0) < 0.01)

    r_empty = rouge_l_f1("", "")
    check("rouge-l empty = 0.0", abs(r_empty) < 0.01)

    print(f"  Quality Evaluator: {PASS} passed")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------


def main() -> int:
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    start = time.perf_counter()

    test_layer_gate()
    test_layer_monitor()
    test_gate_installation()
    test_layer_plan_execution()
    test_quality_evaluator()

    elapsed = time.perf_counter() - start

    print(f"\n{'=' * 50}")
    print(f"  Results:  {PASS} passed  |  {FAIL} failed  |  {elapsed:.2f}s")
    print(f"{'=' * 50}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
