"""routed_inference — runs a transformer model with selective layer execution.

Integrates the CNOS Layer Router (v0.3) with a real HuggingFace model
by accepting a layer plan, configuring the :class:`LayerGate` wrappers,
and executing inference while collecting detailed metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

from layer_hooks import (
    LayerGate,
    LayerMonitor,
    install_layer_gates,
    set_active_layers,
)
from model_loader import ModelBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------


@dataclass
class InferenceMetrics:
    """Per-query inference metrics.

    Attributes:
        query: The input text.
        response: The generated output text.
        baseline_latency_s: Full-model inference time (seconds).
        routed_latency_s: Routed inference time (seconds).
        baseline_layers: All layers (typically ``num_layers``).
        routed_layers: Number of layers actually executed.
        layers_skipped: Number of layers skipped.
        compute_reduction_pct: Percentage of layers skipped.
        peak_memory_mb: Peak CUDA memory during inference.
        num_tokens_generated: Number of output tokens.
        latency_per_token_ms: Average ms per output token.
    """

    query: str = ""
    response: str = ""
    baseline_latency_s: float = 0.0
    routed_latency_s: float = 0.0
    baseline_layers: int = 0
    routed_layers: int = 0
    layers_skipped: int = 0
    compute_reduction_pct: float = 0.0
    peak_memory_mb: float = 0.0
    num_tokens_generated: int = 0
    latency_per_token_ms: float = 0.0


# ---------------------------------------------------------------------------
# RoutedInferenceEngine
# ---------------------------------------------------------------------------


class RoutedInferenceEngine:
    """Wraps a model with gated layers for selective inference.

    Args:
        bundle: A :class:`ModelBundle` from :func:`model_loader.load_model`.
        max_new_tokens: Maximum tokens to generate per query.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> None:
        self.bundle = bundle
        self.model = bundle.model
        self.tokenizer = bundle.tokenizer
        self.device = bundle.device
        self.num_layers = bundle.num_layers
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Install layer gates once
        self.monitor = LayerMonitor(self.num_layers)
        self.gates = install_layer_gates(self.model, self.num_layers, self.monitor)
        logger.info(
            "RoutedInferenceEngine ready  —  %d layers, max_new_tokens=%d, temp=%.2f",
            self.num_layers,
            max_new_tokens,
            temperature,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        active_layers: Optional[Set[int]] = None,
    ) -> tuple[str, InferenceMetrics]:
        """Generate a response, optionally with selective layers.

        Args:
            query: Input prompt.
            active_layers: Set of layer indices to execute.
                ``None`` = all layers (baseline).

        Returns:
            ``(generated_text, metrics)``.
        """
        if active_layers is None:
            active_layers = set(range(self.num_layers))

        # Reset monitor for this query
        self.monitor = LayerMonitor(self.num_layers)

        # Configure gates
        set_active_layers(self.gates, active_layers or set(range(self.num_layers)))

        # Tokenize
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)

        # Measure peak memory before
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Run inference
        start = time.perf_counter()
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - start

        # Decode
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Build metrics
        num_gen = len(generated_ids)
        routed_layers = len(active_layers) if active_layers else self.num_layers
        skipped = self.num_layers - routed_layers
        reduction = (skipped / self.num_layers) * 100.0

        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

        metrics = InferenceMetrics(
            query=query,
            response=response,
            baseline_latency_s=elapsed,
            routed_latency_s=elapsed,
            baseline_layers=self.num_layers,
            routed_layers=routed_layers,
            layers_skipped=skipped,
            compute_reduction_pct=round(reduction, 2),
            peak_memory_mb=round(peak_mem, 1),
            num_tokens_generated=num_gen,
            latency_per_token_ms=(elapsed / max(num_gen, 1)) * 1000,
        )

        return response, metrics

    def generate_baseline(self, query: str) -> tuple[str, InferenceMetrics]:
        """Run full-model inference (all layers) for comparison.

        Returns ``(response, metrics)`` with ``baseline_latency_s`` set.
        """
        response, metrics = self.generate(query, active_layers=set(range(self.num_layers)))
        return response, metrics

    def generate_routed(
        self,
        query: str,
        active_layers: Set[int],
    ) -> tuple[str, InferenceMetrics]:
        """Run inference with a specific layer plan.

        Returns ``(response, metrics)`` with ``routed_latency_s`` set.
        """
        return self.generate(query, active_layers=active_layers)

    def compare(
        self,
        query: str,
        routed_active_layers: Set[int],
    ) -> tuple[InferenceMetrics, InferenceMetrics]:
        """Run both baseline and routed inference on the same query.

        Returns ``(baseline_metrics, routed_metrics)``.
        """
        # Baseline (all layers)
        resp_b, metrics_b = self.generate_baseline(query)

        # Re-init monitor between runs
        self.monitor = LayerMonitor(self.num_layers)

        # Routed
        resp_r, metrics_r = self.generate_routed(query, routed_active_layers)

        # Cross-populate latency fields
        metrics_b.routed_latency_s = metrics_r.routed_latency_s
        metrics_r.baseline_latency_s = metrics_b.baseline_latency_s

        return metrics_b, metrics_r

    def print_comparison(self, baseline: InferenceMetrics, routed: InferenceMetrics) -> None:
        """Print a side-by-side comparison of baseline vs. routed inference."""
        print("\n" + "=" * 60)
        print("  CNOS Routed Inference — Baseline vs. Routed")
        print("=" * 60)
        print(f"  Query:                {baseline.query[:80]}...")
        print(f"  Response (routed):    {routed.response[:120]}...")
        print("-" * 60)
        print(f"  {'Metric':<30} {'Baseline':<15} {'Routed':<15}")
        print(f"  {'-'*28} {'-'*13} {'-'*13}")
        print(f"  {'Layers executed':<30} {baseline.baseline_layers:<15} {routed.routed_layers:<15}")
        print(f"  {'Layers skipped':<30} {baseline.layers_skipped:<15} {routed.layers_skipped:<15}")
        print(f"  {'Compute reduction':<30} {'0.00%':<15} {routed.compute_reduction_pct:<14}%")
        print(f"  {'Latency (s)':<30} {baseline.baseline_latency_s:<15.4f} {routed.routed_latency_s:<15.4f}")
        print(f"  {'Tokens generated':<30} {baseline.num_tokens_generated:<15} {routed.num_tokens_generated:<15}")
        print(f"  {'Latency / token (ms)':<30} {baseline.latency_per_token_ms:<15.2f} {routed.latency_per_token_ms:<15.2f}")
        print(f"  {'Peak memory (MB)':<30} {baseline.peak_memory_mb:<15.1f} {routed.peak_memory_mb:<15.1f}")
        print("=" * 60)

    def cleanup(self) -> None:
        """Remove layer gates and free memory."""
        from layer_hooks import uninstall_layer_gates
        uninstall_layer_gates(self.model, self.gates)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Engine cleaned up")
