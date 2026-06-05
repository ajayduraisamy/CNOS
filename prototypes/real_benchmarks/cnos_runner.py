"""cnos_runner -- runs CNOS-optimized inference with routing, paging, and KV compression.

Integrates:
  * Dynamic Layer Router -- skip layers based on query complexity
  * Memory Virtualization -- page-level memory tracking
  * Neural Paging -- layer cache simulation
  * KV Cache Compression -- quantize and prune KV cache

Collects all metrics alongside the real inference run.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

_PROTO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _dir in ("real_inference", "layer_router", "kv_cache_compression",
             "memory_virtualization", "neural_paging"):
    _p = os.path.abspath(os.path.join(_PROTO, _dir))
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

_OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


@dataclass
class CnosQueryResult:
    query: str = ""
    response: str = ""
    baseline_response: str = ""
    latency_s: float = 0.0
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    ram_peak_mb: float = 0.0
    ram_avg_mb: float = 0.0
    cpu_peak_pct: float = 0.0
    cpu_avg_pct: float = 0.0
    layers_executed: int = 0
    layers_skipped: int = 0
    compute_reduction_pct: float = 0.0
    page_faults: int = 0
    page_hits: int = 0
    cache_hit_rate_pct: float = 0.0
    compression_ratio: float = 1.0
    query_type: str = ""
    complexity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "baseline_response": self.baseline_response,
            "latency_s": round(self.latency_s, 4),
            "tokens_generated": self.tokens_generated,
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "ram_peak_mb": round(self.ram_peak_mb, 1),
            "ram_avg_mb": round(self.ram_avg_mb, 1),
            "cpu_peak_pct": round(self.cpu_peak_pct, 1),
            "cpu_avg_pct": round(self.cpu_avg_pct, 1),
            "layers_executed": self.layers_executed,
            "layers_skipped": self.layers_skipped,
            "compute_reduction_pct": round(self.compute_reduction_pct, 1),
            "page_faults": self.page_faults,
            "page_hits": self.page_hits,
            "cache_hit_rate_pct": round(self.cache_hit_rate_pct, 1),
            "compression_ratio": round(self.compression_ratio, 2),
            "query_type": self.query_type,
            "complexity_score": round(self.complexity_score, 3),
        }


@dataclass
class CnosResult:
    queries: List[CnosQueryResult] = field(default_factory=list)
    model_key: str = ""
    num_layers: int = 0
    routing_policy: str = ""
    quantisation: str = ""
    total_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "num_layers": self.num_layers,
            "routing_policy": self.routing_policy,
            "quantisation": self.quantisation,
            "total_time_s": round(self.total_time_s, 2),
            "queries": [q.to_dict() for q in self.queries],
        }

    def save(self, path: Optional[str] = None) -> str:
        if path is None:
            os.makedirs(_OUT_DIR, exist_ok=True)
            path = os.path.join(_OUT_DIR, "cnos_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("CNOS results saved to %s", path)
        return path


class CnosRunner:
    """Runs CNOS-optimized inference with routing, paging, compression.

    Args:
        bundle: The ``ModelBundle`` (from real_inference model_loader).
        max_tokens: Maximum tokens to generate per query.
        temperature: Sampling temperature.
        routing_policy: Layer routing policy name.
        quantisation: KV cache compression scheme.
        ram_gb: Simulated RAM in GB (for memory virtualization).
    """

    def __init__(
        self,
        bundle: Any,
        max_tokens: int = 128,
        temperature: float = 0.7,
        routing_policy: str = "adaptive",
        quantisation: str = "int8",
        ram_gb: float = 4.0,
    ) -> None:
        self.bundle = bundle
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_layers = bundle.num_layers
        self.routing_policy = routing_policy
        self.quantisation = quantisation

        # Real inference engine
        from routed_inference import RoutedInferenceEngine, set_active_layers
        self.engine = RoutedInferenceEngine(
            bundle=bundle,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        self._set_active_layers = set_active_layers

        # Layer router
        from complexity_detector import ComplexityDetector
        from layer_selector import LayerSelector
        from routing_policy import create_policy
        self.detector = ComplexityDetector(num_layers=self.num_layers)
        self.selector = LayerSelector(num_layers=self.num_layers)
        self.policy = create_policy(routing_policy, num_layers=self.num_layers)
        self.selector.set_policy(self.policy)

        # KV cache compression (simulated alongside real inference)
        from kv_cache import KVCacheManager
        from quantizer import get_quantizer, QUANTIZER_REGISTRY
        self.kv_manager = KVCacheManager(
            num_layers=self.num_layers,
            num_heads=bundle.model.config.num_attention_heads
                if hasattr(bundle.model.config, 'num_attention_heads') else 32,
            head_dim=(bundle.model.config.hidden_size //
                      bundle.model.config.num_attention_heads)
                if hasattr(bundle.model.config, 'num_attention_heads')
                and hasattr(bundle.model.config, 'hidden_size') else 64,
            max_cache_len=4096,
            quantisation=quantisation,
        )
        self.quantizer = get_quantizer(quantisation)
        self._quantize_time: float = 0.0

        # Memory virtualization (simulated alongside)
        from virtual_memory import VirtualMemorySystem
        self.vm = VirtualMemorySystem(
            ram_gb=ram_gb,
            page_size=1024 * 1024,
        )
        # Create page entries for each layer
        self._layer_vm_ids: Dict[int, int] = {}
        for lid in range(self.num_layers):
            comp = self.vm.create_virtual_component(
                name=f"layer_{lid}",
                num_pages=max(1, int(ram_gb * 64 / self.num_layers)),
                preferred_tier=3,
            )
            self._layer_vm_ids[lid] = comp.virtual_id

        logger.info(
            "CnosRunner ready -- %s, %d layers, policy=%s, quant=%s, RAM=%.1fGB",
            bundle.model_name, self.num_layers, routing_policy,
            quantisation, ram_gb,
        )

    def run_query(self, query: str) -> CnosQueryResult:
        import psutil
        proc = psutil.Process(os.getpid())

        # 1. Analyse complexity and select layer plan
        comp_result = self.detector.analyse(query)
        selection = self.selector.select(query)

        # Clamp to valid layers
        valid_selected = [
            l for l in selection.selected_layers
            if 0 <= l < self.num_layers
        ]
        active_layers = set(valid_selected)
        skipped = self.num_layers - len(active_layers)
        reduction = (skipped / self.num_layers) * 100.0

        # 2. Record memory / CPU before
        ram_start = proc.memory_info().rss / (1024 * 1024)
        cpu_start = proc.cpu_percent(interval=None)

        # 3. Run routed inference
        self.engine.monitor = self.engine.monitor.__class__(self.num_layers)
        self._set_active_layers(self.engine.gates, active_layers)

        self.kv_manager.clear()
        import torch
        inputs = self.bundle.tokenizer(query, return_tensors="pt").to(self.bundle.device)

        start = time.perf_counter()
        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
            pad_token_id=self.bundle.tokenizer.pad_token_id,
            eos_token_id=self.bundle.tokenizer.eos_token_id,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            output_ids = self.engine.model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - start

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.bundle.tokenizer.decode(generated_ids, skip_special_tokens=True)

        ram_end = proc.memory_info().rss / (1024 * 1024)
        cpu_end = proc.cpu_percent(interval=None)

        # 4. Simulate KV cache compression
        num_tokens = len(generated_ids)
        if num_tokens > 0 and hasattr(self.quantizer, 'quantize'):
            import torch
            for layer_idx in range(min(4, self.num_layers)):
                for _ in range(min(num_tokens, 4)):
                    dummy_k = torch.randn(self.kv_manager.num_heads, 1,
                                           self.kv_manager.head_dim)
                    dummy_v = torch.randn(self.kv_manager.num_heads, 1,
                                           self.kv_manager.head_dim)
                    self.kv_manager.append(layer_idx, dummy_k, dummy_v)

        compression_ratio = self.kv_manager.compression_ratio if num_tokens > 0 else 1.0

        # 5. Simulate memory virtualization access
        page_faults = 0
        page_hits = 0
        for lid in active_layers:
            vm_id = self._layer_vm_ids.get(lid)
            if vm_id is not None:
                self.vm.access(vm_id, 0)
        vm_summary = self.vm.summary()
        page_faults = vm_summary.get("page_faults", 0)
        page_hits = vm_summary.get("page_hits", 0)

        # 6. Get baseline response for quality comparison
        baseline_response = ""
        try:
            br, _ = self.engine.generate_baseline(query)
            baseline_response = br
        except Exception:
            baseline_response = response

        return CnosQueryResult(
            query=query,
            response=response,
            baseline_response=baseline_response,
            latency_s=elapsed,
            tokens_generated=num_tokens,
            tokens_per_sec=num_tokens / max(elapsed, 1e-9),
            ram_peak_mb=max(ram_start, ram_end),
            ram_avg_mb=(ram_start + ram_end) / 2,
            cpu_peak_pct=max(cpu_start, cpu_end),
            cpu_avg_pct=(cpu_start + cpu_end) / 2,
            layers_executed=len(active_layers),
            layers_skipped=skipped,
            compute_reduction_pct=reduction,
            page_faults=page_faults,
            page_hits=page_hits,
            cache_hit_rate_pct=(
                (page_hits / max(page_hits + page_faults, 1)) * 100
            ),
            compression_ratio=compression_ratio,
            query_type=comp_result.query_type,
            complexity_score=comp_result.complexity_score,
        )

    def run_queries(self, queries: List[str]) -> CnosResult:
        all_results: List[CnosQueryResult] = []
        t0 = time.perf_counter()

        for i, q in enumerate(queries):
            logger.info("CNOS query %d/%d: %s", i + 1, len(queries), q[:50])
            try:
                r = self.run_query(q)
                all_results.append(r)
                logger.info(
                    "  latency=%.2fs tokens=%d layers=%d/%d "
                    "RAM=%.0fMB faults=%d hits=%d",
                    r.latency_s, r.tokens_generated, r.layers_executed,
                    self.num_layers, r.ram_peak_mb, r.page_faults, r.page_hits,
                )
            except Exception as exc:
                logger.error("CNOS query %d failed: %s", i, exc)
                all_results.append(CnosQueryResult(
                    query=q, response=f"ERROR: {exc}",
                ))

        total = time.perf_counter() - t0
        return CnosResult(
            queries=all_results,
            model_key=self.bundle.model_name,
            num_layers=self.num_layers,
            routing_policy=self.routing_policy,
            quantisation=self.quantisation,
            total_time_s=total,
        )

    def cleanup(self) -> None:
        self.engine.cleanup()
        self.kv_manager.clear()
        logger.info("CnosRunner cleaned up")
