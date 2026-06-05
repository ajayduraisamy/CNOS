"""runtime  CNOS v0.7 Integration Engine master coordinator.

Orchestrates all CNOS subsystems into a unified processing pipeline:

  1. Query  receive user input
  2. Route  analyse complexity, select layers (RoutingController)
  3. Load  ensure layers reside in fast memory (MemoryController)
  4. Infer  execute model with selected layers (ModelAdapter)
  5. Cache  compress KV cache entries (CacheController)
  6. Report  produce structured result with all metrics
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

_PROTO = os.path.join(os.path.dirname(__file__), "..")
for _dir in ("neural_paging", "layer_router", "kv_cache_compression",
             "memory_virtualization", "real_inference"):
    _p = os.path.join(_PROTO, _dir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    """Configuration for the CNOS runtime.

    Attributes:
        model_key: Model identifier (``"tinyllama"``, ``"qwen-1.5b"``).
        ram_gb: CPU RAM capacity in GB.
        ssd_gb: SSD swap capacity in GB.
        page_size: Virtual memory page size in bytes.
        eviction_policy: Page eviction strategy.
        prefetch_enabled: Enable automatic prefetching.
        routing_policy: Layer routing strategy.
        quantisation: KV cache compression scheme.
        max_tokens: Maximum generation tokens.
        temperature: Sampling temperature.
        mode: ``"simulate"`` (no real model) or ``"real"`` (requires download).
    """
    model_key: str = "tinyllama"
    ram_gb: float = 4.0
    ssd_gb: float = 100.0
    page_size: int = 1024 * 1024
    eviction_policy: str = "lru"
    prefetch_enabled: bool = True
    routing_policy: str = "adaptive"
    quantisation: str = "int8"
    max_tokens: int = 256
    temperature: float = 0.7
    mode: str = "simulate"


@dataclass
class CnosResult:
    """Structured output from a single CNOS processing pipeline run.

    Attributes:
        query: Original user query.
        response: Generated response text.
        routing: Routing result with complexity and layer selection.
        memory: Memory metrics snapshot.
        cache: Cache compression metrics.
        latency_s: Total end-to-end latency in seconds.
        tokens_generated: Number of tokens generated.
        layers_executed: Number of layers that actually ran.
        layers_skipped: Number of layers bypassed by routing.
        compute_reduction_pct: Fraction of layers skipped.
        pipeline_times_s: Per-stage timing breakdown.
        timestamp: ISO-8601 timestamp.
    """
    query: str = ""
    response: str = ""
    routing: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, Any]] = None
    cache: Optional[Dict[str, Any]] = None
    latency_s: float = 0.0
    tokens_generated: int = 0
    layers_executed: int = 0
    layers_skipped: int = 0
    compute_reduction_pct: float = 0.0
    pipeline_times_s: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "latency_s": round(self.latency_s, 4),
            "tokens_generated": self.tokens_generated,
            "layers_executed": self.layers_executed,
            "layers_skipped": self.layers_skipped,
            "compute_reduction_pct": round(self.compute_reduction_pct, 1),
            "routing": self.routing or {},
            "memory": self.memory or {},
            "cache": self.cache or {},
            "pipeline_times_s": {
                k: round(v, 4) for k, v in self.pipeline_times_s.items()
            },
            "timestamp": self.timestamp,
        }

    def to_markdown_row(self) -> str:
        mem = self.memory or {}
        rout = self.routing or {}
        cache = self.cache or {}
        return (
            f"| {self.query[:40]:40s} "
            f"| {self.layers_executed:3d}/{self.layers_executed + self.layers_skipped:3d} "
            f"| {round(self.compute_reduction_pct, 1):5.1f}% "
            f"| {mem.get('ram_used_gb', 0):6.2f} GB "
            f"| {mem.get('page_faults', 0):4d}/{mem.get('page_hits', 0):4d} "
            f"| {mem.get('hit_rate_pct', 0):5.1f}% "
            f"| {cache.get('compression_ratio', 1):5.2f}x "
            f"| {round(self.latency_s, 3):6.3f}s |"
        )


class CnosRuntime:
    """Master coordinator that connects all CNOS subsystems.

    Typical usage::

        rt = CnosRuntime(RuntimeConfig(model_key="tinyllama", ram_gb=4))
        result = rt.process("Explain quantum entanglement")
        print(result.to_dict())

    Args:
        config: Runtime configuration.
    """

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()
        self._t_start: float = 0.0
        self._init_subsystems()

    def _init_subsystems(self) -> None:
        from model_adapter import create_model_adapter
        from routing_controller import RoutingController
        from memory_controller import MemoryController
        from cache_controller import CacheController

        num_layers = self._resolve_layers()

        self.routing = RoutingController(
            num_layers=num_layers,
            policy_name=self.config.routing_policy,
        )
        self.memory = MemoryController(
            ram_gb=self.config.ram_gb,
            ssd_gb=self.config.ssd_gb,
            page_size=self.config.page_size,
            num_layers=num_layers,
            eviction_policy=self.config.eviction_policy,
            prefetch_enabled=self.config.prefetch_enabled,
        )
        self.cache = CacheController(
            num_layers=num_layers,
            quantisation=self.config.quantisation,
        )
        self.model = create_model_adapter(
            model_key=self.config.model_key,
            mode=self.config.mode,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            ram_mb=self.config.ram_gb * 1024,
        )

        logger.info(
            "CnosRuntime ready  model=%s mode=%s ram=%.1fGB "
            "routing=%s quant=%s evict=%s layers=%d",
            self.config.model_key, self.config.mode, self.config.ram_gb,
            self.config.routing_policy, self.config.quantisation,
            self.config.eviction_policy, num_layers,
        )

    def _resolve_layers(self) -> int:
        from model_adapter import _MODEL_LAYERS
        return _MODEL_LAYERS.get(self.config.model_key, 22)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def process(self, query: str) -> CnosResult:
        """Run the full CNOS pipeline on a single query.

        Args:
            query: User input text.

        Returns:
            A :class:`CnosResult` with response and all metrics.
        """
        times: Dict[str, float] = {}
        t0 = time.perf_counter()
        self._t_start = t0

        route_result = self._route(query)
        times["routing"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        mem_metrics = self._prepare_memory(route_result.selected_layers)
        times["memory"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        gen_result = self._infer(query, route_result.selected_layers)
        times["inference"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        cache_metrics = self._compress_cache()
        times["cache"] = time.perf_counter() - t3

        total = time.perf_counter() - t0

        return CnosResult(
            query=query,
            response=gen_result.get("response", ""),
            routing=route_result.summary(),
            memory=mem_metrics.summary() if mem_metrics else {},
            cache=cache_metrics.summary() if cache_metrics else {},
            latency_s=total,
            tokens_generated=gen_result.get("tokens", 0),
            layers_executed=len(route_result.selected_layers),
            layers_skipped=route_result.num_skipped,
            compute_reduction_pct=route_result.compute_reduction_pct,
            pipeline_times_s=times,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    def process_baseline(self, query: str) -> CnosResult:
        """Run the pipeline with all layers active (no routing optimization).

        Provides a fair baseline for comparison.
        """
        t0 = time.perf_counter()
        gen_result = self._infer_baseline(query)
        infer_time = time.perf_counter() - t0

        return CnosResult(
            query=query,
            response=gen_result.get("response", ""),
            latency_s=infer_time,
            tokens_generated=gen_result.get("tokens", 0),
            layers_executed=self._resolve_layers(),
            layers_skipped=0,
            compute_reduction_pct=0.0,
            pipeline_times_s={"inference": infer_time},
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _route(self, query: str) -> Any:
        return self.routing.select_layers(query)

    def _prepare_memory(self, layer_ids: List[int]) -> Any:
        self.memory.prepare_layers(set(layer_ids))
        return self.memory.get_metrics()

    def _infer(self, query: str, layer_ids: List[int]) -> Dict[str, Any]:
        result = self.model.generate(query, set(layer_ids))
        return {
            "response": result.response,
            "tokens": result.tokens_generated,
            "latency": result.latency_s,
        }

    def _infer_baseline(self, query: str) -> Dict[str, Any]:
        result = self.model.generate_baseline(query)
        return {
            "response": result.response,
            "tokens": result.tokens_generated,
            "latency": result.latency_s,
        }

    def _compress_cache(self) -> Any:
        return self.cache.compress()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        self.model.cleanup()
        self.cache.clear()
        self.memory.reset()
        logger.info("CnosRuntime cleaned up")


def create_runtime(
    model_key: str = "tinyllama",
    ram_gb: float = 4.0,
    mode: str = "simulate",
    **kwargs: Any,
) -> CnosRuntime:
    """Convenience factory for :class:`CnosRuntime`.

    Args:
        model_key: Model identifier.
        ram_gb: RAM in GB.
        mode: ``"simulate"`` or ``"real"``.
        **kwargs: Additional :class:`RuntimeConfig` fields.

    Returns:
        Configured :class:`CnosRuntime` instance.
    """
    config = RuntimeConfig(
        model_key=model_key,
        ram_gb=ram_gb,
        mode=mode,
        **{k: v for k, v in kwargs.items() if hasattr(RuntimeConfig, k)},
    )
    return CnosRuntime(config)
