"""model_adapter  unified model interface supporting TinyLlama and Qwen 1.5B.

Two modes:
  * **real**  delegates to ``real_inference.RoutedInferenceEngine`` (requires
    HuggingFace model download).
  * **simulate**  uses ``neural_paging`` to simulate layer execution for
    benchmarking without a real model.
"""

from __future__ import annotations

import abc
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

_PROTO = os.path.join(os.path.dirname(__file__), "..")
for _dir in ("neural_paging", "real_inference"):
    _p = os.path.join(_PROTO, _dir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

_MODEL_LAYERS: Dict[str, int] = {
    "tinyllama": 22,
    "qwen-1.5b": 28,
    "llama-3.2-1b": 16,
}

_MODEL_NAMES: Dict[str, str] = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
}


@dataclass
class GenerationResult:
    response: str
    tokens_generated: int
    latency_s: float
    layers_executed: int
    layers_skipped: int
    peak_memory_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "response": self.response[:80] + "..." if len(self.response) > 80 else self.response,
            "tokens": self.tokens_generated,
            "latency_s": round(self.latency_s, 3),
            "layers_executed": self.layers_executed,
            "layers_skipped": self.layers_skipped,
        }


class ModelAdapter(abc.ABC):
    """Abstract interface for model inference."""

    @abc.abstractmethod
    def generate(self, query: str, active_layers: Set[int]) -> GenerationResult:
        ...

    @abc.abstractmethod
    def generate_baseline(self, query: str) -> GenerationResult:
        ...

    @abc.abstractmethod
    def cleanup(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def num_layers(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        ...


class RealModelAdapter(ModelAdapter):
    """Wraps ``real_inference.RoutedInferenceEngine`` for real model execution.

    Requires the model to be downloaded from HuggingFace Hub.
    """

    def __init__(
        self,
        model_key: str = "tinyllama",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> None:
        self.model_key = model_key
        self._num_layers = _MODEL_LAYERS.get(model_key, 22)

        from model_loader import load_model
        from routed_inference import RoutedInferenceEngine

        logger.info("Loading real model %s ...", model_key)
        self.bundle = load_model(model_key)
        self.engine = RoutedInferenceEngine(
            bundle=self.bundle,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        logger.info("Model loaded: %s (%d layers)", model_key, self._num_layers)

    def generate(self, query: str, active_layers: Set[int]) -> GenerationResult:
        start = time.perf_counter()
        response, metrics = self.engine.generate(query, active_layers)
        elapsed = time.perf_counter() - start

        return GenerationResult(
            response=response,
            tokens_generated=metrics.num_tokens_generated,
            latency_s=elapsed,
            layers_executed=metrics.routed_layers,
            layers_skipped=metrics.layers_skipped,
            peak_memory_mb=metrics.peak_memory_mb,
            extra={"baseline_latency_s": metrics.baseline_latency_s,
                   "routed_latency_s": metrics.routed_latency_s},
        )

    def generate_baseline(self, query: str) -> GenerationResult:
        start = time.perf_counter()
        response, metrics = self.engine.generate_baseline(query)
        elapsed = time.perf_counter() - start

        return GenerationResult(
            response=response,
            tokens_generated=metrics.num_tokens_generated,
            latency_s=elapsed,
            layers_executed=metrics.baseline_layers,
            layers_skipped=0,
            peak_memory_mb=metrics.peak_memory_mb,
            extra={"baseline_latency_s": metrics.baseline_latency_s},
        )

    def cleanup(self) -> None:
        self.engine.cleanup()
        logger.info("RealModelAdapter cleaned up")

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def model_name(self) -> str:
        return _MODEL_NAMES.get(self.model_key, self.model_key)


class SimulatedModelAdapter(ModelAdapter):
    """Simulates model inference using ``neural_paging`` components.

    Does NOT require a real model download; uses synthetic ``LayerStore``
    layers and ``NeuralPager`` to simulate latency and memory costs.
    """

    def __init__(
        self,
        model_key: str = "tinyllama",
        max_tokens: int = 256,
        temperature: float = 0.7,
        ram_mb: float = 4096.0,
    ) -> None:
        self.model_key = model_key
        self._num_layers = _MODEL_LAYERS.get(model_key, 22)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model_name = _MODEL_NAMES.get(model_key, model_key)

        from layer_store import LayerStore
        from cache_manager import CacheManager
        from prefetcher import Prefetcher, PrefetchStrategy
        from pager import NeuralPager

        self.layer_store = LayerStore(num_layers=self._num_layers, seed=42)
        self.cache_manager = CacheManager(max_ram_mb=ram_mb)
        self.prefetcher = Prefetcher(
            strategy=PrefetchStrategy.SEQUENTIAL,
            num_layers=self._num_layers,
        )
        self.pager = NeuralPager(
            layer_store=self.layer_store,
            cache_manager=self.cache_manager,
            prefetcher=self.prefetcher,
        )

        self._total_accesses = 0

    def generate(self, query: str, active_layers: Set[int]) -> GenerationResult:
        start = time.perf_counter()
        response_parts: List[str] = []

        sim_sequence = sorted(active_layers)
        for layer_id in sim_sequence:
            self.pager.access_layer(layer_id)

        sim_tokens = max(1, self.max_tokens // 4)
        for _ in range(sim_tokens):
            for layer_id in list(active_layers)[:4]:
                self.pager.access_layer(layer_id)

        elapsed = time.perf_counter() - start

        return GenerationResult(
            response=f"[simulated {self.model_key}] {query[:50]}...",
            tokens_generated=sim_tokens,
            latency_s=elapsed,
            layers_executed=len(active_layers),
            layers_skipped=self._num_layers - len(active_layers),
            peak_memory_mb=self.cache_manager.current_usage_mb,
            extra={
                "page_faults": self.pager.metrics.cache_misses,
                "page_hits": self.pager.metrics.cache_hits,
                "cache_usage_mb": self.cache_manager.current_usage_mb,
            },
        )

    def generate_baseline(self, query: str) -> GenerationResult:
        all_layers = set(range(self._num_layers))
        return self.generate(query, all_layers)

    def cleanup(self) -> None:
        self.cache_manager.clear()
        logger.info("SimulatedModelAdapter cleaned up")

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def model_name(self) -> str:
        return self._model_name


def create_model_adapter(
    model_key: str = "tinyllama",
    mode: str = "simulate",
    max_tokens: int = 256,
    temperature: float = 0.7,
    ram_mb: float = 4096.0,
) -> ModelAdapter:
    """Factory function  create a ``ModelAdapter`` for the given mode.

    Args:
        model_key: One of ``"tinyllama"``, ``"qwen-1.5b"``, ``"llama-3.2-1b"``.
        mode: ``"real"`` or ``"simulate"``.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        ram_mb: Simulated RAM in MB (simulate mode only).

    Returns:
        A configured :class:`ModelAdapter` instance.

    Raises:
        ValueError: On unknown model_key or mode.
    """
    if model_key not in _MODEL_LAYERS:
        raise ValueError(f"Unknown model_key={model_key!r}. "
                         f"Options: {list(_MODEL_LAYERS)}")

    if mode == "real":
        return RealModelAdapter(
            model_key=model_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif mode == "simulate":
        return SimulatedModelAdapter(
            model_key=model_key,
            max_tokens=max_tokens,
            temperature=temperature,
            ram_mb=ram_mb,
        )
    else:
        raise ValueError(f"Unknown mode={mode!r}. Use 'real' or 'simulate'.")
