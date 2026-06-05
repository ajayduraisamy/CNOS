"""cache_controller  bridges the KV Cache Compression Engine.

Wraps ``KVCacheManager``, quantizers, pruners, and eviction policies
into a single interface for the CNOS runtime.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

_PROTO = os.path.join(os.path.dirname(__file__), "..", "kv_cache_compression")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    quantisation: str = "fp16"
    pruner: str = "none"
    eviction_policy: str = "none"
    total_tokens: int = 0
    total_memory_mb: float = 0.0
    compression_ratio: float = 1.0
    memory_saved_mb: float = 0.0
    num_evictions: int = 0
    num_prunes: int = 0
    avg_quantize_time_ms: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "quantisation": self.quantisation,
            "pruner": self.pruner,
            "compression_ratio": round(self.compression_ratio, 2),
            "memory_mb": round(self.total_memory_mb, 1),
            "memory_saved_mb": round(self.memory_saved_mb, 1),
            "total_tokens": self.total_tokens,
        }


class CacheController:
    """Manages KV cache with configurable compression, pruning, and eviction.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads per layer.
        head_dim: Dimension per attention head.
        max_cache_len: Maximum cached sequence length.
        quantisation: Compression scheme (``"fp16"``, ``"int8"``, ``"int4"``).
        pruner_name: Pruning strategy (``"oldest_first"``, ``"least_used"``,
            ``"attention_score"``).
        eviction_name: Eviction policy (``"lru"``, ``"lfu"``, ``"adaptive"``).
    """

    def __init__(
        self,
        num_layers: int = 22,
        num_heads: int = 32,
        head_dim: int = 64,
        max_cache_len: int = 4096,
        quantisation: str = "int8",
        pruner_name: str = "oldest_first",
        eviction_name: str = "lru",
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len
        self.quantisation = quantisation
        self.pruner_name = pruner_name
        self.eviction_name = eviction_name

        from kv_cache import KVCacheManager
        from quantizer import get_quantizer
        from pruner import get_pruner
        from eviction_policy import get_eviction_policy

        self.manager = KVCacheManager(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_cache_len=max_cache_len,
            quantisation=quantisation,
        )
        self.quantizer = get_quantizer(quantisation)
        self.pruner = get_pruner(pruner_name)
        self.eviction_policy = get_eviction_policy(eviction_name, self.pruner)

        self._quantize_times: List[float] = []
        self._prune_count: int = 0

        logger.info(
            "CacheController: %d layers, quant=%s, pruner=%s, evict=%s",
            num_layers, quantisation, pruner_name, eviction_name,
        )

    def append(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        position: Optional[int] = None,
    ) -> None:
        self.manager.append(layer_idx, key, value, position)

    def compress(self) -> CacheMetrics:
        import time
        start = time.perf_counter()

        for entry in self.manager.entries:
            if entry is not None and entry.seq_len > 0:
                q_start = time.perf_counter()
                encoded, metadata = self.quantizer.quantize(entry.keys)
                entry.keys = encoded
                entry.dtype = metadata.dtype
                entry.quantisation = metadata.scheme
                self._quantize_times.append((time.perf_counter() - q_start) * 1000)

        elapsed = (time.perf_counter() - start) * 1000
        _ = elapsed

        return CacheMetrics(
            quantisation=self.quantisation,
            pruner=self.pruner_name,
            eviction_policy=self.eviction_name,
            total_tokens=self.manager.total_cached_tokens,
            total_memory_mb=self.manager.total_memory_mb,
            compression_ratio=self.manager.compression_ratio,
            memory_saved_mb=self.manager.memory_saved_mb,
            num_evictions=0,
            num_prunes=self._prune_count,
            avg_quantize_time_ms=(
                sum(self._quantize_times) / len(self._quantize_times)
                if self._quantize_times else 0.0
            ),
        )

    def prune_layer(
        self,
        layer_idx: int,
        target_tokens: int,
    ) -> int:
        entry = self.manager.get_layer_cache(layer_idx)
        if entry.seq_len <= target_tokens:
            return 0
        keep = self.pruner.prune(entry, target_tokens)
        entry.prune_to(keep)
        self._prune_count += 1
        return entry.seq_len - len(keep)

    def clear(self) -> None:
        self.manager.clear()
        self._quantize_times.clear()
        self._prune_count = 0
        logger.info("CacheController cleared")

    @property
    def total_memory_mb(self) -> float:
        return self.manager.total_memory_mb

    @property
    def compression_ratio(self) -> float:
        return self.manager.compression_ratio
