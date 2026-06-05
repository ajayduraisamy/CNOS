"""kv_cache — manages transformer key-value cache with compression support.

Tracks per-layer, per-head keys and values, token positions, and memory
usage across multiple quantisation levels.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DTYPE_BYTES: Dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
    torch.quint8: 1,
    torch.qint8: 1,
    torch.quint4x2: 1,  # 2 int4 values packed per byte
}


# ---------------------------------------------------------------------------
# KV Cache Entry (single layer)
# ---------------------------------------------------------------------------


@dataclass
class KVCacheEntry:
    """Keys, values, and metadata for one transformer layer.

    Attributes:
        keys: Tensor of shape ``(num_heads, seq_len, head_dim)``.
        values: Tensor of shape ``(num_heads, seq_len, head_dim)``.
        token_positions: Absolute position of each token in the sequence.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        dtype: Element data type (after quantisation).
        quantisation: Quantisation scheme name (``"fp16"``, ``"int8"``,
            ``"int4"``).
        scale: Scale factor for quantised tensors (scalar or per-token).
        zero_point: Zero-point for quantised tensors.
        access_count: Number of times this entry has been accessed (for LFU).
        last_access_time: Step of last access (for LRU).
    """

    keys: torch.Tensor
    values: torch.Tensor
    token_positions: List[int] = field(default_factory=list)
    num_heads: int = 0
    head_dim: int = 0
    dtype: torch.dtype = torch.float16
    quantisation: str = "fp16"
    scale: Optional[torch.Tensor] = None
    zero_point: Optional[torch.Tensor] = None
    access_count: int = 0
    last_access_time: int = 0

    def __post_init__(self) -> None:
        if self.keys.dim() >= 2:
            self.num_heads = self.keys.shape[0]
            self.head_dim = self.keys.shape[-1]
        if len(self.token_positions) == 0 and self.keys.dim() >= 2:
            seq_len = self.keys.shape[1]
            self.token_positions = list(range(seq_len))

    @property
    def seq_len(self) -> int:
        """Number of cached tokens for this layer."""
        return self.keys.shape[1] if self.keys.dim() >= 2 else 0

    @property
    def memory_bytes(self) -> int:
        """Memory consumed by keys + values in bytes."""
        elem_bytes = DTYPE_BYTES.get(self.keys.dtype, 2)
        return self.keys.numel() * elem_bytes + self.values.numel() * elem_bytes

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        position: int,
    ) -> None:
        """Append a single token's key/value at the given absolute position."""
        self.keys = torch.cat([self.keys, key], dim=1)
        self.values = torch.cat([self.values, value], dim=1)
        self.token_positions.append(position)

    def prune_to(self, keep_indices: List[int]) -> None:
        """Keep only tokens at *keep_indices* (list of column indices)."""
        if not keep_indices:
            self.keys = self.keys[:, :0, :]
            self.values = self.values[:, :0, :]
            self.token_positions = []
            return
        idx = torch.tensor(keep_indices, device=self.keys.device, dtype=torch.long)
        self.keys = self.keys.index_select(1, idx)
        self.values = self.values.index_select(1, idx)
        self.token_positions = [self.token_positions[i] for i in keep_indices]

    def clear(self) -> None:
        """Remove all cached tokens."""
        self.keys = self.keys[:, :0, :]
        self.values = self.values[:, :0, :]
        self.token_positions = []
        self.access_count = 0
        self.last_access_time = 0


# ---------------------------------------------------------------------------
# KV Cache Manager (all layers)
# ---------------------------------------------------------------------------


class KVCacheManager:
    """Manages KV caches across all transformer layers.

    Args:
        num_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads per layer.
        head_dim: Dimension of each attention head.
        max_cache_len: Maximum tokens to cache per layer before eviction.
        device: Torch device for cache tensors.
        quantisation: Default quantisation scheme.
    """

    def __init__(
        self,
        num_layers: int = 22,
        num_heads: int = 32,
        head_dim: int = 64,
        max_cache_len: int = 4096,
        device: Optional[torch.device] = None,
        quantisation: str = "fp16",
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len
        self.device = device or torch.device("cpu")
        self.quantisation = quantisation
        self.total_tokens_processed = 0

        self.entries: List[KVCacheEntry] = []
        for _ in range(num_layers):
            empty_kv = torch.zeros(num_heads, 0, head_dim, device=self.device)
            self.entries.append(
                KVCacheEntry(keys=empty_kv.clone(), values=empty_kv.clone())
            )

        logger.info(
            "KVCacheManager  %d layers  %d heads  dim=%d  max_len=%d  q=%s",
            num_layers, num_heads, head_dim, max_cache_len, quantisation,
        )

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    def append(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        position: Optional[int] = None,
    ) -> None:
        """Append a single token's key/value pairs for one layer.

        Args:
            layer_idx: Which layer to append to.
            key: Shape ``(num_heads, 1, head_dim)`` or ``(1, num_heads, head_dim)``.
            value: Same shape as *key*.
            position: Absolute token position (auto-assigned if None).
        """
        if position is None:
            position = self.total_tokens_processed
        self.total_tokens_processed += 1

        entry = self.entries[layer_idx]
        k = key.squeeze(1).unsqueeze(1) if key.dim() == 3 and key.shape[1] != 1 else key
        v = value.squeeze(1).unsqueeze(1) if value.dim() == 3 and value.shape[1] != 1 else value
        entry.append(k, v, position)

    def get_layer_cache(self, layer_idx: int) -> KVCacheEntry:
        """Return the cache entry for a specific layer."""
        self.entries[layer_idx].access_count += 1
        return self.entries[layer_idx]

    def clear(self) -> None:
        """Reset all caches."""
        for entry in self.entries:
            entry.clear()
        self.total_tokens_processed = 0

    # ------------------------------------------------------------------
    # Memory metrics
    # ------------------------------------------------------------------

    @property
    def total_memory_bytes(self) -> int:
        """Total memory consumed by all layer caches."""
        return sum(e.memory_bytes for e in self.entries)

    @property
    def total_memory_mb(self) -> float:
        """Total memory in mebibytes."""
        return self.total_memory_bytes / (1024 ** 2)

    @property
    def total_cached_tokens(self) -> int:
        """Sum of cached tokens across all layers."""
        return sum(e.seq_len for e in self.entries)

    @property
    def compression_ratio(self) -> float:
        """Ratio of original dtype (FP32) to current dtype size."""
        if self.quantisation == "fp16":
            return 2.0
        if self.quantisation == "int8":
            return 4.0
        if self.quantisation == "int4":
            return 8.0
        return 1.0

    @property
    def memory_saved_mb(self) -> float:
        """Memory saved relative to FP32 baseline."""
        fp32_bytes_per_elem = 4
        current_bytes_per_elem = DTYPE_BYTES.get(
            self.entries[0].keys.dtype if self.entries else torch.float16, 2
        )
        total_elems = sum(
            e.keys.numel() + e.values.numel() for e in self.entries
        )
        baseline = total_elems * fp32_bytes_per_elem
        current = total_elems * current_bytes_per_elem
        return (baseline - current) / (1024 ** 2)

    def summary(self) -> Dict[str, object]:
        """Return a dictionary of cache statistics."""
        return {
            "num_layers": self.num_layers,
            "total_cached_tokens": self.total_cached_tokens,
            "total_memory_mb": round(self.total_memory_mb, 2),
            "memory_saved_mb": round(self.memory_saved_mb, 2),
            "compression_ratio": self.compression_ratio,
            "quantisation": self.quantisation,
            "max_cache_len": self.max_cache_len,
        }

    def print_summary(self) -> None:
        """Print a human-readable cache summary."""
        s = self.summary()
        print("\n" + "=" * 50)
        print("  KV Cache Summary")
        print("=" * 50)
        print(f"  Layers:              {s['num_layers']}")
        print(f"  Cached tokens:       {s['total_cached_tokens']}")
        print(f"  Total memory:        {s['total_memory_mb']:.2f} MB")
        print(f"  Memory saved:        {s['memory_saved_mb']:.2f} MB")
        print(f"  Compression ratio:   {s['compression_ratio']:.1f}x")
        print(f"  Quantisation:        {s['quantisation']}")
        print("=" * 50)
