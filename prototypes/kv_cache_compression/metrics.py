"""metrics — tracks KV cache compression performance across runs.

Records compression ratio, memory saved, latency impact, and quality
degradation for each quantisation/pruning/eviction configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompressionRecord:
    """Single benchmark record for one configuration.

    Attributes:
        num_tokens: Sequence length simulated.
        quantisation: Quantisation scheme used.
        pruner: Pruning strategy used.
        eviction_policy: Eviction policy used.
        original_memory_mb: Memory without compression (FP32).
        compressed_memory_mb: Memory after quantisation + pruning.
        compression_ratio: ``original / compressed``.
        memory_saved_mb: ``original - compressed``.
        quantize_time_ms: Time to quantize the cache.
        dequantize_time_ms: Time to dequantize the cache.
        total_cache_time_ms: Total time for cache operations.
        quality_score: Optional output quality (1.0 = perfect).
        num_tokens_pruned: Number of tokens removed by pruning.
    """

    num_tokens: int = 0
    quantisation: str = "fp16"
    pruner: str = "none"
    eviction_policy: str = "none"
    original_memory_mb: float = 0.0
    compressed_memory_mb: float = 0.0
    compression_ratio: float = 1.0
    memory_saved_mb: float = 0.0
    quantize_time_ms: float = 0.0
    dequantize_time_ms: float = 0.0
    total_cache_time_ms: float = 0.0
    quality_score: float = 1.0
    num_tokens_pruned: int = 0

    def summary(self) -> Dict[str, object]:
        """Return a dict for display / serialisation."""
        return {
            "num_tokens": self.num_tokens,
            "quantisation": self.quantisation,
            "pruner": self.pruner,
            "eviction": self.eviction_policy,
            "original_mb": round(self.original_memory_mb, 2),
            "compressed_mb": round(self.compressed_memory_mb, 2),
            "compression_ratio": round(self.compression_ratio, 2),
            "memory_saved_mb": round(self.memory_saved_mb, 2),
            "quantize_ms": round(self.quantize_time_ms, 3),
            "dequantize_ms": round(self.dequantize_time_ms, 3),
            "total_ms": round(self.total_cache_time_ms, 3),
            "quality": round(self.quality_score, 4),
            "tokens_pruned": self.num_tokens_pruned,
        }


class CompressionMetrics:
    """Aggregates multiple compression records and produces comparison reports.

    Args:
        records: Optional initial list of records.
    """

    def __init__(self, records: Optional[List[CompressionRecord]] = None) -> None:
        self.records: List[CompressionRecord] = records or []

    def add(self, record: CompressionRecord) -> None:
        """Append a record."""
        self.records.append(record)

    def compare_quantisation(self, num_tokens: int) -> List[CompressionRecord]:
        """Return records for a specific token count, one per quantisation."""
        return [r for r in self.records if r.num_tokens == num_tokens]

    def best_compression(self, num_tokens: int) -> Optional[CompressionRecord]:
        """Return the record with highest compression ratio for *num_tokens*."""
        matches = self.compare_quantisation(num_tokens)
        if not matches:
            return None
        return max(matches, key=lambda r: r.compression_ratio)

    def table_rows(self) -> List[Dict[str, object]]:
        """Return sorted list of summary dicts for table generation."""
        return [r.summary() for r in self.records]

    def print_table(self, title: str = "Compression Benchmark Results") -> None:
        """Pretty-print a comparison table across all records."""
        if not self.records:
            print("No records.")
            return

        rows = self.table_rows()
        print(f"\n{'=' * 110}")
        print(f"  {title}")
        print(f"{'=' * 110}")
        header = (
            f"  {'Tokens':>7} {'Quant':>6} {'Pruner':>14} {'Evict':>10} "
            f"{'Orig MB':>8} {'Comp MB':>8} {'Ratio':>6} {'Saved MB':>8} "
            f"{'Q ms':>6} {'DQ ms':>6}"
        )
        print(header)
        print("  " + "-" * 105)
        for r in rows:
            print(
                f"  {r['num_tokens']:>7} {r['quantisation']:>6} "
                f"{r['pruner']:>14} {r['eviction']:>10} "
                f"{r['original_mb']:>8.2f} {r['compressed_mb']:>8.2f} "
                f"{r['compression_ratio']:>6.2f}x {r['memory_saved_mb']:>8.2f} "
                f"{r['quantize_ms']:>6.2f} {r['dequantize_ms']:>6.2f}"
            )
        print(f"{'=' * 110}")
