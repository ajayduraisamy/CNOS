"""metrics - tracks virtual memory performance counters and generates reports.

Monitors:
    * RAM and SSD usage
    * Page faults and cache hit rate
    * Evictions per tier
    * Prefetch accuracy
    * Access latencies
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VirtualMemoryReport:
    """Single benchmark report for one model/RAM configuration.

    Attributes:
        model_name: Model identifier (e.g. ``"7B"``).
        ram_gb: Available RAM in GB.
        model_params_gb: Total parameter size in GB.
        kv_cache_gb: Total KV cache size in GB.
        total_footprint_gb: Total memory footprint in GB.
        ram_used_gb: Peak RAM usage in GB.
        ssd_used_gb: Peak SSD usage in GB.
        ram_util_pct: RAM utilisation percentage.
        page_faults: Total page faults.
        page_hits: Total page hits.
        hit_rate_pct: Cache hit rate percentage.
        evictions: Total evictions.
        prefetches: Total prefetches issued.
        prefetch_accuracy_pct: Percentage of prefetches actually used.
        total_accesses: Total simulated memory accesses.
        avg_latency_ns: Average access latency in ns.
    """
    model_name: str = ""
    ram_gb: float = 0.0
    model_params_gb: float = 0.0
    kv_cache_gb: float = 0.0
    total_footprint_gb: float = 0.0
    ram_used_gb: float = 0.0
    ssd_used_gb: float = 0.0
    ram_util_pct: float = 0.0
    page_faults: int = 0
    page_hits: int = 0
    hit_rate_pct: float = 0.0
    evictions: int = 0
    prefetches: int = 0
    prefetch_accuracy_pct: float = 0.0
    total_accesses: int = 0
    avg_latency_ns: float = 0.0

    def summary(self) -> Dict[str, object]:
        return {
            "model": self.model_name,
            "ram_gb": self.ram_gb,
            "params_gb": self.model_params_gb,
            "kv_cache_gb": self.kv_cache_gb,
            "footprint_gb": self.total_footprint_gb,
            "ram_used_gb": round(self.ram_used_gb, 2),
            "ssd_used_gb": round(self.ssd_used_gb, 2),
            "ram_util_pct": round(self.ram_util_pct, 1),
            "page_faults": self.page_faults,
            "page_hits": self.page_hits,
            "hit_rate_pct": round(self.hit_rate_pct, 1),
            "evictions": self.evictions,
            "prefetches": self.prefetches,
            "prefetch_accuracy_pct": round(self.prefetch_accuracy_pct, 1),
            "total_accesses": self.total_accesses,
            "avg_latency_ns": round(self.avg_latency_ns, 1),
        }


class VirtualMemoryMetrics:
    """Aggregates counters and produces reports."""

    def __init__(self) -> None:
        self.page_faults: int = 0
        self.page_hits: int = 0
        self.evictions: int = 0
        self.total_accesses: int = 0
        self.total_latency_ns: int = 0
        self.reports: List[VirtualMemoryReport] = []

    def record_fault(self) -> None:
        self.page_faults += 1
        self.total_accesses += 1

    def record_hit(self) -> None:
        self.page_hits += 1
        self.total_accesses += 1

    def record_eviction(self) -> None:
        self.evictions += 1

    def record_latency(self, latency_ns: int) -> None:
        self.total_latency_ns += latency_ns

    @property
    def hit_rate(self) -> float:
        total = self.page_hits + self.page_faults
        if total == 0:
            return 1.0
        return self.page_hits / total

    @property
    def avg_latency_ns(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.total_latency_ns / self.total_accesses

    def produce_report(
        self,
        model_name: str = "7B",
        ram_gb: float = 4.0,
        model_params_gb: float = 14.0,
        kv_cache_gb: float = 2.0,
        ram_used_gb: float = 0.0,
        ssd_used_gb: float = 0.0,
        total_prefetches: int = 0,
        prefetch_accuracy: float = 1.0,
    ) -> VirtualMemoryReport:
        total_footprint = model_params_gb + kv_cache_gb
        ram_util = (ram_used_gb / max(ram_gb, 0.001)) * 100.0
        hit_pct = self.hit_rate * 100.0

        report = VirtualMemoryReport(
            model_name=model_name,
            ram_gb=ram_gb,
            model_params_gb=model_params_gb,
            kv_cache_gb=kv_cache_gb,
            total_footprint_gb=total_footprint,
            ram_used_gb=ram_used_gb,
            ssd_used_gb=ssd_used_gb,
            ram_util_pct=ram_util,
            page_faults=self.page_faults,
            page_hits=self.page_hits,
            hit_rate_pct=hit_pct,
            evictions=self.evictions,
            prefetches=total_prefetches,
            prefetch_accuracy_pct=prefetch_accuracy * 100.0,
            total_accesses=self.total_accesses,
            avg_latency_ns=self.avg_latency_ns,
        )
        self.reports.append(report)
        return report

    def print_report(self, report: VirtualMemoryReport) -> None:
        """Print a single report."""
        s = report.summary()
        print(f"\n{'=' * 60}")
        print(f"  Virtual Memory Report - {s['model']} @ {s['ram_gb']}GB RAM")
        print(f"{'=' * 60}")
        print(f"  Memory footprint:   {s['footprint_gb']:.1f} GB total")
        print(f"    Parameters:       {s['params_gb']:.1f} GB")
        print(f"    KV cache:         {s['kv_cache_gb']:.1f} GB")
        print(f"  Physical usage:")
        print(f"    RAM:              {s['ram_used_gb']:.2f} GB ({s['ram_util_pct']:.1f}%)")
        print(f"    SSD:              {s['ssd_used_gb']:.2f} GB")
        print(f"  Performance:")
        print(f"    Total accesses:   {s['total_accesses']}")
        print(f"    Page hits:        {s['page_hits']}")
        print(f"    Page faults:      {s['page_faults']}")
        print(f"    Hit rate:         {s['hit_rate_pct']:.1f}%")
        print(f"    Evictions:        {s['evictions']}")
        print(f"    Prefetches:       {s['prefetches']}")
        print(f"    Prefetch acc:     {s['prefetch_accuracy_pct']:.1f}%")
        print(f"    Avg latency:      {s['avg_latency_ns']:.0f} ns")
        print(f"{'=' * 60}")

    def print_comparison(self) -> None:
        """Print a comparison table of all reports."""
        if not self.reports:
            print("No reports.")
            return
        print(f"\n{'=' * 130}")
        print(f"  Memory Virtualisation - Cross-Config Comparison")
        print(f"{'=' * 130}")
        header = (
            f"  {'Model':>6} {'RAM':>5} {'Footpr.':>7} "
            f"{'RAM used':>8} {'SSD used':>8} {'RAM%':>5} "
            f"{'Faults':>7} {'Hits':>7} {'Hit%':>5} "
            f"{'Evict':>6} {'Pref.':>6} {'Acc%':>5} "
            f"{'Lat(ns)':>8}"
        )
        print(header)
        print("  " + "-" * 125)
        for r in self.reports:
            s = r.summary()
            print(
                f"  {s['model']:>6} {s['ram_gb']:>4.0f}GB "
                f"{s['footprint_gb']:>6.1f}GB "
                f"{s['ram_used_gb']:>7.2f}GB "
                f"{s['ssd_used_gb']:>7.2f}GB "
                f"{s['ram_util_pct']:>4.1f}% "
                f"{s['page_faults']:>7} {s['page_hits']:>7} "
                f"{s['hit_rate_pct']:>4.1f}% "
                f"{s['evictions']:>6} {s['prefetches']:>6} "
                f"{s['prefetch_accuracy_pct']:>4.1f}% "
                f"{s['avg_latency_ns']:>7.0f}"
            )
        print(f"{'=' * 130}")

    def reset(self) -> None:
        self.page_faults = 0
        self.page_hits = 0
        self.evictions = 0
        self.total_accesses = 0
        self.total_latency_ns = 0
