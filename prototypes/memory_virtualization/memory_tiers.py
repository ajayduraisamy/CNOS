"""memory_tiers  defines the memory hierarchy for virtualised LLM execution.

Tier hierarchy:
    Tier 0  GPU VRAM  (fastest, smallest)
    Tier 1  CPU RAM   (fast, medium)
    Tier 2  Compressed KV Cache (software tier, in-RAM with quantised KV)
    Tier 3  SSD       (slow, largest)

Each tier has a fixed capacity, latency profile, and bandwidth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------


@dataclass
class TierConfig:
    """Fixed configuration for one memory tier.

    Attributes:
        name: Human-readable name (e.g. ``"RAM"``, ``"SSD"``).
        capacity: Total capacity in bytes.
        latency_ns: Access latency in nanoseconds.
        bandwidth_gbps: Read bandwidth in GB/s.
        page_latency_ns: Additional per-page overhead (ns).
        max_transfer_size: Maximum transfer size per I/O operation.
        is_volatile: Whether data is lost on power loss.
        tier_id: Integer identifier (03).
    """
    name: str = ""
    capacity: int = 0
    latency_ns: int = 0
    bandwidth_gbps: float = 0.0
    page_latency_ns: int = 0
    max_transfer_size: int = 0
    is_volatile: bool = True
    tier_id: int = 0

    @property
    def capacity_mb(self) -> float:
        return self.capacity / (1024 ** 2)

    @property
    def capacity_gb(self) -> float:
        return self.capacity / (1024 ** 3)


# ---------------------------------------------------------------------------
# Default tier configurations
# ---------------------------------------------------------------------------

DEFAULT_TIERS: Dict[int, TierConfig] = {
    0: TierConfig(
        name="GPU VRAM",
        capacity=24 * 1024 ** 3,     # 24 GB
        latency_ns=200,
        bandwidth_gbps=900.0,
        page_latency_ns=100,
        max_transfer_size=256 * 1024 ** 2,  # 256 MB
        is_volatile=True,
        tier_id=0,
    ),
    1: TierConfig(
        name="CPU RAM",
        capacity=32 * 1024 ** 3,     # 32 GB
        latency_ns=80,
        bandwidth_gbps=50.0,
        page_latency_ns=200,
        max_transfer_size=64 * 1024 ** 2,   # 64 MB
        is_volatile=True,
        tier_id=1,
    ),
    2: TierConfig(
        name="Compressed KV",
        capacity=4 * 1024 ** 3,      # 4 GB (INT4 compressed KV)
        latency_ns=500,
        bandwidth_gbps=25.0,
        page_latency_ns=500,
        max_transfer_size=16 * 1024 ** 2,   # 16 MB
        is_volatile=True,
        tier_id=2,
    ),
    3: TierConfig(
        name="SSD",
        capacity=500 * 1024 ** 3,    # 500 GB
        latency_ns=100000,           # 100 us
        bandwidth_gbps=3.5,
        page_latency_ns=50000,
        max_transfer_size=1024 * 1024 ** 2,  # 1 GB
        is_volatile=False,
        tier_id=3,
    ),
}


# ---------------------------------------------------------------------------
# Runtime tier state
# ---------------------------------------------------------------------------


class MemoryTier:
    """Runtime representation of one memory tier with usage tracking.

    Args:
        config: The fixed :class:`TierConfig`.
    """

    def __init__(self, config: TierConfig) -> None:
        self.config = config
        self.used: int = 0

    @property
    def free(self) -> int:
        return self.config.capacity - self.used

    @property
    def free_mb(self) -> float:
        return self.free / (1024 ** 2)

    @property
    def free_gb(self) -> float:
        return self.free / (1024 ** 3)

    @property
    def utilisation_pct(self) -> float:
        if self.config.capacity == 0:
            return 0.0
        return (self.used / self.config.capacity) * 100.0

    def can_allocate(self, size: int) -> bool:
        return self.free >= size

    def allocate(self, size: int) -> None:
        if not self.can_allocate(size):
            raise MemoryError(
                f"Not enough space in {self.config.name}: "
                f"need {size / (1024**2):.1f} MB, have {self.free_mb:.1f} MB free"
            )
        self.used += size

    def free_space(self, size: int) -> None:
        self.used = max(0, self.used - size)

    def reset(self) -> None:
        self.used = 0

    @property
    def access_cost_ns(self) -> int:
        """Total access latency in ns for one page read."""
        return self.config.latency_ns + self.config.page_latency_ns

    def transfer_time_ns(self, size: int) -> float:
        """Time in ns to transfer *size* bytes from this tier."""
        bw_bytes_per_ns = self.config.bandwidth_gbps * 1e9 / 8 / 1e9  # GB/s  bytes/ns
        if bw_bytes_per_ns <= 0:
            return float("inf")
        return size / bw_bytes_per_ns


# ---------------------------------------------------------------------------
# Tier manager
# ---------------------------------------------------------------------------


class TierManager:
    """Manages all memory tiers in the hierarchy.

    Args:
        configs: Dict mapping ``tier_id  TierConfig``.
            Defaults to :data:`DEFAULT_TIERS`.
    """

    def __init__(
        self,
        configs: Optional[Dict[int, TierConfig]] = None,
    ) -> None:
        configs = configs or DEFAULT_TIERS
        self.tiers: Dict[int, MemoryTier] = {
            tid: MemoryTier(cfg) for tid, cfg in configs.items()
        }
        logger.info(
            "TierManager: %d tiers (%s)",
            len(self.tiers),
            ", ".join(t.config.name for t in self.tiers.values()),
        )

    def __getitem__(self, tier_id: int) -> MemoryTier:
        return self.tiers[tier_id]

    @property
    def total_capacity_bytes(self) -> int:
        return sum(t.config.capacity for t in self.tiers.values())

    @property
    def total_used_bytes(self) -> int:
        return sum(t.used for t in self.tiers.values())

    def reset(self) -> None:
        for tier in self.tiers.values():
            tier.reset()

    def summary(self) -> Dict[str, object]:
        rows: Dict[str, object] = {}
        for tid, tier in self.tiers.items():
            rows[f"tier_{tid}_name"] = tier.config.name
            rows[f"tier_{tid}_capacity_gb"] = round(tier.config.capacity_gb, 1)
            rows[f"tier_{tid}_used_gb"] = round(tier.used / (1024 ** 3), 3)
            rows[f"tier_{tid}_free_gb"] = round(tier.free_gb, 3)
            rows[f"tier_{tid}_util_pct"] = round(tier.utilisation_pct, 1)
        return rows

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n{'=' * 60}")
        print(f"  Memory Tier Summary")
        print(f"{'=' * 60}")
        print(f"  {'Tier':<20} {'Capacity':<12} {'Used':<12} {'Free':<12} {'Util':<8}")
        print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
        for tid in sorted(self.tiers.keys()):
            print(
                f"  {s[f'tier_{tid}_name']:<20} "
                f"{s[f'tier_{tid}_capacity_gb']:<8.1f}GB "
                f"{s[f'tier_{tid}_used_gb']:<8.3f}GB "
                f"{s[f'tier_{tid}_free_gb']:<8.3f}GB "
                f"{s[f'tier_{tid}_util_pct']:<6.1f}%"
            )
        print(f"{'=' * 60}")
