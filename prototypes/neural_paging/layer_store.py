"""LayerStore — simulated on-disk storage of transformer layers.

Each layer is a named artifact with a size (MB), compute cost, and type
tag.  ``load_layer`` simulates the I/O cost of reading from disk by
blocking for a duration proportional to layer size.  ``unload_layer``
releases the simulated memory.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Lightweight sentinel to indicate a layer is loaded without allocating real bytes
_LOADED_SENTINEL = object()

# ---------------------------------------------------------------------------
# Layer metadata
# ---------------------------------------------------------------------------


@dataclass
class LayerMeta:
    """Metadata for a single transformer layer stored on disk.

    Attributes:
        layer_id: Zero-based index in the model (0 … num_layers-1).
        name: Human-readable label (e.g. "encoder.layer.23").
        size_mb: Simulated size of the layer on disk / in RAM.
        compute_cost: Abstract compute units needed to execute the layer.
        layer_type: Categorical tag — "attention", "ff", "norm", "embed".
        is_loaded: Whether the layer is currently resident in memory.
    """

    layer_id: int
    name: str
    size_mb: float
    compute_cost: float = 1.0
    layer_type: str = "ff"
    is_loaded: bool = False


# ---------------------------------------------------------------------------
# LayerStore
# ---------------------------------------------------------------------------


class LayerStore:
    """Simulates a disk-backed repository of transformer layers.

    All 80 layers are created at initialisation and can be loaded or
    unloaded individually.  Loading incurs a simulated I/O delay.

    Args:
        num_layers: Total number of layers in the model (default 80).
        seed: Random seed for reproducible layer sizes / types.
    """

    # Proportion of layers by type (must sum to 1.0)
    TYPE_DISTRIBUTION: tuple[tuple[str, float, float], ...] = (
        ("attention", 0.35, 2.0),
        ("ff", 0.45, 1.5),
        ("norm", 0.15, 0.3),
        ("embed", 0.05, 0.8),
    )

    # Size range per layer type: (min_mb, max_mb)
    SIZE_RANGES: dict[str, tuple[float, float]] = {
        "attention": (80.0, 200.0),
        "ff": (120.0, 280.0),
        "norm": (5.0, 20.0),
        "embed": (30.0, 90.0),
    }

    def __init__(self, num_layers: int = 80, seed: int = 42) -> None:
        self.num_layers = num_layers
        self._rng = random.Random(seed)
        self._layers: Dict[int, LayerMeta] = {}
        self._loaded_data: Dict[int, bytes] = {}
        self._load_count: int = 0
        self._unload_count: int = 0
        self._total_load_bytes: int = 0

        self._build_layers()
        logger.info(
            "LayerStore initialised with %d layers (total size: %.0f MB)",
            num_layers,
            sum(l.size_mb for l in self._layers.values()),
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _assign_type(self) -> tuple[str, float]:
        """Pick a layer type + compute cost according to the distribution."""
        r = self._rng.random()
        cumulative = 0.0
        for t, prob, cost in self.TYPE_DISTRIBUTION:
            cumulative += prob
            if r <= cumulative:
                return t, cost
        return self.TYPE_DISTRIBUTION[-1][0], self.TYPE_DISTRIBUTION[-1][2]

    def _build_layers(self) -> None:
        """Create all layer metadata entries."""
        for i in range(self.num_layers):
            layer_type, cost = self._assign_type()
            min_sz, max_sz = self.SIZE_RANGES[layer_type]
            size = round(self._rng.uniform(min_sz, max_sz), 1)
            meta = LayerMeta(
                layer_id=i,
                name=f"encoder.layer.{i}",
                size_mb=size,
                compute_cost=cost,
                layer_type=layer_type,
                is_loaded=False,
            )
            self._layers[i] = meta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_layer(self, layer_id: int) -> LayerMeta:
        """Load a layer from disk into memory.

        Simulates I/O latency proportional to the layer's size.
        Returns the :class:`LayerMeta` of the loaded layer.

        Raises:
            KeyError: If *layer_id* does not exist.
        """
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found in store")

        meta = self._layers[layer_id]
        if meta.is_loaded:
            logger.debug("Layer %d already loaded (idempotent)", layer_id)
            return meta

        # Simulate disk I/O: ~10 MB/ms read speed
        io_delay = meta.size_mb / 10.0 * 0.001
        time.sleep(io_delay)

        # Mark as loaded with a lightweight sentinel (no real allocation)
        self._loaded_data[layer_id] = _LOADED_SENTINEL
        meta.is_loaded = True
        self._load_count += 1
        self._total_load_bytes += int(meta.size_mb * 1024 * 1024)

        logger.debug("Loaded layer %d (%.1f MB, %.2f ms I/O)", layer_id, meta.size_mb, io_delay * 1000)
        return meta

    def unload_layer(self, layer_id: int) -> None:
        """Unload a layer from memory, freeing its simulated RAM.

        Raises:
            KeyError: If *layer_id* does not exist.
        """
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found in store")

        meta = self._layers[layer_id]
        if not meta.is_loaded:
            return

        self._loaded_data.pop(layer_id, None)
        meta.is_loaded = False
        self._unload_count += 1
        logger.debug("Unloaded layer %d", layer_id)

    def get_meta(self, layer_id: int) -> LayerMeta:
        """Return the metadata for a layer without loading it."""
        if layer_id not in self._layers:
            raise KeyError(f"Layer {layer_id} not found")
        return self._layers[layer_id]

    def is_loaded(self, layer_id: int) -> bool:
        """Check whether a layer is currently resident in memory."""
        return self._layers[layer_id].is_loaded

    @property
    def loaded_size_mb(self) -> float:
        """Total size (MB) of all currently loaded layers."""
        return sum(
            m.size_mb for m in self._layers.values() if m.is_loaded
        )

    @property
    def total_disk_size_mb(self) -> float:
        """Total size of all layers on disk."""
        return sum(m.size_mb for m in self._layers.values())

    @property
    def load_count(self) -> int:
        return self._load_count

    @property
    def unload_count(self) -> int:
        return self._unload_count
