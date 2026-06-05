"""layer_hooks — wraps transformer decoder layers for selective execution.

Two mechanisms:
    1. :class:`LayerGate` — wraps each decoder layer so it can be
       dynamically enabled or disabled per forward pass.
    2. :class:`LayerMonitor` — hooks into each layer to record
       execution time, memory use, and enable/disable state.

Usage::

    bundle = load_model("tinyllama")
    gates = install_layer_gates(bundle.model, bundle.num_layers)
    monitor = LayerMonitor(bundle.num_layers)
    set_active_layers(gates, [0, 1, 5, 10, 15, 21])
    # ... run inference ...
    stats = monitor.summary()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from model_loader import ModelBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LayerGate
# ---------------------------------------------------------------------------


class LayerGate(nn.Module):
    """Wraps a single transformer decoder layer, allowing it to be skipped.

    When ``active`` is ``True``, the underlying layer executes normally.
    When ``False``, the gate returns a passthrough — the hidden states
    are returned unchanged and the KV cache is not updated.

    Args:
        layer: The original decoder layer (e.g. ``LlamaDecoderLayer``).
        layer_id: Zero-based index of this layer in the model.
    """

    def __init__(self, layer: nn.Module, layer_id: int) -> None:
        super().__init__()
        self.layer = layer
        self.layer_id = layer_id
        self._active: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self._active = value

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Run the layer or return a passthrough tensor.

        In transformers v5.x each decoder layer returns ``hidden_states``
        directly (a single tensor).  When the gate is inactive we return
        the input tensor unchanged.
        """
        if self._active:
            return self.layer(*args, **kwargs)
        return self._passthrough(*args, **kwargs)

    def _passthrough(self, *args, **kwargs) -> torch.Tensor:
        """Return hidden_states unchanged (skip computation)."""
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError("LayerGate passthrough requires hidden_states")
        return hidden_states


# ---------------------------------------------------------------------------
# LayerMonitor
# ---------------------------------------------------------------------------


@dataclass
class LayerStats:
    """Per-layer runtime statistics accumulated during inference."""

    calls: int = 0
    total_time_s: float = 0.0
    skipped_calls: int = 0
    peak_memory_mb: float = 0.0


class LayerMonitor:
    """Collects execution statistics from every layer gate.

    Args:
        num_layers: Total decoder layers in the model.
    """

    def __init__(self, num_layers: int) -> None:
        self.stats: List[LayerStats] = [LayerStats() for _ in range(num_layers)]
        self._timers: Dict[int, float] = {}

    def record_start(self, layer_id: int, skipped: bool) -> None:
        """Called before a layer executes (or is skipped)."""
        if skipped:
            self.stats[layer_id].skipped_calls += 1
        self.stats[layer_id].calls += 1
        self._timers[layer_id] = time.perf_counter()

    def record_end(self, layer_id: int) -> None:
        """Called after a layer completes (or is passed through)."""
        if layer_id in self._timers:
            elapsed = time.perf_counter() - self._timers[layer_id]
            self.stats[layer_id].total_time_s += elapsed
            del self._timers[layer_id]

        # Track CUDA peak memory if available
        if torch.cuda.is_available():
            cur = torch.cuda.max_memory_allocated() / (1024 ** 2)
            if cur > self.stats[layer_id].peak_memory_mb:
                self.stats[layer_id].peak_memory_mb = cur

    def summary(self) -> Dict[str, object]:
        """Return aggregate statistics across all layers."""
        total_calls = sum(s.calls for s in self.stats)
        total_skipped = sum(s.skipped_calls for s in self.stats)
        total_time = sum(s.total_time_s for s in self.stats)
        active_calls = total_calls - total_skipped

        return {
            "total_layer_calls": total_calls,
            "active_calls": active_calls,
            "skipped_calls": total_skipped,
            "total_time_s": round(total_time, 4),
            "avg_time_per_active_call_s": (
                round(total_time / max(active_calls, 1), 6)
            ),
            "per_layer": [
                {
                    "layer_id": i,
                    "calls": s.calls,
                    "skipped": s.skipped_calls,
                    "total_time_s": round(s.total_time_s, 4),
                    "peak_memory_mb": round(s.peak_memory_mb, 1),
                }
                for i, s in enumerate(self.stats)
            ],
        }

    def print_summary(self) -> None:
        """Print a human-readable summary."""
        s = self.summary()
        print("\n  LayerMonitor Summary")
        print(f"  Total layer calls:     {s['total_layer_calls']}")
        print(f"  Active calls:          {s['active_calls']}")
        print(f"  Skipped calls:         {s['skipped_calls']}")
        print(f"  Total time (all):      {s['total_time_s']:.4f}s")
        print(f"  Avg / active call:     {s['avg_time_per_active_call_s']:.6f}s")


# ---------------------------------------------------------------------------
# Gate installation
# ---------------------------------------------------------------------------


def _locate_decoder_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    """Walk common model paths to find the decoder layer ModuleList.

    Supports: LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM, GPT2LMHeadModel.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        return model.decoder.layers
    # Fallback: search for first nn.ModuleList with LayerNorm neighbours
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 1:
            candidate = module[0]
            if hasattr(candidate, "self_attn") or hasattr(candidate, "attention"):
                logger.info("Found decoder layers via heuristic at %s", name)
                return module
    return None


def install_layer_gates(
    model: nn.Module,
    num_layers: int,
    monitor: Optional[LayerMonitor] = None,
) -> List[LayerGate]:
    """Replace all decoder layers with :class:`LayerGate` wrappers.

    The original layers are preserved inside each gate.  Use
    :func:`set_active_layers` after installation to control which
    layers execute.

    Args:
        model: The loaded HuggingFace model.
        num_layers: Number of decoder layers.
        monitor: Optional :class:`LayerMonitor` for statistics.

    Returns:
        A list of all installed :class:`LayerGate` instances (indexed
        by layer id).

    Raises:
        RuntimeError: If the decoder layers cannot be located.
    """
    layers_module = _locate_decoder_layers(model)
    if layers_module is None:
        raise RuntimeError(
            "Could not locate decoder layer ModuleList in the model. "
            f"Model type: {type(model).__name__}"
        )

    gates: List[LayerGate] = []
    for i in range(num_layers):
        original = layers_module[i]
        gate = LayerGate(original, i)
        layers_module[i] = gate
        gates.append(gate)

    logger.info("Installed %d LayerGate wrappers", len(gates))
    return gates


def set_active_layers(
    gates: List[LayerGate],
    active_ids: Set[int],
) -> None:
    """Enable or disable each gate according to a set of active layer ids.

    Args:
        gates: List of all :class:`LayerGate` instances.
        active_ids: Set of layer indices that should execute.
    """
    for i, gate in enumerate(gates):
        gate.active = i in active_ids
    active = sum(1 for g in gates if g.active)
    logger.debug("Active layers: %d / %d", active, len(gates))


def uninstall_layer_gates(
    model: nn.Module,
    gates: List[LayerGate],
) -> None:
    """Restore original layers by removing the gate wrappers.

    Args:
        model: The model whose layers were gated.
        gates: The :class:`LayerGate` list returned by
            :func:`install_layer_gates`.
    """
    layers_module = _locate_decoder_layers(model)
    if layers_module is None:
        logger.warning("Could not locate layers module for uninstall")
        return
    for i, gate in enumerate(gates):
        if i < len(layers_module):
            layers_module[i] = gate.layer
    logger.info("Uninstalled %d LayerGate wrappers", len(gates))
