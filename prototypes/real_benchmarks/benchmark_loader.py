"""benchmark_loader -- loads real HuggingFace models with timing and memory measurement.

Wraps the existing ``real_inference.model_loader`` and adds:
  * Load-time measurement
  * Post-load memory profiling
  * Graceful download failure handling
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_PROTO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "real_inference"))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

from model_loader import load_model as _real_load  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_KEYS: Dict[str, str] = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}

MODEL_LAYERS: Dict[str, int] = {
    "tinyllama": 22,
    "qwen-1.5b": 28,
}


@dataclass
class LoadResult:
    """Result of loading a model.

    Attributes:
        success: Whether the model loaded successfully.
        model_key: The requested model key.
        model_id: The HuggingFace model ID.
        bundle: The ``ModelBundle`` (None if load failed).
        load_time_s: Time taken to download and load.
        memory_mb: Process RSS after loading.
        error: Error message if load failed.
    """
    success: bool = False
    model_key: str = ""
    model_id: str = ""
    bundle: Any = None
    load_time_s: float = 0.0
    memory_mb: float = 0.0
    error: str = ""

    def summary(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_id": self.model_id,
            "success": self.success,
            "load_time_s": round(self.load_time_s, 2),
            "memory_mb": round(self.memory_mb, 1),
            "num_layers": self.bundle.num_layers if self.bundle else 0,
            "device": str(self.bundle.device) if self.bundle else "",
            "dtype": str(self.bundle.dtype) if self.bundle else "",
        }


def _get_process_memory_mb() -> float:
    """Return current process RSS in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def load_model(
    model_key: str = "tinyllama",
    device: Any = None,
    dtype: Any = None,
    max_memory: Optional[Dict[str, str]] = None,
    timeout_s: float = 300.0,
) -> LoadResult:
    """Load a model and tokenizer with timing and memory measurement.

    Args:
        model_key: ``"tinyllama"`` or ``"qwen-1.5b"``.
        device: Override device (auto-detected if None).
        dtype: Override dtype (auto-detected if None).
        max_memory: Per-device memory limit dict.
        timeout_s: Maximum seconds to wait for download.

    Returns:
        A :class:`LoadResult` with the model bundle or error details.
    """
    if model_key not in MODEL_KEYS:
        return LoadResult(
            success=False,
            model_key=model_key,
            model_id="",
            error=f"Unknown model_key={model_key!r}. "
                  f"Options: {list(MODEL_KEYS.keys())}",
        )

    model_id = MODEL_KEYS[model_key]
    mem_before = _get_process_memory_mb()

    # Patience: try-load with timeout by overriding socket timeout
    import socket
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_s)

    try:
        # Convert string device/dtype to torch types for real_inference
        import torch
        _dev = torch.device(device) if isinstance(device, str) else device
        _dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        start = time.perf_counter()
        bundle = _real_load(
            model_key=model_key,
            device=_dev,
            dtype=_dtype,
            max_memory=max_memory,
        )
        elapsed = time.perf_counter() - start

        mem_after = _get_process_memory_mb()
        load_mem = max(0.0, mem_after - mem_before)

        return LoadResult(
            success=True,
            model_key=model_key,
            model_id=model_id,
            bundle=bundle,
            load_time_s=elapsed,
            memory_mb=load_mem,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - time.perf_counter()
        return LoadResult(
            success=False,
            model_key=model_key,
            model_id=model_id,
            load_time_s=elapsed,
            memory_mb=_get_process_memory_mb() - mem_before,
            error=str(exc),
        )

    finally:
        socket.setdefaulttimeout(old_timeout)
