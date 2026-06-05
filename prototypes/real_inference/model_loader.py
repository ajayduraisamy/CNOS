"""model_loader — loads a HuggingFace transformer model with memory-aware configuration.

Supports TinyLlama, Qwen 2.5 1.5B, and Llama 3.2 1B.  Detects
available hardware (CUDA, MPS, CPU) and selects an appropriate dtype
to stay within memory constraints.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, str] = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
}

# Model metadata for informed defaults
MODEL_META: dict[str, dict] = {
    "tinyllama": {"num_layers": 22, "hidden_size": 2048, "num_heads": 32},
    "qwen-1.5b": {"num_layers": 28, "hidden_size": 2048, "num_heads": 32},
    "llama-3.2-1b": {"num_layers": 16, "hidden_size": 2048, "num_heads": 32},
}


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------


def pick_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def pick_dtype(device: torch.device, ram_gb: Optional[float] = None) -> torch.dtype:
    """Pick the best dtype based on device and available RAM.

    Args:
        device: Target device.
        ram_gb: Total system RAM in GB (auto-detected if None).

    Returns:
        ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``.
    """
    if device.type == "cuda":
        return torch.float16
    if ram_gb is None:
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            ram_gb = 8.0  # conservative default

    # bfloat16 is supported on most modern CPUs, but fall back to float32
    # if RAM is plentiful.
    if ram_gb >= 16.0:
        return torch.float32
    if hasattr(torch, "bfloat16") and torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """Container for the loaded model, tokenizer, and metadata.

    Attributes:
        model: The HuggingFace model (eval mode).
        tokenizer: The matching tokenizer.
        device: The torch device the model lives on.
        dtype: The dtype used for model weights.
        num_layers: Number of transformer decoder layers.
        model_name: Short name (e.g. ``"tinyllama"``).
    """

    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizerFast
    device: torch.device
    dtype: torch.dtype
    num_layers: int
    model_name: str = "tinyllama"


def load_model(
    model_key: str = "tinyllama",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    use_cache: bool = True,
    max_memory: Optional[dict] = None,
) -> ModelBundle:
    """Load a model and tokenizer from the HuggingFace hub.

    Args:
        model_key: Short key from ``MODEL_REGISTRY``.
        device: Override auto-detected device.
        dtype: Override auto-detected dtype.
        use_cache: Enable KV cache.
        max_memory: Per-device memory limit dict (e.g. ``{"cuda:0": "4GiB"}``).

    Returns:
        A :class:`ModelBundle` with model ready for inference.

    Raises:
        ValueError: If *model_key* is not in the registry.
        ImportError: If the model cannot be downloaded or loaded.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_key}'. Choose from {list(MODEL_REGISTRY.keys())}"
        )

    model_id = MODEL_REGISTRY[model_key]
    meta = MODEL_META[model_key]
    device = device or pick_device()
    dtype = dtype or pick_dtype(device)

    logger.info(
        "Loading %s (%s) on %s with %s",
        model_key, model_id, device, dtype,
    )

    # Build kwargs for from_pretrained
    pretrain_kwargs: dict = {
        "dtype": dtype,
        "use_cache": use_cache,
        "low_cpu_mem_usage": True,
    }
    # Only use device_map on CUDA; on CPU/MPS we load then .to(device)
    if device.type == "cuda":
        pretrain_kwargs["device_map"] = "auto"
    if max_memory is not None:
        pretrain_kwargs["max_memory"] = max_memory

    # Wrap in try/except for network / OOM failures
    try:
        logger.info("Downloading tokenizer from %s ...", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        logger.info("Downloading model %s (this may take a while)...", model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **pretrain_kwargs)
    except Exception as exc:
        raise ImportError(f"Failed to load model {model_id}: {exc}") from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    if device.type != "cuda":
        model.to(device)

    # Determine actual number of decoder layers
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = meta["num_layers"]

    logger.info(
        "Model loaded: %s  |  %d layers  |  %.0fM params  |  device=%s  dtype=%s",
        model_key,
        num_layers,
        model.num_parameters() / 1e6,
        device,
        dtype,
    )

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        num_layers=num_layers,
        model_name=model_key,
    )
