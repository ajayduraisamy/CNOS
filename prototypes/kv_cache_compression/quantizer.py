"""quantizer — supports FP16, INT8, and INT4 quantisation for KV cache tensors.

Each quantiser provides ``quantize(tensor) → (encoded, metadata)`` and
``dequantize(encoded, metadata) → tensor`` round-trip operations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class QuantMetadata:
    """Metadata needed to dequantize a tensor.

    Attributes:
        dtype: Original tensor dtype.
        scale: Scale factor (scalar, per-token vector, or per-channel).
        zero_point: Zero point for asymmetric quantisation.
        scheme: Quantisation scheme name.
        shape: Original tensor shape.
    """
    dtype: torch.dtype = torch.float16
    scale: Optional[torch.Tensor] = None
    zero_point: Optional[torch.Tensor] = None
    scheme: str = "fp16"
    shape: Optional[Tuple[int, ...]] = None


# ---------------------------------------------------------------------------
# Base quantiser
# ---------------------------------------------------------------------------


class BaseQuantizer(ABC):
    """Abstract base for all KV cache quantisers."""

    @abstractmethod
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, QuantMetadata]:
        """Quantize a tensor and return ``(encoded, metadata)``."""

    @abstractmethod
    def dequantize(self, encoded: torch.Tensor, metadata: QuantMetadata) -> torch.Tensor:
        """Restore a tensor from its quantised representation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name like ``"fp16"``, ``"int8"``, ``"int4"``."""

    @property
    @abstractmethod
    def bits_per_element(self) -> int:
        """Number of bits per tensor element."""


# ---------------------------------------------------------------------------
# FP16 quantiser (identity + dtype cast)
# ---------------------------------------------------------------------------


class FP16Quantizer(BaseQuantizer):
    """Cast FP32 tensors to FP16.  No actual quantisation loss for
    already-FP16 data; this exists for a uniform interface."""

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, QuantMetadata]:
        meta = QuantMetadata(
            dtype=tensor.dtype,
            scheme="fp16",
            shape=tuple(tensor.shape),
        )
        return tensor.to(torch.float16), meta

    def dequantize(self, encoded: torch.Tensor, metadata: QuantMetadata) -> torch.Tensor:
        return encoded.to(metadata.dtype)

    @property
    def name(self) -> str:
        return "fp16"

    @property
    def bits_per_element(self) -> int:
        return 16


# ---------------------------------------------------------------------------
# INT8 quantiser (per-tensor asymmetric)
# ---------------------------------------------------------------------------


class INT8Quantizer(BaseQuantizer):
    """Symmetric INT8 quantisation.

    Maps ``x`` to ``round(clip(x / scale, -128, 127))`` where
    ``scale = max(|x|) / 127``.  No zero-point needed.
    """

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, QuantMetadata]:
        orig_dtype = tensor.dtype
        tensor_fp32 = tensor.float()
        abs_max = tensor_fp32.abs().max()

        if abs_max < 1e-8:
            scale = torch.tensor(1.0, device=tensor.device)
            encoded = torch.zeros_like(tensor_fp32, dtype=torch.int8)
        else:
            scale = abs_max / 127.0
            encoded = (tensor_fp32 / scale).round().clamp(-128, 127).to(torch.int8)

        meta = QuantMetadata(
            dtype=orig_dtype,
            scale=scale,
            scheme="int8",
            shape=tuple(tensor.shape),
        )
        return encoded, meta

    def dequantize(self, encoded: torch.Tensor, metadata: QuantMetadata) -> torch.Tensor:
        scale = metadata.scale if metadata.scale is not None else torch.tensor(1.0)
        restored = encoded.float() * scale.float()
        return restored.to(metadata.dtype)

    @property
    def name(self) -> str:
        return "int8"

    @property
    def bits_per_element(self) -> int:
        return 8


# ---------------------------------------------------------------------------
# INT4 quantiser (symmetric, bit-packed)
# ---------------------------------------------------------------------------


class INT4Quantizer(BaseQuantizer):
    """Symmetric INT4 quantisation with bit-packing.

    Pairs of INT4 values are packed into a single byte::
        ``packed = (val0 & 0xF) | ((val1 & 0xF) << 4)``

    Note:
        INT4 range is [-8, 7] inclusive.
    """

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, QuantMetadata]:
        orig_dtype = tensor.dtype
        tensor_fp32 = tensor.float()
        abs_max = tensor_fp32.abs().max()

        if abs_max < 1e-8:
            scale = torch.tensor(1.0, device=tensor.device)
            flat = torch.zeros(tensor_fp32.numel(), dtype=torch.int8, device=tensor.device)
        else:
            scale = abs_max / 7.0
            # Scale to [-8, 7], clamp, round to int4
            scaled = (tensor_fp32 / scale).round().clamp(-8, 7).to(torch.int8)
            flat = scaled.flatten()

        # Ensure even number of elements for packing
        if flat.numel() % 2 != 0:
            flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8, device=tensor.device)])

        # Pack two int4 values per byte
        evens = (flat[0::2].to(torch.uint8) & 0x0F)
        odds = ((flat[1::2].to(torch.uint8) & 0x0F) << 4)
        packed = (evens | odds).to(torch.uint8)

        meta = QuantMetadata(
            dtype=orig_dtype,
            scale=scale,
            scheme="int4",
            shape=tuple(tensor.shape),
        )
        return packed, meta

    def dequantize(self, encoded: torch.Tensor, metadata: QuantMetadata) -> torch.Tensor:
        scale = metadata.scale if metadata.scale is not None else torch.tensor(1.0)
        num_elements = 1
        if metadata.shape:
            num_elements = 1
            for d in metadata.shape:
                num_elements *= d

        # Unpack: each byte → two int4 values (sign-extend from 4 bits)
        low = (encoded & 0x0F).to(torch.int8)
        high = ((encoded >> 4) & 0x0F).to(torch.int8)

        # Sign-extend 4-bit to 8-bit
        low = torch.where(low > 7, low - 16, low)
        high = torch.where(high > 7, high - 16, high)

        interleaved = torch.zeros(encoded.numel() * 2, dtype=torch.int8, device=encoded.device)
        interleaved[0::2] = low
        interleaved[1::2] = high

        restored = interleaved[:num_elements].float() * scale.float()
        if metadata.shape:
            restored = restored.reshape(metadata.shape)
        return restored.to(metadata.dtype)

    @property
    def name(self) -> str:
        return "int4"

    @property
    def bits_per_element(self) -> int:
        return 4


# ---------------------------------------------------------------------------
# Quantiser registry
# ---------------------------------------------------------------------------

QUANTIZER_REGISTRY: Dict[str, BaseQuantizer] = {
    "fp16": FP16Quantizer(),
    "int8": INT8Quantizer(),
    "int4": INT4Quantizer(),
}


def get_quantizer(name: str) -> BaseQuantizer:
    """Return a quantiser by name.  Raises ``KeyError`` on unknown name."""
    if name not in QUANTIZER_REGISTRY:
        raise KeyError(
            f"Unknown quantiser '{name}'.  Choose from {list(QUANTIZER_REGISTRY.keys())}"
        )
    return QUANTIZER_REGISTRY[name]
