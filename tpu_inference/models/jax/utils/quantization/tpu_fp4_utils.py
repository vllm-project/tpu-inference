"""Utilities for handling TPU FP4 packed weights and quantization metadata."""

from typing import Any, Dict, List

import torch
from vllm.model_executor.layers.quantization import (  # type: ignore
    QUANTIZATION_METHODS, register_quantization_config)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig  # type: ignore

from .fp4_utils import pack_fp4_from_fp32, unpack_fp4

# TPU FP4 constants
TPU_FP4_SUBCHANNEL_SIZE: int = 256
TPU_FP4_BYTES_PER_SUBCHANNEL: int = TPU_FP4_SUBCHANNEL_SIZE // 2
TPU_FP4_QUANT_METHOD: str = "tpu_fp4"


class TpuFp4Config(QuantizationConfig):
    """Minimal QuantizationConfig so vLLM recognizes TPU FP4 weights."""

    @classmethod
    def get_name(cls) -> str:
        return TPU_FP4_QUANT_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TpuFp4Config":
        inst = cls()
        inst.fmt = str(config.get("fmt", "")).lower()
        inst.weight_block_size = config.get("weight_block_size")
        return inst

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return None


def _validate_blocks(blocks_u8: torch.Tensor) -> None:
    if blocks_u8.dtype != torch.uint8:
        raise ValueError(
            f"Expected TPU FP4 blocks to be uint8, got {blocks_u8.dtype}")
    if blocks_u8.shape[-1] % 2 != 0:
        raise ValueError(
            "TPU FP4 blocks must have an even number of packed nibbles; "
            f"got last dimension {blocks_u8.shape[-1]}")


def _validate_scales(scales: torch.Tensor) -> None:
    if scales.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("TPU FP4 scales must be bf16 or fp32; got "
                         f"{scales.dtype}")


def dequant_tpu_fp4_to_bf16(blocks_u8: torch.Tensor,
                            scales: torch.Tensor) -> torch.Tensor:
    """Dequantize TPU FP4 packed weights into bfloat16 tensors.

    TPU checkpoints keep the logical axis flattened instead of reshaping to
    explicit 256-wide subchannels (unlike MXFP4's 32). This avoids padding
    overhead when the reducation dim is not divisible by 256. Since 32 is much
    smaller than 256, this situation occurs with more models than MXFP4.
    """
    _validate_blocks(blocks_u8)
    _validate_scales(scales)

    fp4_vals = unpack_fp4(blocks_u8)
    float_scales = scales.to(torch.float32)

    if fp4_vals.ndim != float_scales.ndim:
        raise ValueError(
            "TPU FP4 blocks and scales must have the same rank: blocks "
            f"{fp4_vals.shape} vs scales {float_scales.shape}")

    # Ensure scale prefix dims can broadcast to the decoded codes prefix dims.
    for idx, (code_dim, scale_dim) in enumerate(
            zip(fp4_vals.shape[:-1], float_scales.shape[:-1])):
        if scale_dim not in (1, code_dim):
            raise ValueError(
                "TPU FP4 scale prefix dim is not broadcastable to codes: "
                f"dim {idx}, blocks {code_dim}, scales {scale_dim}")

    # Expand scales to match the decoded prefix dims before repeating along the
    # implicit 256-wide subchannel axis that remains flattened in checkpoints.
    expanded_prefix = (*fp4_vals.shape[:-1], float_scales.shape[-1])
    float_scales = float_scales.expand(expanded_prefix)

    expanded_scales = torch.repeat_interleave(float_scales,
                                              TPU_FP4_SUBCHANNEL_SIZE,
                                              dim=-1)
    if expanded_scales.shape[-1] < fp4_vals.shape[-1]:
        raise ValueError("TPU FP4 scales do not cover the decoded values: "
                         f"{expanded_scales.shape[-1]} < {fp4_vals.shape[-1]}")

    expanded_scales = expanded_scales[..., :fp4_vals.shape[-1]]
    return (fp4_vals * expanded_scales).to(torch.bfloat16)


def unpack_tpu_fp4_to_fp32(
        blocks_u8: torch.Tensor,
        scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode TPU FP4 blocks and scales to float32 tensors suitable for QArray."""
    _validate_blocks(blocks_u8)
    _validate_scales(scales)

    fp4_vals = unpack_fp4(blocks_u8)
    float_scales = scales.to(torch.float32)

    if fp4_vals.ndim != float_scales.ndim:
        raise ValueError(
            "TPU FP4 blocks and scales must have the same rank: blocks "
            f"{fp4_vals.shape} vs scales {float_scales.shape}")

    return fp4_vals, float_scales


def pack_tpu_fp4_from_fp32(codes_fp32: torch.Tensor) -> torch.Tensor:
    """Pack float32 FP4 codes into TPU FP4 uint8 blocks."""
    return pack_fp4_from_fp32(codes_fp32)


def ensure_tpu_fp4_registered() -> None:
    """Register the TPU FP4 quantization config with vLLM if available."""
    if TPU_FP4_QUANT_METHOD in QUANTIZATION_METHODS:
        _extend_tpu_supported_quantization()
        return
    register_quantization_config(TPU_FP4_QUANT_METHOD)(TpuFp4Config)
    _extend_tpu_supported_quantization()


def _extend_tpu_supported_quantization() -> None:
    from tpu_inference.platforms import TpuPlatform  # type: ignore

    methods = getattr(TpuPlatform, "supported_quantization", None)
    if isinstance(methods, list) and TPU_FP4_QUANT_METHOD not in methods:
        methods.append(TPU_FP4_QUANT_METHOD)
