import torch

from .fp4_utils import unpack_fp4

# MXFP4 constants
MXFP4_BLOCK_SIZE: int = 32
# Exponent-only e8m0 scale bias used by MXFP4 scales
MXFP4_SCALE_BIAS: int = 127
# Name used in config.json quantization_config["quant_method"]
MXFP4_QUANT_METHOD: str = "mxfp4"


def e8m0_to_fp32(u8: torch.Tensor) -> torch.Tensor:
    """Convert e8m0 uint8 exponents to power-of-two scales using MXFP4_SCALE_BIAS.

    Uses ldexp for exact power-of-two scaling: 1.0 * 2**(u8 - bias).
    """
    exponents = (u8.to(torch.int32) - int(MXFP4_SCALE_BIAS)).to(torch.int32)
    ones = torch.ones_like(u8, dtype=torch.float32)
    return torch.ldexp(ones, exponents)


def dequant_mxfp4_to_bf16(blocks_u8: torch.Tensor,
                          scales_u8: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 blocks/scales into bfloat16 values.

    Args:
        blocks_u8: uint8 tensor shaped [..., Kb, 16], each byte holds 2 FP4 codes.
        scales_u8: uint8 tensor shaped [..., Kb], exponent-only e8m0 per 32-value block.

    Returns:
        torch.bfloat16 tensor with last logical dimension K = Kb * 32.
    """
    if blocks_u8.dtype != torch.uint8 or scales_u8.dtype != torch.uint8:
        raise ValueError(
            f"Expected uint8 inputs, got blocks={blocks_u8.dtype}, scales={scales_u8.dtype}"
        )
    # Unpack FP4 codes to float32 values [..., Kb, 32]
    fp4_vals = unpack_fp4(blocks_u8)  # (..., Kb, 32)
    # Compute power-of-two scales and apply per block
    scales = e8m0_to_fp32(scales_u8).unsqueeze(-1)  # (..., Kb, 1)
    full = (fp4_vals * scales).reshape(*fp4_vals.shape[:-2],
                                       fp4_vals.shape[-2] * MXFP4_BLOCK_SIZE)
    return full.to(torch.bfloat16)


def unpack_mxfp4_to_fp32(
        blocks_u8: torch.Tensor,
        scales_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode MXFP4 packed blocks and e8m0 scales to float32 codes and scales.

    Args:
        blocks_u8: uint8 tensor shaped [..., Kb, 16], each byte packs two FP4 codes.
        scales_u8: uint8 tensor shaped [..., Kb], exponent-only e8m0 per block.

    Returns:
        (codes_fp32, scales_fp32), where
        - codes_fp32 has shape [..., Kb*32] and dtype float32
        - scales_fp32 has shape [..., Kb] and dtype float32
    """
    if blocks_u8.dtype != torch.uint8 or scales_u8.dtype != torch.uint8:
        raise ValueError(
            f"Expected uint8 inputs, got blocks={blocks_u8.dtype}, scales={scales_u8.dtype}"
        )
    fp4_vals = unpack_fp4(blocks_u8)  # (..., Kb, 32) float32
    codes_fp32 = fp4_vals.reshape(*fp4_vals.shape[:-2],
                                  fp4_vals.shape[-2] * MXFP4_BLOCK_SIZE)
    scales_fp32 = e8m0_to_fp32(scales_u8)  # (..., Kb) float32
    return codes_fp32, scales_fp32
