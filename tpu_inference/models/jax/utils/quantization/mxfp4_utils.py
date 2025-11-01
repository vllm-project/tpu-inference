# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

# MXFP4 constants
MXFP4_BLOCK_SIZE: int = 32
# Exponent-only e8m0 scale bias used by MXFP4 scales
MXFP4_SCALE_BIAS: int = 127


# Precompute a small LUT once; move to device on demand (cheap 16-element copy)
FP4_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,       # 0b0000-0b0111
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # 0b1000-0b1111
], dtype=torch.float32)


def _unpack_uint8_to_fp4_values(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 (..., 16) -> fp4 values (..., 32) using low->high nibble order.

    Returns float32 values corresponding to FP4 codebook entries.
    """
    assert packed.dtype == torch.uint8
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    idx = torch.stack([low, high], dim=-1).flatten(-2)
    lut = FP4_LUT.to(packed.device)
    return lut[idx.long()]


def _e8m0_to_scale(u8: torch.Tensor) -> torch.Tensor:
    """Convert e8m0 uint8 exponents to power-of-two scales using MXFP4_SCALE_BIAS.

    Uses ldexp for exact power-of-two scaling: 1.0 * 2**(u8 - bias).
    """
    exponents = (u8.to(torch.int32) - int(MXFP4_SCALE_BIAS)).to(torch.int32)
    ones = torch.ones_like(u8, dtype=torch.float32)
    return torch.ldexp(ones, exponents)


def dequant_mxfp4_to_bf16(blocks_u8: torch.Tensor, scales_u8: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 blocks/scales into bfloat16 values.

    Args:
      blocks_u8: uint8 tensor shaped [..., Kb, 16], each byte holds 2 FP4 codes.
      scales_u8: uint8 tensor shaped [..., Kb], exponent-only e8m0 per 32-value block.

    Returns:
      torch.bfloat16 tensor with last logical dimension K = Kb * 32.
    """
    if blocks_u8.dtype != torch.uint8 or scales_u8.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 inputs, got blocks={blocks_u8.dtype}, scales={scales_u8.dtype}")
    # Unpack FP4 codes to float32 values [..., Kb, 32]
    fp4_vals = _unpack_uint8_to_fp4_values(blocks_u8)  # (..., Kb, 32)
    # Compute power-of-two scales and apply per block
    scales = _e8m0_to_scale(scales_u8).unsqueeze(-1)    # (..., Kb, 1)
    full = (fp4_vals * scales).reshape(*fp4_vals.shape[:-2], fp4_vals.shape[-2] * MXFP4_BLOCK_SIZE)
    return full.to(torch.bfloat16)
