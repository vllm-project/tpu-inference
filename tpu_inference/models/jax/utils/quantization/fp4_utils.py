"""Shared helpers for FP4 nibble packing and unpacking."""

import torch

# FP4 lookup table; copied once, moved to the target device on demand.
FP4_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,  # 0b0000-0b0111
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,  # 0b1000-0b1111
    ],
    dtype=torch.float32,
)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Expand uint8 packed FP4 bytes into float32 codes.

    Expects the last dimension to contain packed bytes (two FP4 values per
    byte). The returned tensor has the same prefix dimensions with the last
    axis doubled.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(
            f"Expected FP4 packed tensor to be uint8, got {packed.dtype}")
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    idx = torch.stack([low, high], dim=-1).flatten(-2)
    lut = FP4_LUT.to(packed.device)
    return lut[idx.long()]


def fp4_indices_from_values(values: torch.Tensor) -> torch.Tensor:
    """Map float32 values to nearest FP4 code indices."""
    if values.dtype != torch.float32:
        raise ValueError("Expected float32 values for FP4 conversion")

    lut = FP4_LUT.to(values.device)
    diff = torch.abs(values.unsqueeze(-1) - lut)
    indices = torch.argmin(diff, dim=-1).to(torch.uint8)

    zero_mask = values == 0
    if zero_mask.any():
        neg_zero_mask = torch.logical_and(zero_mask, torch.signbit(values))
        pos_zero_mask = zero_mask ^ neg_zero_mask
        indices[pos_zero_mask] = 0
        indices[neg_zero_mask] = 8

    return indices


def pack_fp4_from_fp32(codes_fp32: torch.Tensor) -> torch.Tensor:
    """Pack float32 FP4 codes into uint8 bytes."""
    indices = fp4_indices_from_values(codes_fp32)
    if indices.shape[-1] % 2 != 0:
        raise ValueError("Last dimension must be even to pack into bytes")

    low = indices[..., ::2]
    high = indices[..., 1::2]
    return (low | (high << 4)).to(torch.uint8)
