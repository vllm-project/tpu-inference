# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tiling, config-building and activation support for the fused GMM kernel.

Forked from `gmm_v2.py` in this same package; the shortest route through
this file is to read it as a diff against that one.
"""

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Util.


def swigluoai(gate: jax.Array,
              up: jax.Array,
              *,
              alpha: float = 1.702,
              limit: float = 7.0) -> jax.Array:
    """Activation used in some models such as GPT-OSS."""

    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    glu = gate * jax.nn.sigmoid(alpha * gate)
    return (up + 1.0) * glu


def silu_and_mul_with_clamp(gate: jax.Array,
                            up: jax.Array,
                            limit: float = 10.0) -> jax.Array:
    """Activation used in some models DeepSeek V4."""
    # The limit value is from DSV4's config.
    # TODO: pass limit from model config, instead of hardcoding here.
    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    return jax.nn.silu(gate) * up


def apply_act_fn(acc: jax.Array, fuse_act: str | None):
    """Applies a fused activation function to the accumulator.

    This function is used when an activation function is fused with the matrix
    multiplication. The input accumulator `acc` is expected to contain
    concatenated results for both the 'gate' and 'up' projections.

    Args:
        acc: The accumulator array, with the last dimension being 2 * tile_n.
        fuse_act: The name of the activation function to apply.

    Returns:
        The result of applying the activation function.

    Raises:
        NotImplementedError: If an unsupported `fuse_act` is provided.
    """

    if fuse_act is None:
        return acc

    acc_gate, acc_up = jnp.split(acc, 2, -1)
    match fuse_act:
        case "silu":
            return jax.nn.silu(acc_gate) * acc_up
        case "gelu":
            return jax.nn.gelu(acc_gate) * acc_up
        case "gelu_tanh":
            return jax.nn.gelu(acc_gate, approximate=True) * acc_up
        case "swigluoai":
            return swigluoai(acc_gate, acc_up)
        case "silu_and_mul_with_clamp":
            return silu_and_mul_with_clamp(acc_gate, acc_up)
        case _:
            raise NotImplementedError(
                f"Unsupported activation function: {fuse_act}")


def align_to(x, a):
    return pl.cdiv(x, a) * a


# Block size (columns along k) for the in-kernel dynamic lhs quantization.
from tpu_inference.kernels.megablox.gmm_v2 import LHS_QUANT_BLOCK_SIZE


def _is_pow2(b) -> bool:
    return isinstance(b, int) and b > 0 and (b & (b - 1)) == 0


# Pallas scalars are signed; these helpers assume a >= 0, which lets a
# static power-of-two divisor lower to a single shift or mask.


def _udiv(a, b):
    """Unsigned division fallback for non-power-of-two static b."""
    if isinstance(a, jax.Array):
        return (a.astype(jnp.uint32) // jnp.uint32(b)).astype(jnp.int32)
    return a // b


def unsigned_floor_div(a, b):
    """a // b assuming a >= 0; single shift for power-of-two b."""
    if _is_pow2(b):
        return a >> (b.bit_length() - 1)
    return _udiv(a, b)


def unsigned_mod(a, b):
    """a % b assuming a >= 0; single mask for power-of-two b."""
    if _is_pow2(b):
        return a & (b - 1)
    if isinstance(a, jax.Array):
        return (a.astype(jnp.uint32) % jnp.uint32(b)).astype(jnp.int32)
    return a % b


def unsigned_cdiv(a, b):
    """pl.cdiv(a, b) assuming a >= 0; add+shift for power-of-two b."""
    if _is_pow2(b):
        return (a + b - 1) >> (b.bit_length() - 1)
    return _udiv(a + b - 1, b)


def unsigned_align_down(a, b):
    """(a // b) * b assuming a >= 0; single mask for power-of-two b."""
    if _is_pow2(b):
        return a & ~(b - 1)
    return a - unsigned_mod(a, b)


# Define data classes.


class RhsRef(ABC):
    """Abstract class that defines interfaces for rhs values."""

    @abstractmethod
    def get_weight(self) -> jax.Array:
        ...

    @abstractmethod
    def get_scale(self) -> jax.Array:
        ...

    @abstractmethod
    def get_bias(self) -> jax.Array:
        ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef(RhsRef):
    """Dataclass for a single weights."""

    weight: Any
    scale: Any | None
    bias: Any | None

    def get_weight(self) -> jax.Array:
        return self.weight[...]

    def get_scale(self) -> jax.Array:
        return self.scale[...]

    def get_bias(self) -> jax.Array:
        return self.bias[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef(RhsRef):
    """Dataclass for gate and up weights used in fused activation."""

    gate: WeightsRef
    up: WeightsRef

    def get_weight(self) -> jax.Array:
        w_gate = self.gate.get_weight()
        w_up = self.up.get_weight()
        return jnp.concatenate([w_gate, w_up], axis=-1)

    def get_scale(self) -> jax.Array:
        s_gate = self.gate.get_scale()
        s_up = self.up.get_scale()
        return jnp.concatenate([s_gate, s_up], axis=-1)

    def get_bias(self) -> jax.Array:
        b_gate = self.gate.get_bias()
        b_up = self.up.get_bias()
        return jnp.concatenate([b_gate, b_up], axis=-1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    # gm_id_to_group_id uses only the low 9 bits (group_id < 512); the
    # high bits are reserved, do not pack anything else into them.
    gm_id_to_group_id: jax.Array
    gm_id_to_m_offset: jax.Array
    # Precomputed in fill_metadata:
    #   row_start[gm] = m_offset[gm] // size_lhs_sublane
    #   row_size[gm] = cdiv(m_offset[gm+1], size_lhs_sublane) - row_start[gm]
    gm_id_to_row_start: jax.Array
    gm_id_to_row_size: jax.Array


@dataclasses.dataclass(frozen=True)
class TileSizes:
    tile_m: int
    tile_k: int
    tile_n: int


@dataclasses.dataclass(frozen=True)
class Dimensions:
    size_m: int
    size_k: int
    size_n: int
    size_group: int
    size_lhs_group: int
    size_lhs_sublane: int


@dataclasses.dataclass(frozen=True)
class InputConfigs:
    quant_dtype: jnp.dtype | None
    quant_block_size: int | None
    dtype: jnp.dtype
    has_bias: bool = False
    has_scale: bool = False

    @property
    def should_bitcast(self) -> bool:
        bits = jax.dtypes.itemsize_bits(self.dtype)
        return bits < 8

    @property
    def should_dequantize_before_matmul(self) -> bool:
        """Dequantize rhs before matmul if block size limits MXU utilization."""
        if not self.has_scale:
            return False
        mxu_size = pltpu.get_tpu_info().mxu_column_size
        return self.quant_block_size < mxu_size

    @property
    def should_dequantize_after_matmul(self) -> bool:
        return self.has_scale and not self.should_dequantize_before_matmul


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
    tiles: TileSizes
    dims: Dimensions
    lhs_cfgs: InputConfigs
    rhs_cfgs: InputConfigs
    out_dtype: jnp.dtype
    acc_dtype: jnp.dtype
    zero_init: bool
    fuse_act: str | None
    # Use unconditional (always_changes) pipeline waits/copies for refs
    # whose block indices change on every grid step.
    unconditional_pipeline: bool = True

    @property
    def num_quant_blocks_per_tile_k(self) -> int:
        return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)

    @property
    def out_size_n(self) -> int:
        if self.fuse_act is None:
            return self.dims.size_n
        else:
            return self.dims.size_n // 2


TileFn = Callable[[Dimensions, InputConfigs, InputConfigs, int, str | None],
                  TileSizes]


class IndexMaps:
    """Index maps for GMM kernel."""

    def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs):
        self.metadata_ref = metadata_ref
        self.cfgs = cfgs

    def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        row_start = self.metadata_ref.gm_id_to_row_start[gm_id]
        row_size = self.metadata_ref.gm_id_to_row_size[gm_id]

        return (pl.ds(row_start, row_size), 0, k_id)

    def rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array,
                             k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, k_id, n_id)

    def rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array,
                           _: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, 0, n_id)

    def rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array,
                            k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        # Simply multiplying k_id by num_quant_blocks_per_tile_k will not work
        # since a single quant block could be shared along multiple k tile.
        k_row = k_id * self.cfgs.tiles.tile_k
        b_row = unsigned_floor_div(k_row, self.cfgs.rhs_cfgs.quant_block_size)
        b_tile_id = unsigned_floor_div(b_row,
                                       self.cfgs.num_quant_blocks_per_tile_k)
        return (group_id, b_tile_id, 0, n_id)

    def out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
        is_last_gm = gm_id == (pl.num_programs(1) - 1)
        row_start = self.metadata_ref.gm_id_to_row_start[gm_id]
        # capped_row_end (m_end // sublane) is the next tile's row_start;
        # last_row_end (cdiv(m_end, sublane)) is row_start + row_size.
        capped_row_end = self.metadata_ref.gm_id_to_row_start[gm_id + 1]
        last_row_end = row_start + self.metadata_ref.gm_id_to_row_size[gm_id]
        row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
        metadata_ref: MetadataRef, cfgs: GmmConfigs
) -> Tuple[Tuple[WeightsRef, WeightsRef], pl.BlockSpec]:
    """Generates block specs for the given lhs, rhs, and out refs."""

    index_map = IndexMaps(metadata_ref, cfgs)
    bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m //
                                       cfgs.dims.size_lhs_sublane)

    # always_changes is sound only for refs whose block index changes on
    # every grid step: lhs always, out only when tile_k covers size_k.
    # Feature-detected because older pl.Buffered lacks the field.
    supports_always_changes = any(f.name == "always_changes"
                                  for f in dataclasses.fields(pl.Buffered))
    lhs_unconditional = cfgs.unconditional_pipeline
    out_unconditional = (cfgs.unconditional_pipeline
                         and pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k) == 1)

    def buffered(unconditional):
        if supports_always_changes:
            return pl.Buffered(buffer_count=2, always_changes=unconditional)
        return pl.Buffered(buffer_count=2)

    lhs_weight_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
        index_map.lhs_index_map,
        pipeline_mode=buffered(lhs_unconditional),
    )

    lhs_block_spec = WeightsRef(
        weight=lhs_weight_spec,
        scale=None,
        bias=None,
    )

    tile_k_rhs = cfgs.tiles.tile_k
    if cfgs.rhs_cfgs.should_bitcast:
        packing = pl.cdiv(32, jax.dtypes.itemsize_bits(cfgs.rhs_cfgs.dtype))
        tile_k_rhs //= packing

    # rhs weight/scale stay predicated: their indices repeat when an expert
    # spans multiple gm tiles, and always_changes needs buffer_count == 2.
    rhs_weight_spec = pl.BlockSpec(
        (None, tile_k_rhs, cfgs.tiles.tile_n),
        index_map.rhs_weight_index_map,
        pipeline_mode=pl.Buffered(buffer_count=3),
    )
    rhs_scale_block_spec = rhs_bias_block_spec = None
    if cfgs.rhs_cfgs.has_bias:
        rhs_bias_block_spec = pl.BlockSpec(
            (None, 1, cfgs.tiles.tile_n),
            index_map.rhs_bias_index_map,
        )
    if cfgs.rhs_cfgs.has_scale:
        rhs_scale_block_spec = pl.BlockSpec(
            (None, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n),
            index_map.rhs_scale_index_map,
        )

    rhs_block_spec = WeightsRef(
        weight=rhs_weight_spec,
        scale=rhs_scale_block_spec,
        bias=rhs_bias_block_spec,
    )

    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
        index_map.out_index_map,
        pipeline_mode=buffered(out_unconditional),
    )

    return (lhs_block_spec, rhs_block_spec), out_block_spec


# Define kernels.


def matmul_tile(
    tiled_lhs: jax.Array,  # [tile_m, tile_k]
    tiled_rhs_ref: RhsRef,  # [tile_k, tile_n]
    *,
    cfgs: GmmConfigs,
    is_last_k_step: bool,
) -> jax.Array:
    """Computes one [tile_m, tile_n] matmul tile.

    Takes the lhs tile as a VALUE (not a ref) so callers can feed either a
    pipelined VMEM ref load or an in-register intermediate. Returns the
    accumulator value in cfgs.acc_dtype; the caller handles cross-k
    accumulation, bias, activation, masking and output stores.

    ROW COUNT: the matmul's m extent is taken from the lhs value's own
    leading dimension, NOT from cfgs.tiles.tile_m, so a caller may hand in
    a row-count-bucketed tile and get back a [bucket_m, tile_n]
    accumulator.
    """
    tpu_info = pltpu.get_tpu_info()
    mxu_size = tpu_info.mxu_column_size

    # Step 1: Input pre-processing.
    tiled_rhs = tiled_rhs_ref.get_weight()
    # When rhs is packed into uint32, bitcast unpacks it along the K axis,
    # expanding K back to tile_k.
    if cfgs.rhs_cfgs.should_bitcast:
        tiled_rhs = pltpu.bitcast(tiled_rhs, cfgs.rhs_cfgs.dtype)
    rhs_tile_n = tiled_rhs.shape[1]

    # This should only be taken in the case where we don't requantize
    # the scales and thus we need to dequantize inside VMEM to avoid small
    # contracting dimmensions
    rhs_qbs = cfgs.rhs_cfgs.quant_block_size
    if cfgs.rhs_cfgs.should_dequantize_before_matmul:
        tiled_rhs_scale = tiled_rhs_ref.get_scale().astype(cfgs.lhs_cfgs.dtype)
        num_blocks = cfgs.num_quant_blocks_per_tile_k
        tiled_rhs_dequant = tiled_rhs.astype(cfgs.lhs_cfgs.dtype).reshape(
            num_blocks, rhs_qbs, rhs_tile_n)
        tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
        tiled_rhs = tiled_rhs_dequant.reshape(cfgs.tiles.tile_k, rhs_tile_n)
        rhs_qbs = cfgs.tiles.tile_k

    valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
    if is_last_k_step and valid_k != 0:
        mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape,
                                        0) < valid_k
        tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)

    # Step 2: Matmul.
    acc_rows = tiled_lhs.shape[0]
    acc_list = []
    if cfgs.lhs_cfgs.quant_dtype is None:
        # Unquantized matmul path.
        for start_n in range(0, rhs_tile_n, mxu_size):
            end_n = min(rhs_tile_n, start_n + mxu_size)
            col_size = end_n - start_n

            acc_n = jnp.zeros((acc_rows, col_size), dtype=cfgs.acc_dtype)
            for start_k in range(0, cfgs.tiles.tile_k, rhs_qbs):
                end_k = min(cfgs.tiles.tile_k, start_k + rhs_qbs)

                block_acc = jnp.matmul(
                    tiled_lhs[:, start_k:end_k],
                    tiled_rhs[start_k:end_k, start_n:end_n],
                    preferred_element_type=jnp.float32,
                ).astype(cfgs.acc_dtype)

                if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                    b_id = start_k // rhs_qbs
                    tiled_rhs_scale = tiled_rhs_ref.get_scale()
                    block_acc *= tiled_rhs_scale[b_id, :,
                                                 start_n:end_n].astype(
                                                     cfgs.acc_dtype)

                acc_n += block_acc
            acc_list.append(acc_n)
    else:
        # Quantized matmul path.
        lhs_q_dtype = cfgs.lhs_cfgs.quant_dtype
        q_block_size = cfgs.lhs_cfgs.quant_block_size

        if jnp.issubdtype(lhs_q_dtype, jnp.floating):
            dtype_max = float(jnp.finfo(lhs_q_dtype).max)
            preferred_element_type = jnp.float32
        else:
            dtype_max = float(jnp.iinfo(lhs_q_dtype).max)
            preferred_element_type = jnp.int32

        # Without n outer loop, result of quantized matmul becomes available only
        # at the last iteration of the loop. This means [tile_m, tile_n] value
        # needs to be stored until the last iteration. By adding n outer loop,
        # result of [tile_m, mxu_size] becomes available at the end of every k
        # inner loop which can be used to pipeline subsequent VPU or VST ops with
        # MXU ops for the next [tile_m, mxu_size].
        for start_n in range(0, rhs_tile_n, mxu_size):
            end_n = min(rhs_tile_n, start_n + mxu_size)
            col_size = end_n - start_n

            acc_n = jnp.zeros((acc_rows, col_size), dtype=cfgs.acc_dtype)
            for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
                end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)

                block_lhs = tiled_lhs[:, start_k:end_k]
                block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]

                # Perform lhs quantization. Note that for every block_lhs,
                # same computation will be performed tiles_n//mxu_size times.
                # But we can let compiler perform CSE and avoid recomputation.
                block_abs_max = jnp.max(jnp.abs(block_lhs),
                                        axis=1,
                                        keepdims=True)
                block_scale = block_abs_max / dtype_max

                # If block_scale=0, it will cause division by zero and return either
                # NaN or Inf. Since this can cause numeric issue when downcasting to
                # quantized value, we convert them into 0.
                block_scale_inv = jnp.where(block_scale == 0, 0,
                                            1 / block_scale)
                # Convert lhs into quantized dtype.
                block_lhs_q = (block_lhs * block_scale_inv).astype(lhs_q_dtype)

                # Unlike unquantized path, compiler may not perform implicit type
                # conversion due to numeric concerns. As this can cause unsupported
                # matmul error, explicit type conversion is performed.
                if not tpu_info.is_matmul_supported(lhs_q_dtype,
                                                    block_rhs.dtype):
                    block_rhs = block_rhs.astype(lhs_q_dtype)

                block_acc = jnp.matmul(
                    block_lhs_q,
                    block_rhs,
                    preferred_element_type=preferred_element_type,
                ).astype(cfgs.acc_dtype)

                block_acc *= block_scale.astype(cfgs.acc_dtype)

                # Apply rhs subchannel scale per quant block.
                if cfgs.rhs_cfgs.should_dequantize_after_matmul:
                    b_id = start_k // rhs_qbs
                    rhs_scale_slice = tiled_rhs_ref.get_scale()
                    block_acc *= rhs_scale_slice[b_id, :,
                                                 start_n:end_n].astype(
                                                     cfgs.acc_dtype)

                acc_n += block_acc
            acc_list.append(acc_n)
    return jnp.concatenate(acc_list, axis=1)


def compute_local_row_bounds(metadata_ref: MetadataRef, gm_id: jax.Array,
                             sublane: int) -> tuple[jax.Array, jax.Array]:
    """Tile-local row window [m_start_local, m_end_local) of gm_id's group.

    Rows outside this window hold values from adjacent groups and must be
    masked.
    """
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = unsigned_align_down(m_start, sublane)
    return m_start - m_offset, m_end - m_offset


def mask_out_of_group_rows(acc: jax.Array, m_start_local: jax.Array,
                           m_end_local: jax.Array) -> jax.Array:
    """Zeroes rows outside [m_start_local, m_end_local) along axis 0."""
    iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
    mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
    return jnp.where(mask, acc, 0)


def store_output_tile(
    acc_masked: jax.Array,  # [tile_m // sublane, sublane, tile_n]
    tiled_out_ref: jax.Array,  # same shape
    partial_out_ref: jax.Array,  # [sublane, tile_n]
    gm_id: jax.Array,
    m_end_local: jax.Array,
    *,
    sublane: int,
):
    """Writes a masked tile with sublane-boundary partial accumulation.

    ROW COUNT: acc_masked may cover only the leading `acc_masked.shape[0]`
    sublane rows of the tile; the trailing rows of tiled_out_ref are then
    left untouched, and the pipeline's bounded output window never reaches
    past what was written.
    """
    # Write the final output to the output ref.
    num_rows = acc_masked.shape[0]
    if num_rows == tiled_out_ref.shape[0]:
        tiled_out_ref[...] = acc_masked.astype(tiled_out_ref.dtype)
    else:
        tiled_out_ref[:num_rows] = acc_masked.astype(tiled_out_ref.dtype)

    # If this is the first tile for grid[n_id, :, :], we initialize the
    # partial out to zeros. Otherwise, partial out from last tile of
    # grid[n_id-1, :, :] can be used and cause numeric issues.
    partial_out_zeros = jnp.zeros_like(partial_out_ref)

    # Accumulate the partial output from the previous step.
    tiled_out_ref[0] += jnp.where(gm_id == 0, partial_out_zeros,
                                  partial_out_ref[...])

    # Consider following case where size_lhs_sublane = 4, number denotes group
    # id and | denotes boundaries between sublanes:
    # | 0 0 1 2 | 2 2 2 2 | 3 3 4 4 |
    #
    # Assuming group id of current step is 1, current step will not completely
    # fill size_lhs_sublane rows and will be revisited at the next step. By
    # storing the partial rows into the partial_out_ref, the next step can
    # read them and accumulate to them.  Additionally, for group id of 2,
    # since it completely fills the size_lhs_sublane rows, we need to zero out
    # partial_out_ref to avoid numeric error for group 3.
    last_row = unsigned_floor_div(m_end_local, sublane)
    partial_out_ref[...] = jnp.where(
        unsigned_mod(m_end_local, sublane) == 0,
        partial_out_zeros,
        tiled_out_ref[last_row],
    )


def fill_metadata(
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
    """Fills the metadata for the given lhs group sizes and group offset.

    Iterates over the lhs group sizes and if the group id is valid, determines
    the number of gm tiles that are needed to process the current group. Then,
    it fills starting and ending offset (gm_id_to_m_offset), and the group id
    (gm_id_to_group_id) for each gm tile.

    Args:
        lhs_group_sizes_ref: The group sizes of lhs.
        group_offset_ref: Offset of the first group to process.
        metadata_ref: Metadata that is used to determine the group id and m offsets
            for each gmm tile.
        cfgs: GmmConfigs.

    Returns:
        The number of gm tiles to process lhs with given group offset.
    """

    group_offset = group_offset_ref[0]
    max_num_group = group_offset + cfgs.dims.size_group
    sublane = cfgs.dims.size_lhs_sublane
    tile_m = cfgs.tiles.tile_m

    def write_tile(tm_id, group_id, curr, nxt):
        # The [tm_id] entries come from the previous tile's [tm_id + 1]
        # writes, so each tile only writes its own + 1 entries.
        metadata_ref.gm_id_to_group_id[tm_id] = group_id
        metadata_ref.gm_id_to_m_offset[tm_id + 1] = nxt
        metadata_ref.gm_id_to_row_start[tm_id + 1] = unsigned_floor_div(
            nxt, sublane)
        metadata_ref.gm_id_to_row_size[tm_id] = (
            unsigned_cdiv(nxt, sublane) - unsigned_floor_div(curr, sublane))

    @jax.named_scope("prefix_sum_loop")
    def prefix_sum(i, s):
        return s + lhs_group_sizes_ref[i]

    prefix = lax.fori_loop(0, group_offset, prefix_sum, 0)
    # write_tile only writes [tm_id + 1] entries, so seed [0] here.
    metadata_ref.gm_id_to_m_offset[0] = prefix

    # Each group's first tile is stored unconditionally to slot num_gm,
    # which advances by 0 for an empty group. Skipped groups write up to
    # index num_gm + 1, which the scratch at the pallas_call is padded for.
    @jax.named_scope("group_scan_loop")
    def group_scan(lhs_group_id, carry):
        num_gm, start = carry
        group_id = lhs_group_id - group_offset
        group_size = lhs_group_sizes_ref[lhs_group_id]
        end = start + group_size

        local_offset = unsigned_mod(start, sublane)
        tm_size = jnp.minimum(tile_m - local_offset, group_size)
        nxt = start + tm_size
        write_tile(num_gm, group_id, start, nxt)

        # Rare path: group spans more than one tile.
        @jax.named_scope("spill_tm_loop")
        def spill_body(c):
            tm_id, curr = c
            lo = unsigned_mod(curr, sublane)
            sz = jnp.minimum(tile_m - lo, end - curr)
            nx = curr + sz
            write_tile(tm_id, group_id, curr, nx)
            return tm_id + 1, nx

        tm_final, _ = lax.while_loop(lambda c: c[1] < end, spill_body,
                                     (num_gm + 1, nxt))

        num_gm = jnp.where(group_size > 0, tm_final, num_gm)
        return num_gm, end

    num_gm, _ = lax.fori_loop(group_offset, max_num_group, group_scan,
                              (0, prefix))

    # zero_out_start reads m_offset[0] and m_offset[num_gm], and must see
    # 0 to zero the whole output when no group produced a tile.
    m0 = metadata_ref.gm_id_to_m_offset[0]
    m0 = jnp.where(num_gm == 0, 0, m0)
    metadata_ref.gm_id_to_m_offset[0] = m0
    # row_start[0] = m_offset[0] // sublane, like every other tile: with
    # group_offset != 0 the first tile's rows start at the prefix offset,
    # not at row 0.
    metadata_ref.gm_id_to_row_start[0] = unsigned_floor_div(m0, sublane)
    return num_gm


def zero_out_start(
    out_ref: jax.Array,  # [size_m, size_n]
    zero_ref: jax.Array,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array,  # [1]
    metadata_ref: MetadataRef,
    num_gm: jax.Array,
    *,
    dims: Dimensions,
):
    """Zero out output rows that are not used in the computation."""

    num_lanes = pltpu.get_tpu_info().num_lanes
    assert num_lanes == zero_ref.shape[-1]
    zero_ref[...] = jnp.zeros_like(zero_ref)

    zero_dma = zero_ref.reshape(-1, dims.size_lhs_sublane, num_lanes)
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
    row_size = zero_dma.shape[0]

    compute_start = metadata_ref.gm_id_to_m_offset[0]
    compute_end = metadata_ref.gm_id_to_m_offset[num_gm]

    left_zero_start = 0
    left_zero_end = unsigned_floor_div(compute_start, dims.size_lhs_sublane)
    left_zero_size = left_zero_end - left_zero_start
    left_num_loops = unsigned_cdiv(left_zero_size, row_size)

    right_zero_start = unsigned_cdiv(compute_end, dims.size_lhs_sublane)
    right_zero_end = out_dma.shape[0]
    # Clamp: with invalid inputs this goes negative, and unsigned_cdiv on
    # a negative wraps to ~2^28 iterations of unbounded DMAs.
    right_zero_size = jnp.maximum(right_zero_end - right_zero_start, 0)
    right_num_loops = unsigned_cdiv(right_zero_size, row_size)

    def fill_zero(i, zero_size, *, start, end):
        dma_start = start + i * row_size
        dma_end = jnp.minimum(dma_start + row_size, end)
        dma_size = dma_end - dma_start

        # Static loop. Will be unrolled during compile time.
        for n_start in range(0, out_dma.shape[-1], num_lanes):
            n_end = n_start + num_lanes
            pltpu.make_async_copy(
                src_ref=zero_dma.at[pl.ds(0, dma_size)],
                dst_ref=out_dma.at[pl.ds(dma_start, dma_size), :,
                                   n_start:n_end],
                sem=semaphore_ref.at[0],
            ).start(priority=1)

        return zero_size + dma_size

    @jax.named_scope("left_fill_zero")
    def left_fill_zero(i, zero_size):
        return fill_zero(i,
                         zero_size,
                         start=left_zero_start,
                         end=left_zero_end)

    @jax.named_scope("right_fill_zero")
    def right_fill_zero(i, zero_size):
        return fill_zero(i,
                         zero_size,
                         start=right_zero_start,
                         end=right_zero_end)

    zero_size = lax.fori_loop(0, left_num_loops, left_fill_zero, 0)
    zero_size = lax.fori_loop(0, right_num_loops, right_fill_zero, zero_size)
    return zero_size


def zero_out_end(
    out_ref: jax.Array,  # [size_m, size_n]
    semaphore_ref: jax.Array,  # [1]
    zero_size: jax.Array,
    *,
    dims: Dimensions,
):
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
    pltpu.make_async_copy(
        src_ref=out_dma.at[pl.ds(0, zero_size)],
        dst_ref=out_dma.at[pl.ds(0, zero_size)],
        sem=semaphore_ref.at[0],
    ).wait()


def sublane_tiling(dtype: jnp.dtype) -> int:
    """Sublane tiling for dtype on this TPU generation.

    The device record raises NotImplementedError past generation 7, so
    from 7 on its rule is applied directly: the large second-minor
    tiling, num_sublanes * (32 // bitwidth).
    """
    tpu_info = pltpu.get_tpu_info()
    if tpu_info.generation < 7:
        return tpu_info.get_sublane_tiling(dtype)
    return tpu_info.num_sublanes * (32 // jax.dtypes.itemsize_bits(dtype))


def validate_inputs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    fuse_act: str | None = None,
) -> Dimensions:
    """Validates the inputs for the GMM kernel."""

    size_m = lhs.shape[0]
    size_group, size_k, size_n = rhs.shape
    size_lhs_group = group_sizes.shape[0]

    assert size_group <= size_lhs_group
    assert lhs.shape == (size_m, size_k)
    if rhs_bias is not None:
        assert rhs_bias.shape == (size_group, 1, size_n)
    if rhs_scale is not None:
        num_quant_blocks = rhs_scale.shape[1]
        assert rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n)
        assert size_k % num_quant_blocks == 0
    assert group_offset.shape == (1, )

    size_lhs_sublane = sublane_tiling(lhs.dtype)
    if jax.dtypes.itemsize_bits(lhs.dtype) == 8:
        # 8-bit lhs tiles sublanes in groups of 32, so the sublane count
        # cannot be clamped to size_m: callers pad rows instead.
        assert size_m % size_lhs_sublane == 0, (
            f"8-bit lhs requires size_m ({size_m}) to be a multiple of the "
            f"sublane tiling ({size_lhs_sublane}). Pad lhs rows to a "
            f"multiple of {size_lhs_sublane} "
            "and slice the output back; group_sizes can stay unchanged "
            "(padded rows land past m_offset[num_gm] and are never read).")
    else:
        size_lhs_sublane = min(size_lhs_sublane, size_m)
        assert size_m % size_lhs_sublane == 0, (
            f"size_m ({size_m}) must be a multiple of the sublane tiling "
            f"({size_lhs_sublane}). Pad lhs rows to a multiple and slice "
            "the output back; padded rows past m_offset[num_gm] are never "
            "read.")
    if fuse_act is not None:
        num_lanes = pltpu.get_tpu_info().num_lanes
        if size_n % (2 * num_lanes) != 0:
            raise ValueError(
                f"{size_n=} should be divisible by 2 * num_lanes when fuse_act is "
                "enabled since we need to split n dimension for gate and up.")

    return Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_group=size_group,
        size_lhs_group=size_lhs_group,
        size_lhs_sublane=size_lhs_sublane,
    )


def get_cost_estimate(cfgs: GmmConfigs):
    """Returns the cost estimate for the GMM kernel."""

    dims = cfgs.dims
    lhs_dtype = cfgs.lhs_cfgs.quant_dtype or cfgs.lhs_cfgs.dtype
    rhs_dtype = cfgs.rhs_cfgs.dtype

    # We use bits for rhs since it could sub-byte dtype like int4.
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)
    fp32_bytes = jnp.dtype(jnp.float32).itemsize

    # TODO(kyuyeunk): Add compute flops for quant, dequant, and bias.
    flops = 2 * dims.size_m * dims.size_k * dims.size_n

    lhs_bytes = dims.size_m * dims.size_k * lhs_dtype.itemsize

    rhs_size = dims.size_group * dims.size_k * dims.size_n
    rhs_bytes = rhs_size * rhs_bits // 8
    if cfgs.rhs_cfgs.has_scale:
        num_quant_blocks = pl.cdiv(dims.size_k, cfgs.rhs_cfgs.quant_block_size)
        rhs_bytes += dims.size_group * num_quant_blocks * dims.size_n * fp32_bytes
    if cfgs.rhs_cfgs.has_bias:
        rhs_bytes += dims.size_group * dims.size_n * fp32_bytes

    out_bytes = dims.size_m * cfgs.out_size_n * cfgs.out_dtype.itemsize

    total_bytes = lhs_bytes + rhs_bytes + out_bytes

    return pl.CostEstimate(
        flops=flops,
        bytes_accessed=total_bytes,
        transcendentals=0,
    )


def make_gmm_configs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    *,
    tile_info: TileSizes | TileFn,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype | None,
    acc_dtype: jnp.dtype | None,
    maybe_quantize_lhs: bool,
    zero_initialize: bool,
    fuse_act: str | None = None,
    unconditional_pipeline: bool = True,
):
    """Fills the GMM config for the GMM kernel."""

    dims = validate_inputs(lhs, rhs, rhs_scale, rhs_bias, group_sizes,
                           group_offset, fuse_act)

    if rhs_scale is not None:
        has_scale = True
        rhs_quant_dtype = rhs.dtype
        num_blocks = rhs_scale.shape[1]
        block_size = dims.size_k // num_blocks
    else:
        has_scale = False
        num_blocks = 1
        rhs_quant_dtype = None
        block_size = dims.size_k

    rhs_cfgs = InputConfigs(
        quant_dtype=rhs_quant_dtype,
        quant_block_size=block_size,
        dtype=rhs.dtype,
        has_bias=rhs_bias is not None,
        has_scale=has_scale,
    )

    # Post-matmul dequant walks k in LHS_QUANT_BLOCK_SIZE steps and reads
    # the rhs scale block as start_k // block_size, so a block that is not
    # a whole number of lhs quantization blocks would be skipped.
    if (rhs_cfgs.should_dequantize_after_matmul and num_blocks > 1
            and block_size % LHS_QUANT_BLOCK_SIZE != 0):
        raise ValueError(
            f"rhs scale block size {block_size} ({num_blocks} blocks "
            f"over size_k {dims.size_k}) must be a multiple of the lhs "
            f"quantization block size ({LHS_QUANT_BLOCK_SIZE}); the "
            "k loop would otherwise skip scale blocks")

    lhs_q_dtype = None
    if maybe_quantize_lhs and rhs_cfgs.should_dequantize_after_matmul:
        # Choose lhs quantization dtype based on TPU hardware support.
        is_rhs_float = jnp.issubdtype(rhs_quant_dtype, jnp.floating)
        tpu_info = pltpu.get_tpu_info()
        # Check if there is hardware compute support for rhs dtype group.
        if tpu_info.fp8_ops_per_second > 0:
            # Special handling for 4-bit integer rhs as it can be converted to fp8
            # without a numeric issues. Note that this is not the case for 4-bit
            # floating rhs as conversion to int8 will cause numeric issues.
            is_rhs_4bits = jax.dtypes.itemsize_bits(rhs_quant_dtype) == 4
            if is_rhs_float or is_rhs_4bits:
                lhs_q_dtype = jnp.float8_e4m3fn.dtype
        if tpu_info.int8_ops_per_second > 0:
            if not is_rhs_float:
                lhs_q_dtype = jnp.int8.dtype

    lhs_cfgs = InputConfigs(
        quant_dtype=lhs_q_dtype,
        # Input quantization involves reading all elements in a block to compute
        # scale value. Since this operation is very memory intensive, we use a
        # block size that is small enough to minimize memory overhead but large
        # enough to minimize compute overhead of quantization.
        quant_block_size=LHS_QUANT_BLOCK_SIZE,
        dtype=lhs.dtype,
    )

    if out_dtype is None:
        out_dtype = lhs.dtype

    if acc_dtype is None:
        if lhs_cfgs.quant_dtype is None:
            acc_dtype = jnp.float32.dtype
        else:
            # Input quantization requires elementwise ops which can put pressure on
            # VPUs. Using faster bf16 hardware during accumulation can help offset the
            # pressure.
            acc_dtype = jnp.bfloat16.dtype

    if isinstance(tile_info, TileSizes):
        tiles = tile_info
    else:
        tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)

    return GmmConfigs(
        dims=dims,
        tiles=tiles,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        out_dtype=jnp.dtype(out_dtype),
        acc_dtype=jnp.dtype(acc_dtype),
        zero_init=zero_initialize,
        fuse_act=fuse_act,
        unconditional_pipeline=unconditional_pipeline,
    )


def get_metadata(cfgs: GmmConfigs) -> dict[str, str | int | float]:
    cfgs_dict = dataclasses.asdict(cfgs)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if not isinstance(val, str | int | float):
            val = str(val)
        ret[key] = val
    return ret
