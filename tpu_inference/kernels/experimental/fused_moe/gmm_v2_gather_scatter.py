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

import dataclasses
import functools
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# --- Base dataclass + block-size helpers (kept here so the package is small) ---


@dataclasses.dataclass(frozen=True)
class TileSizes:
    tile_m: int
    tile_k: int
    tile_n: int


def _recover_quant_block_size(size_k, num_blocks):
    """Round the ceil(size_k / num_blocks) estimate up to the next power of two."""
    if num_blocks <= 1:
        return size_k
    approx = -(-size_k // num_blocks)
    return 1 << (approx - 1).bit_length()


def get_tuned_block_sizes_v2(
    m,
    k,
    n,
    num_current_groups,
    lhs_dtype,
    rhs_dtype,
    maybe_quantize_lhs,
    rhs_quant_block_size,
    default_block_sizes,
    lhs_indices=None,
    output_indices=None,
    fuse_act=None,
    fused=False,
):
    # Device-specific tuned tables were removed; return the default.
    return default_block_sizes


get_tuned_block_sizes = get_tuned_block_sizes_v2


def get_maybe_quantize_lhs(
    rhs_dtype=None,
    rhs_quant_block_size: int | None = None,
    lhs_quant_block_size: int | None = None,
) -> bool:
    """Return whether to online-quantize LHS activations.

    FP4 weights use W4A16 when RHS scale blocks are smaller than the LHS
    online-quant block; otherwise they use W4A8. Other quantized RHS dtypes keep
    the existing online-quantized LHS path.
    """
    if rhs_dtype is not None and jnp.dtype(rhs_dtype) == jnp.float4_e2m1fn:
        if rhs_quant_block_size is None or lhs_quant_block_size is None:
            return False
        return rhs_quant_block_size >= lhs_quant_block_size
    return True


# Util.
def align_to(x, a):
    return pl.cdiv(x, a) * a


# Define data classes.
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    gm_id_to_group_id: jax.Array
    gm_id_to_m_offset: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef:
    weight: Any
    scale: Any | None
    bias: Any | None

    def get_weight(self):
        return self.weight[...]

    def get_scale(self):
        return self.scale[...]

    def get_bias(self):
        return self.bias[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef:
    """Wraps gate and up WeightsRef for fused activation."""

    gate: WeightsRef
    up: WeightsRef

    def get_weight(self):
        return jnp.concatenate([self.gate.weight[...], self.up.weight[...]],
                               axis=-1)

    def get_scale(self):
        return jnp.concatenate([self.gate.scale[...], self.up.scale[...]],
                               axis=-1)

    def get_bias(self):
        return jnp.concatenate([self.gate.bias[...], self.up.bias[...]],
                               axis=-1)


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
    packing: int = 1
    num_quant_blocks: int = 1


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
    tiles: TileSizes
    dims: Dimensions
    lhs_cfgs: InputConfigs
    rhs_cfgs: InputConfigs
    out_dtype: jnp.dtype
    acc_dtype: jnp.dtype
    zero_init: bool
    fuse_act: str | None = None
    has_post_norm: bool = False

    @property
    def num_quant_blocks_per_tile_k(self) -> int:
        return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)

    @property
    def out_size_n(self) -> int:
        if self.fuse_act is None:
            return self.dims.size_n
        return self.dims.size_n // 2


TileFn = Callable[[Dimensions, InputConfigs, InputConfigs, int, str | None],
                  TileSizes]


def apply_act_fn(acc: jax.Array, fuse_act: str | None) -> jax.Array:
    """Apply fused activation (split gate/up, activate, multiply)."""
    if fuse_act is None:
        return acc
    acc_gate, acc_up = jnp.split(acc, 2, -1)
    match fuse_act:
        case "silu":
            return jax.nn.silu(acc_gate) * acc_up
        case "gelu":
            return jax.nn.gelu(acc_gate) * acc_up
        case "swigluoai":
            limit = 7.0
            alpha = 1.702
            acc_gate = jnp.clip(acc_gate, max=limit)
            acc_up = jnp.clip(acc_up, min=-limit, max=limit)
            return (acc_gate * jax.nn.sigmoid(alpha * acc_gate)) * (acc_up + 1)
        case _:
            raise NotImplementedError(f"Unsupported fuse_act: {fuse_act}")


class IndexMaps:
    """Index maps for GMM kernel."""

    def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs):
        self.metadata_ref = metadata_ref
        self.cfgs = cfgs

    def lhs_index_map(self, gm_id: jax.Array, _: jax.Array, k_id: jax.Array):
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, k_id)

    def rhs_weight_index_map(self, gm_id: jax.Array, n_id: jax.Array,
                             k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, k_id, n_id)

    def rhs_bias_index_map(self, gm_id: jax.Array, n_id: jax.Array,
                           _: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, 0, n_id)

    def rhs_scale_index_map(self, gm_id: jax.Array, n_id: jax.Array,
                            k_id: jax.Array):
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        b_id = (k_id *
                self.cfgs.tiles.tile_k) // self.cfgs.rhs_cfgs.quant_block_size
        b_tile_id = b_id // self.cfgs.num_quant_blocks_per_tile_k
        return (group_id, b_tile_id, 0, n_id)

    def out_index_map(self, gm_id: jax.Array, n_id: jax.Array, _: jax.Array):
        is_last_gm = gm_id == (pl.num_programs(0) - 1)
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
        last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
        row_size = row_end - row_start

        return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
        metadata_ref: MetadataRef, cfgs: GmmConfigs
) -> Tuple[Tuple[pl.BlockSpec, WeightsRef], pl.BlockSpec]:
    """Generates block specs for the given lhs, rhs, and out refs."""

    index_map = IndexMaps(metadata_ref, cfgs)
    bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m //
                                       cfgs.dims.size_lhs_sublane)

    lhs_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
        index_map.lhs_index_map,
    )

    rhs_weight_spec = pl.BlockSpec(
        (None, cfgs.tiles.tile_k // cfgs.rhs_cfgs.packing, cfgs.tiles.tile_n),
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
    )

    return (lhs_block_spec, rhs_block_spec), out_block_spec


# Define kernels.
def inner_kernel(
    # In
    tiled_lhs_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_k]
    tiled_rhs_ref: WeightsRef,  # [tile_k, tile_n]
    # Out
    tiled_out_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n] or [tile_m, tile_n] in scatter_mode
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, partial_out_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
    gs_gate_ref=None,
    gs_up_ref=None,
    scatter_mode: bool = False,
    _k_id: int | None = None,
    _num_k: int | None = None,
    _gm_id: int | None = None,
    _n_id: int | None = None,
):
    """Inner kernel invoked by emit_pipeline to perform matmul.

    tiled_lhs_ref and tiled_out_ref points to rows [m_start:m_end] of lhs and out.
    Additionally, m_start and m_end does not have to align with tile boundaries
    [m_offset:m_offset+tile_m]. Therefore, rows [m_offset:m_start] and
    [m_end:m_offset+tile_m] of tiled_lhs_ref and tiled_out_ref will contain
    invalid data and needs to be masked out.

    Args:
        tiled_lhs_ref: Contains value lhs[m_start:m_end, k_start:k_end]
        tiled_rhs_ref: Contains value rhs[g_id, k_start:k_end, n_start:n_end]. where
            g_id is the group associated with lhs[m_start:m_end, :]
        tiled_out_ref: Contains value out[m_start:m_end, n_start:n_end]
        partial_out_ref: Contains last size_lhs_sublane rows of the previous output.
            Will be initialized to zero if this is first tile for grid[n_id, :, :].
        acc_ref: Reference to the accumulator.
        metadata_ref: Reference to the metadata.
        cfgs: GmmConfigs.
    """
    # Do not feed an unquantized LHS directly into MXU with a quantized RHS:
    # that mixed-dtype matmul has produced numerically wrong TPU results
    # (~70% cos-sim vs. truth). The quantized RHS path below avoids that in
    # both supported cases: W4A8/W8A8 online-quantizes LHS before matmul,
    # while W4A16 promotes/dequantizes RHS to bf16 before matmul.
    # Small FP4 RHS scale blocks are folded into RHS and requantized to the
    # LHS quant dtype before matmul so their per-block scales are not dropped.
    _resolved_gm_id = pl.program_id(0) if _gm_id is None else _gm_id

    def _matmul(is_first_k_step: bool, is_last_k_step: bool):
        tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[...]
        tiled_rhs = tiled_rhs_ref.get_weight()

        num_quant_blocks_per_tile_k = cfgs.num_quant_blocks_per_tile_k

        # When rhs is packed (quantized dtype packed into uint32), unpack it
        # back to the original dtype using pltpu.bitcast which operates on K
        # axis. This expands the K dimension back to tile_k.
        if cfgs.rhs_cfgs.packing > 1:
            tiled_rhs = pltpu.bitcast(tiled_rhs, cfgs.rhs_cfgs.quant_dtype)

        def _valid_m_mask(shape):
            gm_id = _resolved_gm_id
            m_start = metadata_ref.gm_id_to_m_offset[gm_id]
            m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
            m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
            m_start_local = m_start - m_offset
            m_end_local = m_end - m_offset
            iota = lax.broadcasted_iota(jnp.int32, shape, 0)
            return jnp.logical_and(m_start_local <= iota, iota < m_end_local)

        def _get_scale_slice(b_id, start_n, end_n):
            rhs_scale = tiled_rhs_ref.get_scale()
            return rhs_scale[..., b_id, :, start_n:end_n]

        def _online_quantize(block, axis, quant_dtype):
            if jnp.issubdtype(quant_dtype, jnp.floating):
                dtype_max = float(jnp.finfo(quant_dtype).max)
            else:
                dtype_max = float(jnp.iinfo(quant_dtype).max)

            block_f32 = block.astype(jnp.float32)
            amax = jnp.max(jnp.abs(block_f32), axis=axis, keepdims=True)
            amax_safe = jnp.where(amax == 0, jnp.ones_like(amax), amax)
            block_q = (block_f32 / amax_safe * dtype_max).astype(quant_dtype)
            block_scale = amax / dtype_max
            return block_q, block_scale

        valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
        if is_last_k_step and valid_k != 0:
            mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape,
                                            0) < valid_k
            tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)
            mask_lhs = lax.broadcasted_iota(jnp.int32, tiled_lhs.shape,
                                            1) < valid_k
            tiled_lhs = jnp.where(mask_lhs, tiled_lhs, 0)

        if cfgs.rhs_cfgs.quant_dtype is None:
            # Unquantized RHS matmul path.
            acc_list = []
            mxu_size = pltpu.get_tpu_info().mxu_column_size
            rhs_qbs = cfgs.rhs_cfgs.quant_block_size
            rhs_tile_n = tiled_rhs.shape[-1]
            for start_n in range(0, rhs_tile_n, mxu_size):
                end_n = min(rhs_tile_n, start_n + mxu_size)
                col_size = end_n - start_n

                acc_n = jnp.zeros((cfgs.tiles.tile_m, col_size),
                                  dtype=acc_ref.dtype)
                for b_id in range(num_quant_blocks_per_tile_k):
                    k_start = b_id * rhs_qbs
                    k_end = (b_id + 1) * rhs_qbs
                    partial_result = jnp.matmul(
                        tiled_lhs[:, k_start:k_end],
                        tiled_rhs[k_start:k_end, start_n:end_n],
                        preferred_element_type=jnp.float32,
                    )
                    if cfgs.rhs_cfgs.has_scale:
                        rhs_scale_slice = _get_scale_slice(
                            b_id, start_n, end_n)
                        partial_result *= rhs_scale_slice
                    acc_n = acc_n + partial_result
                acc_list.append(acc_n.astype(acc_ref.dtype))
            acc = jnp.concatenate(acc_list, axis=1)
        else:
            # Quantized RHS path. When one stored RHS scale covers a full LHS
            # quant block, apply it after matmul. Otherwise fold the per-slice
            # RHS scales into RHS before matmul so small scale blocks are not
            # dropped. Online quantization avoids reciprocal scales; zero
            # blocks use amax_safe=1 and scale=0.
            q_block_size: int | None = cfgs.lhs_cfgs.quant_block_size
            fp8_activation_quant = cfgs.lhs_cfgs.quant_dtype is not None
            rhs_qbs = cfgs.rhs_cfgs.quant_block_size
            rhs_scale_matches_lhs_block = (rhs_qbs >= q_block_size
                                           and rhs_qbs % q_block_size == 0)
            apply_rhs_scale_after_matmul = cfgs.rhs_cfgs.has_scale and (
                rhs_scale_matches_lhs_block)

            # Without n outer loop, result of quantized matmul becomes available only
            # at the last iteration of the loop. This means [tile_m, tile_n] value
            # needs to be stored until the last iteration. By adding n outer loop,
            # result of [tile_m, mxu_size] becomes available at the end of every k
            # inner loop which can be used to pipeline subsequent VPU or VST ops with
            # MXU ops for the next [tile_m, mxu_size].
            acc_list = []
            mxu_size = pltpu.get_tpu_info().mxu_column_size
            rhs_tile_n = tiled_rhs.shape[-1]
            for start_n in range(0, rhs_tile_n, mxu_size):
                end_n = min(rhs_tile_n, start_n + mxu_size)
                col_size = end_n - start_n

                acc_n = jnp.zeros((cfgs.tiles.tile_m, col_size),
                                  dtype=acc_ref.dtype)

                for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
                    end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)

                    block_lhs = tiled_lhs[:, start_k:end_k]
                    block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]

                    if cfgs.rhs_cfgs.has_scale and not apply_rhs_scale_after_matmul:
                        scaled_rhs_slices = []
                        rhs_k_start = start_k
                        while rhs_k_start < end_k:
                            b_id = rhs_k_start // rhs_qbs
                            rhs_k_end = min((b_id + 1) * rhs_qbs, end_k)
                            local_k_start = rhs_k_start - start_k
                            local_k_end = rhs_k_end - start_k
                            rhs_scale_slice = _get_scale_slice(
                                b_id, start_n, end_n)
                            block_rhs_slice = block_rhs[
                                local_k_start:local_k_end]
                            scaled_rhs_slice = block_rhs_slice.astype(
                                jnp.bfloat16) * rhs_scale_slice.astype(
                                    jnp.bfloat16)
                            scaled_rhs_slices.append(scaled_rhs_slice)
                            rhs_k_start = rhs_k_end
                        block_rhs_for_quant = jnp.concatenate(
                            scaled_rhs_slices, axis=0)
                    else:
                        block_rhs_for_quant = block_rhs.astype(jnp.bfloat16)

                    if fp8_activation_quant:
                        block_lhs_for_matmul, lhs_scale = _online_quantize(
                            block_lhs,
                            axis=1,
                            quant_dtype=cfgs.lhs_cfgs.quant_dtype)
                        if cfgs.rhs_cfgs.has_scale and not apply_rhs_scale_after_matmul:
                            # Fold small RHS scale blocks into RHS first, then
                            # requantize to the activation quant dtype so
                            # matmul uses fp8 x fp8.
                            block_rhs_for_matmul, rhs_online_scale = _online_quantize(
                                block_rhs_for_quant,
                                axis=0,
                                quant_dtype=cfgs.lhs_cfgs.quant_dtype,
                            )
                        else:
                            block_rhs_for_matmul = block_rhs
                            rhs_online_scale = None
                    else:
                        block_lhs_for_matmul = block_lhs.astype(jnp.bfloat16)
                        block_rhs_for_matmul = block_rhs_for_quant

                    block_acc = jnp.matmul(
                        block_lhs_for_matmul,
                        block_rhs_for_matmul,
                        preferred_element_type=jnp.float32,
                    ).astype(acc_ref.dtype)

                    if fp8_activation_quant:
                        block_acc *= lhs_scale.astype(acc_ref.dtype)
                        if rhs_online_scale is not None:
                            block_acc *= rhs_online_scale.astype(acc_ref.dtype)

                    if apply_rhs_scale_after_matmul:
                        b_id = start_k // rhs_qbs
                        rhs_scale_slice = _get_scale_slice(
                            b_id, start_n, end_n)
                        block_acc *= rhs_scale_slice.astype(acc_ref.dtype)

                    acc_n += block_acc

                acc_list.append(acc_n)
            acc = jnp.concatenate(acc_list, axis=1)

        if not is_first_k_step:
            acc += acc_ref[...]

        if is_last_k_step:
            if cfgs.rhs_cfgs.has_scale and gs_gate_ref is not None:
                _gid = metadata_ref.gm_id_to_group_id[_resolved_gm_id]
                if cfgs.fuse_act is not None:
                    n_iota = lax.broadcasted_iota(jnp.int32, acc.shape, 1)
                    global_scale = jnp.where(
                        n_iota >= acc.shape[1] // 2,
                        gs_up_ref[_gid],
                        gs_gate_ref[_gid],
                    )
                else:
                    global_scale = gs_gate_ref[_gid]
                acc *= global_scale.astype(acc.dtype)

            if cfgs.rhs_cfgs.has_bias:
                acc += tiled_rhs_ref.get_bias().astype(acc.dtype)

            acc = apply_act_fn(acc, cfgs.fuse_act)

            gm_id = _resolved_gm_id
            # Mask out rows that does not belong to the current group.
            m_start = metadata_ref.gm_id_to_m_offset[gm_id]
            m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
            m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
            m_start_local = m_start - m_offset
            m_end_local = m_end - m_offset

            iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
            mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)

            if scatter_mode:
                # In scatter mode, write masked acc directly to 2D output.
                # No 3D reshape needed — avoids VMEM tiling mismatch with DMA.
                # No partial_out needed — each scatter tile is independent.
                acc_masked = jnp.where(mask, acc, 0)
                tiled_out_ref[...] = acc_masked.astype(tiled_out_ref.dtype)
            else:
                acc_masked = jnp.where(mask, acc,
                                       0).reshape(tiled_out_ref.shape)

                # Write the final output to the output ref.
                tiled_out_ref[...] = acc_masked.astype(tiled_out_ref.dtype)

                # partial_out is n-aware: use n_id to index into the correct
                # n-tile slice of partial_out_ref (shape: sls × total_n).
                n_id = pl.program_id(1) if _n_id is None else _n_id
                n_offset = n_id * cfgs.tiles.tile_n
                tile_n = cfgs.tiles.tile_n
                sls = cfgs.dims.size_lhs_sublane
                partial_out_zeros = jnp.zeros((sls, tile_n),
                                              dtype=partial_out_ref.dtype)

                # Accumulate the partial output from the previous gm step
                # at the same n position.
                tiled_out_ref[0] += jnp.where(
                    gm_id == 0,
                    partial_out_zeros,
                    partial_out_ref[:, pl.ds(n_offset, tile_n)],
                )

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
                last_row = m_end_local // cfgs.dims.size_lhs_sublane
                partial_out_ref[:, pl.ds(n_offset, tile_n)] = jnp.where(
                    m_end_local % cfgs.dims.size_lhs_sublane == 0,
                    partial_out_zeros,
                    tiled_out_ref[last_row],
                )
        else:
            acc_ref[...] = acc

    # Define matmul wrapper functions.
    @jax.named_scope("matmul_first_last")
    def matmul_first_last():
        _matmul(is_first_k_step=True, is_last_k_step=True)

    @jax.named_scope("matmul_first")
    def matmul_first():
        _matmul(is_first_k_step=True, is_last_k_step=False)

    @jax.named_scope("matmul")
    def matmul():
        _matmul(is_first_k_step=False, is_last_k_step=False)

    @jax.named_scope("matmul_last")
    def matmul_last():
        _matmul(is_first_k_step=False, is_last_k_step=True)

    # Select and execute matmul function based on the current step.
    num_k = pl.num_programs(2) if _num_k is None else _num_k
    k_id = pl.program_id(2) if _k_id is None else _k_id

    is_first_k_step = k_id == 0
    is_last_k_step = k_id == (num_k - 1)

    lax.cond(
        is_first_k_step,
        lambda: lax.cond(
            is_last_k_step,
            matmul_first_last,
            matmul_first,
        ),
        lambda: lax.cond(
            is_last_k_step,
            matmul_last,
            matmul,
        ),
    )


@jax.named_scope("fill_metadata")
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
    metadata_ref.gm_id_to_m_offset[0] = 0

    @jax.named_scope("inner_tm_loop")
    def inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id):
        local_offset = curr_m_offset % cfgs.dims.size_lhs_sublane
        tm_size = jnp.minimum(cfgs.tiles.tile_m - local_offset,
                              end_m_offset - curr_m_offset)

        metadata_ref.gm_id_to_group_id[tm_id] = group_id

        next_m_offset = curr_m_offset + tm_size
        metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
        metadata_ref.gm_id_to_m_offset[tm_id + 1] = next_m_offset

        return next_m_offset

    @jax.named_scope("outer_group_loop")
    def outer_group_loop(lhs_group_id, carry):
        num_gm, start_m_offset = carry

        group_id = lhs_group_id - group_offset
        group_size = lhs_group_sizes_ref[lhs_group_id]
        end_m_offset = start_m_offset + group_size

        # Assume following arguments:
        # - size_lhs_sublane & tile_m = 4
        # - group_size = 3
        # - start_m_offset = 7
        #
        # If we visualize it, it will look like this where:
        # - |: denotes boundaries between sublanes
        # - 0: denotes values for other groups
        # - 1: denotes values for the current group
        # | 0 0 0 0 | 0 0 0 1 | 1 1 0 0 |
        #
        # In this example, we see that we require processing 2 m tiles.
        # But, performing a naive cdiv(group_size, tile_m) will return 1.
        # Instead, adding local_offset will give us the correct value.
        local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
        aligned_group_size = group_size + local_offset
        curr_num_gm = pl.cdiv(aligned_group_size, cfgs.tiles.tile_m)

        # We need to handle cases where we should not process the group.
        # 1. Even if group_size is 0, if local_offset is not 0, cdiv will return 1.
        # 2. If group comes before the group_offset, we should not process it.
        should_process = jnp.logical_and(group_size > 0, group_id >= 0)
        curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
        next_num_gm = num_gm + curr_num_gm

        tm_loop_fn = functools.partial(
            inner_tm_loop,
            end_m_offset=end_m_offset,
            group_id=group_id,
        )
        lax.fori_loop(num_gm, next_num_gm, tm_loop_fn, start_m_offset)

        return next_num_gm, end_m_offset

    num_gm, _ = lax.fori_loop(0, max_num_group, outer_group_loop, (0, 0))
    return num_gm


@jax.named_scope("zero_out_start")
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
    left_zero_end = compute_start // dims.size_lhs_sublane
    left_zero_size = left_zero_end - left_zero_start
    left_num_loops = pl.cdiv(left_zero_size, row_size)

    right_zero_start = pl.cdiv(compute_end, dims.size_lhs_sublane)
    right_zero_end = out_dma.shape[0]
    right_zero_size = right_zero_end - right_zero_start
    right_num_loops = pl.cdiv(right_zero_size, row_size)

    def fill_zero(i, zero_size, *, start, end):
        dma_start = start + i * row_size
        dma_end = jnp.minimum(dma_start + row_size, end)
        dma_size = dma_end - dma_start

        # Static loop. Will be unrolled during compile time.
        for n_start in range(0, out_ref.shape[-1], num_lanes):
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


@jax.named_scope("zero_out_end")
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


@jax.named_scope("zero_out_start_3d")
def zero_out_start_3d(
        out_ref: jax.Array,  # [size_m, size_n // num_lanes, num_lanes]
        zero_src_ref: jax.Array,  # [tile_rows, size_n // num_lanes, num_lanes]
        semaphore_ref: jax.Array,  # [1]
):
    """Zero out ALL output rows via DMA. Required for DMA scatter where output
    positions are non-contiguous (scattered).

    Reuses an existing 3D VMEM scratch (e.g. scatter_staging_ref) as the zero
    source — no dedicated zero_ref allocation needed. The caller must ensure
    zero_src_ref is not in use when this function runs.
    """

    zero_src_ref[...] = jnp.zeros_like(zero_src_ref)
    row_size = zero_src_ref.shape[0]

    total_rows = out_ref.shape[0]
    num_loops = pl.cdiv(total_rows, row_size)

    def fill_zero(i, zero_size):
        dma_start = i * row_size
        dma_end = jnp.minimum(dma_start + row_size, total_rows)
        dma_size = dma_end - dma_start

        pltpu.make_async_copy(
            src_ref=zero_src_ref.at[pl.ds(0, dma_size)],
            dst_ref=out_ref.at[pl.ds(dma_start, dma_size)],
            sem=semaphore_ref.at[0],
        ).start(priority=1)

        return zero_size + dma_size

    zero_size = lax.fori_loop(0, num_loops, fill_zero, 0)
    return zero_size


@jax.named_scope("zero_out_end_3d")
def zero_out_end_3d(
    out_ref: jax.Array,  # [size_m, size_n // num_lanes, num_lanes]
    semaphore_ref: jax.Array,  # [1]
    zero_size: jax.Array,
):
    """Wait for all zero-fill DMAs to complete. Works with 3D output refs."""
    pltpu.make_async_copy(
        src_ref=out_ref.at[pl.ds(0, zero_size)],
        dst_ref=out_ref.at[pl.ds(0, zero_size)],
        sem=semaphore_ref.at[0],
    ).wait()


@jax.named_scope("dma_gather_gm_start")
def dma_gather_gm_start(src_ref,
                        dst_ref,
                        indices_ref,
                        sem_ref,
                        gm_id,
                        metadata_ref,
                        divisor: int = 1):
    """Start gathering rows for a specific gm tile via DMA.

    src_ref and dst_ref must be 3D: (rows, k // num_lanes, num_lanes).
    No reshape — reshape on refs breaks dynamic pl.ds offsets.

    `divisor`: optional integer divisor applied to indices_ref values before
    use. Set > 1 when indices_ref contains packed values (e.g.,
    `combined = lhs_idx * divisor + extra_field`) and we need to recover the
    actual src_row via integer division. Default 1 = no unpacking.
    """
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    sls = pltpu.get_tpu_info().get_sublane_tiling(src_ref.dtype)
    m_start_local = m_start % sls

    def _gather_body(i, _):
        row = m_start + i
        src_row = indices_ref[row]
        if divisor != 1:
            src_row = src_row // divisor
        pltpu.make_async_copy(
            src_ref=src_ref.at[pl.ds(src_row, 1), :, :],
            dst_ref=dst_ref.at[pl.ds(m_start_local + i, 1), :, :],
            sem=sem_ref,
        ).start()
        return _

    lax.fori_loop(0, m_end - m_start, _gather_body, 0)


@jax.named_scope("dma_gather_gm_wait")
def dma_gather_gm_wait(dst_ref, sem_ref, gm_id, metadata_ref):
    """Wait for all gather DMAs for a specific gm tile to complete.

    dst_ref must be 3D: (rows, k // num_lanes, num_lanes).
    """
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    num_rows = m_end - m_start
    sls = pltpu.get_tpu_info().get_sublane_tiling(dst_ref.dtype)
    m_start_local = m_start % sls
    pltpu.make_async_copy(
        src_ref=dst_ref.at[pl.ds(m_start_local, num_rows), :, :],
        dst_ref=dst_ref.at[pl.ds(m_start_local, num_rows), :, :],
        sem=sem_ref,
    ).wait()


@jax.named_scope("dma_scatter_gm_start")
def dma_scatter_gm_start(src_ref, dst_ref, indices_ref, sem_ref, gm_id,
                         metadata_ref):
    """Start scattering rows for a specific gm tile via DMA.

    src_ref and dst_ref must be 3D: (rows, n // num_lanes, num_lanes).
    No reshape — reshape on refs breaks dynamic pl.ds offsets.
    """
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    sls = pltpu.get_tpu_info().get_sublane_tiling(src_ref.dtype)
    m_start_local = m_start % sls

    def _scatter_body(i, _):
        row = m_start + i
        dst_row = indices_ref[row]
        pltpu.make_async_copy(
            src_ref=src_ref.at[pl.ds(m_start_local + i, 1), :, :],
            dst_ref=dst_ref.at[pl.ds(dst_row, 1), :, :],
            sem=sem_ref,
        ).start()
        return _

    lax.fori_loop(0, m_end - m_start, _scatter_body, 0)


@jax.named_scope("dma_scatter_gm_wait")
def dma_scatter_gm_wait(src_ref, sem_ref, gm_id, metadata_ref):
    """Wait for all scatter DMAs for a specific gm tile to complete.

    src_ref must be 3D: (rows, n // num_lanes, num_lanes).
    """
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    num_rows = m_end - m_start
    sls = pltpu.get_tpu_info().get_sublane_tiling(src_ref.dtype)
    m_start_local = m_start % sls
    pltpu.make_async_copy(
        src_ref=src_ref.at[pl.ds(m_start_local, num_rows), :, :],
        dst_ref=src_ref.at[pl.ds(m_start_local, num_rows), :, :],
        sem=sem_ref,
    ).wait()


def calculate_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
) -> TileSizes:
    """Calculate optimal tile sizes for GMM kernel."""

    lhs_dtype = lhs_cfgs.quant_dtype or lhs_cfgs.dtype
    rhs_dtype = rhs_cfgs.dtype
    lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

    # When using bf16 for lhs and rhs, 128 is the largest tile_m value that is
    # safe to use for most scenarios. But if lower bitwidth is used, we need
    # to tweak tile_m to account for using faster hardware unit.
    # TODO(kyuyeunk): Account for different TPU hardware specs.
    bf16_bf16_tile_m = 128
    lhs_mod = min(pl.cdiv(16, lhs_bits), 2)
    rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
    tile_m = bf16_bf16_tile_m * lhs_mod // rhs_mod
    tile_m = min(tile_m, dims.size_m)

    # Subtract non-rhs VMEM overhead before computing per-buffer budget.
    # Overhead includes: gathered_lhs_2x (DMA gather), acc/partial_out buffers,
    # tiled_out_2x (DMA scatter), and compiler spill headroom.
    lhs_bits_item = jax.dtypes.itemsize_bits(lhs_cfgs.dtype)
    _overhead = (
        2 * tile_m * dims.size_k * lhs_bits_item // 8  # gathered_lhs_2x
        + 5 * 1024 * 1024  # acc, partial_out, spill headroom
    )
    rhs_vmem_budget = max(vmem_limit_bytes - _overhead, vmem_limit_bytes // 2)

    # Calculate vmem limit for a single rhs buffer when using triple buffers.
    num_rhs_buffers = 3
    rhs_vmem_target = rhs_vmem_budget // num_rhs_buffers
    base_rhs_size_bytes = dims.size_k * dims.size_n * rhs_bits // 8

    # To avoid stalling MXU, we add some buffer room where tile_n cannot go
    # smaller than 2x of mxu_column_size.
    tile_n_limit = pltpu.get_tpu_info().mxu_column_size * 2
    tile_n_limit = min(tile_n_limit, dims.size_n)

    # When fuse_act is set, tile_n tiles the output N (= size_n // 2).
    # base_rhs_size_bytes still uses size_n to account for loading both halves.
    size_n_per_rhs = dims.size_n
    if fuse_act is not None:
        size_n_per_rhs //= 2
        tile_n_limit = min(tile_n_limit, size_n_per_rhs)

    # Initialize tile_k and tile_n to their maximum valid values.
    num_k_tiles = num_n_tiles = 1
    num_lanes = pltpu.get_tpu_info().num_lanes
    tile_k = align_to(dims.size_k, num_lanes)
    tile_n = align_to(size_n_per_rhs, num_lanes)

    # Multiple k tiles will introduce accumulation overhead. Thus, we first try
    # to fit rhs into vmem by only adjusting tile_n.

    # Decrease tile_n until rhs fits in vmem target.
    while (pl.cdiv(base_rhs_size_bytes, num_n_tiles) > rhs_vmem_target
           and tile_n > tile_n_limit):
        num_n_tiles += 1
        tile_n = align_to(size_n_per_rhs,
                          num_n_tiles * num_lanes) // num_n_tiles

    # If decreasing tile_n is no longer possible, we decrease tile_k instead.
    if tile_n < tile_n_limit:
        num_n_tiles -= 1
        tile_n = align_to(size_n_per_rhs,
                          num_n_tiles * num_lanes) // num_n_tiles

        # Decrease tile_k until rhs fits in vmem target.
        base_rhs_size_bytes = pl.cdiv(base_rhs_size_bytes, num_n_tiles)
        while pl.cdiv(base_rhs_size_bytes, num_k_tiles) > rhs_vmem_target:
            num_k_tiles += 1
            tile_k = align_to(dims.size_k,
                              num_k_tiles * num_lanes) // num_k_tiles

    if tile_n == 0 or tile_k == 0:
        raise ValueError(
            f"Could not find valid tile sizes for {dims=} and {rhs_vmem_target=}."
        )

    return TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)


def validate_inputs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    fuse_act: str | None = None,
    packed_nvfp4: bool = False,
):
    """Validates the inputs for the GMM kernel."""

    size_m = lhs.shape[0]
    size_group, size_k_raw, size_n = rhs.shape
    size_k = size_k_raw * 2 if packed_nvfp4 else size_k_raw
    size_lhs_group = group_sizes.shape[0]

    assert size_group <= size_lhs_group
    assert lhs.shape == (
        size_m,
        size_k,
    ), f"lhs.shape={lhs.shape} expected=({size_m}, {size_k}), rhs.shape={rhs.shape}, group_sizes.shape={group_sizes.shape}"
    assert rhs.shape == (size_group, size_k_raw, size_n)
    if rhs_bias is not None:
        assert rhs_bias.shape == (size_group, 1, size_n)
    if rhs_scale is not None:
        num_quant_blocks = rhs_scale.shape[1]
        assert rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n)
        # When K is zero-padded for DMA alignment, size_k may not divide
        # num_quant_blocks evenly.  The original (unpadded) K always does.
        # The inner kernel uses quant_block_size from make_gmm_configs which
        # is derived from the original K, so this is safe.

    assert group_offset.shape == (1, )

    size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype)
    size_lhs_sublane = min(size_lhs_sublane, size_m)

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

    flops = 2 * dims.size_m * dims.size_k * dims.size_n

    lhs_bytes = dims.size_m * dims.size_k * lhs_dtype.itemsize

    rhs_bytes = (dims.size_group * dims.size_k * dims.size_n *
                 jax.dtypes.itemsize_bits(rhs_dtype)) // 8
    if cfgs.rhs_cfgs.has_scale:
        rhs_bytes += (dims.size_group * cfgs.rhs_cfgs.num_quant_blocks *
                      dims.size_n * jnp.dtype(jnp.float32).itemsize)
    if cfgs.rhs_cfgs.has_bias:
        rhs_bytes += dims.size_group * dims.size_n * jnp.dtype(
            jnp.float32).itemsize

    out_bytes = dims.size_m * dims.size_n * jnp.dtype(cfgs.out_dtype).itemsize

    total_bytes = lhs_bytes + rhs_bytes + out_bytes

    return pl.CostEstimate(
        flops=flops,
        bytes_accessed=total_bytes,
        transcendentals=0,
    )


def get_scope_name(dims: Dimensions, tiles: TileSizes) -> str:
    return (
        f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}"
        f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
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
    lhs_indices: jax.Array | None = None,
    output_indices: jax.Array | None = None,
    original_k: int | None = None,
    fuse_act: str | None = None,
    post_expert_norm_weight: jax.Array | None = None,
    original_size_m: int | None = None,
    original_size_k: int | None = None,
    original_size_n: int | None = None,
    packed_nvfp4: bool = False,
):
    """Fills the GMM config for the GMM kernel."""

    dims = validate_inputs(
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        group_sizes,
        group_offset,
        fuse_act=fuse_act,
        packed_nvfp4=packed_nvfp4,
    )

    if rhs_scale is not None:
        has_scale = True
        rhs_quant_dtype = jnp.float4_e2m1fn.dtype if packed_nvfp4 else rhs.dtype
        num_blocks = rhs_scale.shape[1]
        # When K was zero-padded for DMA alignment, use the original K
        # to compute block_size so quant block boundaries stay correct.
        _k_for_blocks = original_k if original_k is not None else dims.size_k
        block_size = _recover_quant_block_size(_k_for_blocks, num_blocks)
        rhs_packing = 8 if packed_nvfp4 else 32 // jax.dtypes.itemsize_bits(
            rhs.dtype)
    else:
        has_scale = False
        rhs_quant_dtype = None
        num_blocks = 1
        block_size = dims.size_k
        rhs_packing = 1

    rhs_cfgs = InputConfigs(
        quant_dtype=rhs_quant_dtype,
        quant_block_size=block_size,
        dtype=rhs.dtype,
        has_bias=rhs_bias is not None,
        has_scale=has_scale,
        packing=rhs_packing,
        num_quant_blocks=num_blocks,
    )

    lhs_quant_block_size = 256 if rhs_cfgs.quant_block_size < 512 else 512

    lhs_q_dtype = None
    if (maybe_quantize_lhs and get_maybe_quantize_lhs(
            rhs_quant_dtype,
            rhs_cfgs.quant_block_size,
            lhs_quant_block_size,
    ) and rhs_quant_dtype is not None):
        # Choose lhs quantization dtype based on TPU hardware support.
        is_rhs_float = jnp.issubdtype(rhs_quant_dtype, jnp.floating)
        tpu_info = pltpu.get_tpu_info()
        # Check if there is hardware compute support for rhs dtype group.
        if is_rhs_float:
            if tpu_info.fp8_ops_per_second > 0:
                lhs_q_dtype = jnp.float8_e4m3fn.dtype
        else:
            if tpu_info.int8_ops_per_second > 0:
                lhs_q_dtype = jnp.int8.dtype

    lhs_cfgs = InputConfigs(
        quant_dtype=lhs_q_dtype,
        # Input quantization involves reading all elements in a block to compute
        # scale value. Since this operation is very memory intensive, we use a
        # block size that is small enough to minimize memory overhead but large
        # enough to minimize compute overhead of quantization.
        quant_block_size=lhs_quant_block_size,
        dtype=lhs.dtype,
    )

    if out_dtype is None:
        out_dtype = lhs.dtype

    if acc_dtype is None:
        acc_dtype = acc_dtype = jnp.float32.dtype

    if isinstance(tile_info, TileSizes):
        tiles = tile_info
    else:
        tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)
        default_block_sizes = (tiles.tile_m, tiles.tile_k, tiles.tile_n)
        # Use original sizes (before padding) if provided, to keep lookup keys consistent
        # across different kernels
        tile_m, tile_k, tile_n = get_tuned_block_sizes(
            dims.size_m if original_size_m is None else original_size_m,
            dims.size_k if original_size_k is None else original_size_k,
            dims.size_n if original_size_n is None else original_size_n,
            dims.size_group,
            lhs_cfgs.dtype,
            rhs_cfgs.dtype,
            maybe_quantize_lhs,
            rhs_cfgs.quant_block_size,
            default_block_sizes,
            lhs_indices,
            output_indices,
            fuse_act,
        )
        tiles = TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)

    return GmmConfigs(
        dims=dims,
        tiles=tiles,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        out_dtype=out_dtype,
        acc_dtype=acc_dtype,
        zero_init=zero_initialize,
        fuse_act=fuse_act,
        has_post_norm=post_expert_norm_weight is not None,
    )


# =============================================================================
# Fused kernel: gather + GMM1 + activation + GMM2 + scatter
# =============================================================================
# Time-shares the VMEM weight buffer between GMM1 and GMM2, keeping the
# intermediate activation result in VMEM to avoid an HBM round-trip.
# Prototype: bf16 only, synchronous weight reads (no pipelining).
# =============================================================================
@dataclasses.dataclass(frozen=True)
class FusedDims:
    """Dimensions for the fused gather+GMM1+act+GMM2+scatter kernel."""

    size_m: int  # Number of gathered/scattered rows
    size_k1: int  # K for GMM1 (= hidden_size H, original unpadded)
    size_n1: int  # N for GMM1 (= 2 * intermediate_size, gate+up)
    size_k2: int  # K for GMM2 (= intermediate_size I)
    size_n2: int  # N for GMM2 (= hidden_size H), aligned for DMA
    original_n2: int  # Original (unpadded) N for GMM2
    size_group: int  # Number of experts on this shard
    size_lhs_group: int  # Total number of expert groups
    size_lhs_sublane: int  # Sublane tiling for LHS dtype
    intermediate_size: int = 0  # Original I (= N1_orig // 2) for activation split
    has_bias: bool = False  # Whether bias refs are present
    quant_block_size: int | None = None  # RHS quantization block size
    num_scale_blocks1: int = 0  # K1 // quant_block_size (for scale DMA)
    num_scale_blocks2: int = 0  # K2 // quant_block_size (for scale DMA)
    padded_k1: int = 0  # DMA-aligned K1 (>= size_k1); 0 means no padding
