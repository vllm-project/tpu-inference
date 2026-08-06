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
"""Fused MoE FFN kernel: GMM1 + activation + GMM2 in one Pallas pipeline.

Bitwise-equal to gmm_v2(act(gmm_v2(lhs, w1, fuse_act)), w2), with the
intermediate held in VMEM. Both matmuls run at num_k = num_n = 1.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.megablox.gmm_v2_fused_support import (
    LHS_QUANT_BLOCK_SIZE, FusedWeightsRef, GmmConfigs, MetadataRef, RhsRef,
    TileSizes, WeightsRef, align_to, apply_act_fn, compute_local_row_bounds,
    fill_metadata, generate_block_specs, get_cost_estimate, get_metadata,
    make_gmm_configs, mask_out_of_group_rows, matmul_tile, store_output_tile,
    unsigned_floor_div, zero_out_end, zero_out_start)
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Mirrors the Buffered(...) counts in generate_block_specs (lhs/out
# double-buffered, weights triple-buffered) so the VMEM estimate stays in sync.
LHS_BUFFER_COUNT = 2
WEIGHT_BUFFER_COUNT = 3
OUT_BUFFER_COUNT = 2
# The bias block spec carries no pipeline_mode, so it takes the pipeline's
# default double buffering rather than the weights' triple.
BIAS_BUFFER_COUNT = 2

PARTIAL_OUT_SUBLANES = 1  # partial_out rows = PARTIAL_OUT_SUBLANES * sublane
ZERO_REF_TARGET_BYTES = 2 * 1024 * 1024

# Each arm holds a full copy of the fused body, so the arm count is
# instruction memory paid at every grid step's branch target.
MAX_BUCKET_ARMS = 4

# Row-count bucket base used by the MoE layer. Matches the fp8 lhs sublane
# tiling, so every arm is a whole number of sublanes.
GMM_FUSED_BUCKET_BASE = 32


def unsupported_reason(lhs, w1, w2, w1_scale, w2_scale, w1_bias,
                       w2_bias) -> str | None:
    """Why gmm_fused cannot stand in for the two-gmm_v2 pair, or None."""
    if w1_scale is None or w2_scale is None:
        return ("the fused kernel only has the quantized (postscale) "
                "matmul path and the weights carry no scales")
    # Answer rather than raise on a malformed operand: the caller branches
    # on this string and a raise here is a boot failure, not a fallback.
    operands = [("lhs", lhs, 2), ("w1", w1, 3), ("w2", w2, 3),
                ("w1_scale", w1_scale, 4), ("w2_scale", w2_scale, 4)]
    if w1_bias is not None:
        operands.append(("w1_bias", w1_bias, 3))
    if w2_bias is not None:
        operands.append(("w2_bias", w2_bias, 3))
    for name, operand, rank in operands:
        shape = getattr(operand, "shape", None)
        if shape is None:
            return (f"{name} is {type(operand).__name__} and carries no "
                    "shape the fused kernel can read")
        if len(shape) != rank:
            return (f"{name} has rank {len(shape)}; the fused kernel takes "
                    f"a rank-{rank} operand there")
    size_m, hidden = lhs.shape
    if size_m == 0:
        return "the fused kernel needs at least one row and the lhs is empty"
    if w1.shape[1] != hidden:
        return (f"w1 contracts over {w1.shape[1]} but the lhs is {hidden} "
                "wide; the fused kernel takes unpadded-hidden weights only")
    if w1.shape[2] % 2 != 0:
        return (f"w1 output width {w1.shape[2]} is odd and cannot split "
                "into a gate and an up half")
    inter = w1.shape[2] // 2
    if w2.shape != (w1.shape[0], inter, hidden):
        return (f"w2 shape {w2.shape} != {(w1.shape[0], inter, hidden)}; the "
                "fused kernel keeps the intermediate in VMEM and cannot trim "
                "a padded one between the matmuls")
    # Bias layout, one row per expert, matching gmm_v2's own contract.
    for name, bias, size_n in (("w1_bias", w1_bias, 2 * inter),
                               ("w2_bias", w2_bias, hidden)):
        if bias is not None and bias.shape != (w1.shape[0], 1, size_n):
            return (f"{name} layout {bias.shape} is not "
                    f"[groups, 1, {size_n}]")
    # Scale layout. The shape-only conditions come first so that a host
    # with no device record still gets the real reason when there is one.
    scale_blocks = []
    for name, scale, size_k, size_n in (("w1", w1_scale, hidden, 2 * inter),
                                        ("w2", w2_scale, inter, hidden)):
        blocks = scale.shape[1]
        if scale.shape != (w1.shape[0], blocks, 1, size_n) or blocks == 0:
            return (f"{name}_scale layout {scale.shape} is not "
                    f"[groups, blocks, 1, {size_n}]")
        if size_k % blocks != 0:
            return (f"{name}_scale block count {blocks} does not divide the "
                    f"contraction {size_k}")
        scale_blocks.append((name, size_k // blocks, blocks))
    # The postscale path needs an rhs scale block at least as wide as the
    # MXU column, which is the chip's own number.
    try:
        mxu_size = pltpu.get_tpu_info().mxu_column_size
    except Exception as e:  # no device record to read the geometry from
        return (f"the fused kernel's device record cannot be read here "
                f"({e}); its MXU column and VMEM budget are the chip's own "
                "numbers")
    for name, block, blocks in scale_blocks:
        if block < mxu_size:
            return (f"{name}_scale block {block} is narrower than the MXU "
                    f"column ({mxu_size}), which takes the pre-matmul "
                    "dequant path the fused kernel does not carry")
        if blocks > 1 and block % LHS_QUANT_BLOCK_SIZE != 0:
            return (f"{name}_scale block {block} is not a multiple of the "
                    f"lhs quantization block ({LHS_QUANT_BLOCK_SIZE}); the "
                    "k loop would skip scale blocks")
    # Everything the builder itself refuses: VMEM fit at every legal tile
    # height, and a row count that is a whole number of sublanes.
    try:
        build_stage_configs(lhs,
                            w1,
                            w2,
                            w1_scale,
                            w2_scale,
                            jax.ShapeDtypeStruct((w1.shape[0], ), jnp.int32),
                            jax.ShapeDtypeStruct((1, ), jnp.int32),
                            w1_bias=w1_bias,
                            w2_bias=w2_bias)
    except Exception as e:
        return f"the fused kernel cannot build these shapes: {e}"
    return None


def fused_inner_kernel(
        # In
        tiled_lhs_ref: RhsRef,
        # weight: [tile_m // sublane, sublane, hidden]
        # scale (optional): [tile_m // sublane, sublane, hidden // 512]
        w1_ref: RhsRef,  # FusedWeightsRef: gate/up [hidden, inter] (+ scales)
        w2_ref: RhsRef,  # WeightsRef: [inter, hidden] (+ scale)
        # Out
    tiled_out_ref: jax.Array,  # [tile_m // sublane, sublane, hidden]
        # Scratch
    partial_out_ref: jax.Array,  # [sublane, hidden]
        metadata_ref: MetadataRef,
        *,
        cfgs1: GmmConfigs,  # GMM1 (gate/up) config, fuse_act set
        cfgs2: GmmConfigs,  # GMM2 (down) config, fuse_act None
        bucket_base: int | None = None,  # row-count bucket granularity
):
    """Per-gm-tile fused FFN body: GMM1 -> act -> bridge -> GMM2 -> store."""
    sublane = cfgs1.dims.size_lhs_sublane
    tile_m = cfgs1.tiles.tile_m
    gm_id = pl.program_id(1)
    m_start_local, m_end_local = compute_local_row_bounds(
        metadata_ref, gm_id, sublane)

    # Every op after the accumulator is row-wise, so a row this tile retains
    # is bitwise equal to what the sequential gmm_v2 pair computes for it.
    def body(rows: int):
        """The per-tile dataflow over the leading `rows` rows of the tile."""
        lhs_2d = tiled_lhs_ref.weight.reshape(-1, cfgs1.tiles.tile_k)
        tiled_lhs = lhs_2d[...] if rows == tile_m else lhs_2d[:rows]
        acc1 = matmul_tile(tiled_lhs, w1_ref, cfgs=cfgs1, is_last_k_step=True)

        # The pair adds the expert bias on the last contraction step and
        # before the activation; both matmuls here are single-step, so this
        # is that step. get_bias concatenates the gate and up halves in the
        # same order get_weight does.
        if cfgs1.rhs_cfgs.has_bias:
            acc1 += w1_ref.get_bias().astype(acc1.dtype)

        act = apply_act_fn(acc1, cfgs1.fuse_act)

        # Mask rows this tile's group does not own before the bridge: their
        # act values are another expert's, and GMM2 must consume 0 there.
        mid = mask_out_of_group_rows(act, m_start_local, m_end_local)

        # Cast to the dtype the pair materializes in HBM here BEFORE the
        # requant, or GMM2 quantizes values the pair never saw.
        mid = mid.astype(cfgs1.out_dtype)

        acc2 = matmul_tile(mid, w2_ref, cfgs=cfgs2, is_last_k_step=True)

        # Same placement as GMM1: the bias lands on the accumulator before
        # the mask, so rows this tile does not own still leave as zeros.
        if cfgs2.rhs_cfgs.has_bias:
            acc2 += w2_ref.get_bias().astype(acc2.dtype)

        acc2_masked = mask_out_of_group_rows(acc2, m_start_local,
                                             m_end_local).reshape(
                                                 rows // sublane, sublane,
                                                 tiled_out_ref.shape[2])
        store_output_tile(acc2_masked,
                          tiled_out_ref,
                          partial_out_ref,
                          gm_id,
                          m_end_local,
                          sublane=sublane)

    if bucket_base is None:
        body(tile_m)
        return

    # The index reaches num_arms on a completely owned tile; lax.switch
    # clamps it onto the full-tile arm, which is the correct arm there.
    branches = [
        functools.partial(body, bucket_base * (i + 1))
        for i in range(tile_m // bucket_base)
    ]
    lax.switch(unsigned_floor_div(m_end_local, bucket_base), branches)


def _run_fused_pipeline(
    num_gm: jax.Array,
    metadata_ref: MetadataRef,
    lhs_ref: WeightsRef,
    w1_ref: WeightsRef,
    w2_ref: WeightsRef,
    out_ref: jax.Array,
    partial_out_ref: jax.Array,
    zero_ref: jax.Array | None,
    semaphore_ref: jax.Array | None,
    *,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
    bucket_base: int | None = None,
):
    """Zero-init, block specs and the fused software pipeline."""
    sublane = cfgs1.dims.size_lhs_sublane

    if cfgs2.zero_init:
        zero_size = zero_out_start(
            out_ref,
            zero_ref,
            semaphore_ref,
            metadata_ref,
            num_gm,
            dims=cfgs2.dims,
        )

    # Sub-byte weight packing: inert for fp8, used by the four-bit preset.
    if cfgs1.rhs_cfgs.should_bitcast:
        w1_ref = dataclasses.replace(w1_ref,
                                     weight=w1_ref.weight.bitcast(jnp.uint32))
    if cfgs2.rhs_cfgs.should_bitcast:
        w2_ref = dataclasses.replace(w2_ref,
                                     weight=w2_ref.weight.bitcast(jnp.uint32))

    (lhs_spec, w1_spec), _ = generate_block_specs(metadata_ref, cfgs1)
    (_, w2_spec), out_spec = generate_block_specs(metadata_ref, cfgs2)

    # w1 is [g, hidden, 2 * inter]: gate = [..., :inter], up = [..., inter:].
    w1_up_ref = jax.tree.map(lambda x: x.at[..., cfgs1.out_size_n:], w1_ref)
    w1_ref = FusedWeightsRef(gate=w1_ref, up=w1_up_ref)
    w1_spec = FusedWeightsRef(gate=w1_spec, up=w1_spec)

    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(fused_inner_kernel,
                          cfgs1=cfgs1,
                          cfgs2=cfgs2,
                          bucket_base=bucket_base),
        grid=(1, num_gm, 1),
        in_specs=(lhs_spec, w1_spec, w2_spec),
        out_specs=out_spec,
    )

    # Bounded slice requires second last dim to be aligned to the sublane
    # size. Weight refs use static tiling thus reshape is not needed.
    lhs_weight_ref = lhs_ref.weight
    lhs_weight_in = lhs_weight_ref.reshape(-1, sublane,
                                           lhs_weight_ref.shape[-1])
    lhs_in = WeightsRef(weight=lhs_weight_in, scale=None, bias=None)
    out_in = out_ref.reshape(-1, sublane, out_ref.shape[-1])
    scratches = [partial_out_ref, metadata_ref]
    pipeline_fn(lhs_in, w1_ref, w2_ref, out_in, scratches=scratches)

    if cfgs2.zero_init:
        zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs2.dims)


def fused_kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: WeightsRef,  # weight: [size_m, hidden],
    # scale (optional): [size_m, hidden // 512] f32
    w1_ref: WeightsRef,  # [size_group, hidden, 2 * inter] (+ scale)
    w2_ref: WeightsRef,  # [size_group, inter, hidden] (+ scale)
    # Out
    out_ref: jax.Array,  # [size_m, hidden]
    # Scratch memory
    partial_out_ref: jax.Array,  # [sublane, hidden]
    metadata_ref: MetadataRef,
    zero_ref: jax.Array | None,
    semaphore_ref: jax.Array | None,
    *,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
    bucket_base: int | None = None,
):
    """Entry point for the fused FFN kernel (in-kernel metadata scan)."""
    num_gm = fill_metadata(
        lhs_group_sizes_ref,
        group_offset_ref,
        metadata_ref,
        cfgs=cfgs2,
    )
    _run_fused_pipeline(num_gm,
                        metadata_ref,
                        lhs_ref,
                        w1_ref,
                        w2_ref,
                        out_ref,
                        partial_out_ref,
                        zero_ref,
                        semaphore_ref,
                        cfgs1=cfgs1,
                        cfgs2=cfgs2,
                        bucket_base=bucket_base)


def _derive_bucket_base(tile_m: int, sublane: int, requested: int) -> int:
    """Row-count bucket base a realized tile height can actually host."""
    ladder = [
        tile_m // arms for arms in range(MAX_BUCKET_ARMS, 0, -1)
        if tile_m % arms == 0 and (tile_m // arms) % sublane == 0
    ]
    base = next((b for b in ladder if b >= requested), tile_m)
    # A rung equal to tile_m is bucketing off, which is what the caller asked
    # to avoid, so take the finest rung instead whenever one exists.
    if base == tile_m:
        base = ladder[0]
    return base


def _default_tile_m(size_m: int) -> int:
    """Default gm-tile rows; matches calculate_tiling for fp8 x fp8."""
    return min(128, size_m)


def fused_vmem_estimate(cfgs1: GmmConfigs, cfgs2: GmmConfigs) -> int:
    """Approximate VMEM footprint (bytes) of the fused pipeline."""
    t1, t2 = cfgs1.tiles, cfgs2.tiles
    lhs_bytes = jax.dtypes.itemsize_bits(cfgs1.lhs_cfgs.dtype) // 8
    out_bytes = jnp.dtype(cfgs2.out_dtype).itemsize
    acc_bytes = jnp.dtype(cfgs1.acc_dtype).itemsize
    sublane = cfgs1.dims.size_lhs_sublane

    lhs_vmem = LHS_BUFFER_COUNT * t1.tile_m * t1.tile_k * lhs_bytes
    # w1: gate + up weight tiles + channelwise f32 scales.
    w1_tile_bits = t1.tile_k * t1.tile_n * jax.dtypes.itemsize_bits(
        cfgs1.rhs_cfgs.dtype)
    w1_scale_bytes = cfgs1.num_quant_blocks_per_tile_k * t1.tile_n * 4
    w1_vmem = WEIGHT_BUFFER_COUNT * 2 * (w1_tile_bits // 8 + w1_scale_bytes)
    if cfgs1.rhs_cfgs.has_bias:
        w1_vmem += BIAS_BUFFER_COUNT * 2 * t1.tile_n * 4
    w2_tile_bits = t2.tile_k * t2.tile_n * jax.dtypes.itemsize_bits(
        cfgs2.rhs_cfgs.dtype)
    w2_scale_bytes = cfgs2.num_quant_blocks_per_tile_k * t2.tile_n * 4
    w2_vmem = WEIGHT_BUFFER_COUNT * (w2_tile_bits // 8 + w2_scale_bytes)
    if cfgs2.rhs_cfgs.has_bias:
        w2_vmem += BIAS_BUFFER_COUNT * t2.tile_n * 4
    out_vmem = OUT_BUFFER_COUNT * t1.tile_m * t2.tile_n * out_bytes
    # Live intermediates (acc1, act/mid, acc2); num_k == 1 by contract, so no
    # accumulator scratch is allocated.
    live_vmem = t1.tile_m * (2 * t1.tile_n + t1.tile_n + t2.tile_n) * acc_bytes
    partial_vmem = (PARTIAL_OUT_SUBLANES * sublane * t2.tile_n * out_bytes)
    zero_vmem = ZERO_REF_TARGET_BYTES if cfgs2.zero_init else 0
    return (lhs_vmem + w1_vmem + w2_vmem + out_vmem + live_vmem +
            partial_vmem + zero_vmem)


def default_vmem_limit_bytes() -> int:
    """The kernel's own VMEM budget: 90% of this generation's capacity."""
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)


def build_stage_configs(
    lhs,
    w1,
    w2,
    w1_scale,
    w2_scale,
    group_sizes,
    group_offset,
    *,
    w1_bias=None,
    w2_bias=None,
    fuse_act: str = "silu",
    tile_m: int | None = None,
    vmem_limit_bytes: int | None = None,
    acc_dtype: jnp.dtype | None = None,
    zero_initialize: bool = True,
    unconditional_pipeline: bool = True,
):
    """Both stage configs -> (cfgs1, cfgs2, tiles1, tiles2, tile_m)."""
    if vmem_limit_bytes is None:
        vmem_limit_bytes = default_vmem_limit_bytes()
    size_m, hidden = lhs.shape
    inter = w1.shape[2] // 2
    tile_m_arg = tile_m
    tile_m = tile_m_arg if tile_m_arg is not None else _default_tile_m(size_m)
    mid_proxy = jax.ShapeDtypeStruct((size_m, inter), lhs.dtype)
    while True:
        tiles1 = TileSizes(tile_m=tile_m, tile_k=hidden, tile_n=inter)
        tiles2 = TileSizes(tile_m=tile_m, tile_k=inter, tile_n=hidden)
        cfgs1 = make_gmm_configs(
            lhs,
            w1,
            w1_scale,
            w1_bias,
            group_sizes,
            group_offset,
            tile_info=tiles1,
            vmem_limit_bytes=vmem_limit_bytes,
            out_dtype=None,  # bridge dtype, like the sequential mid
            acc_dtype=acc_dtype,
            maybe_quantize_lhs=True,
            zero_initialize=False,  # no HBM intermediate to zero
            fuse_act=fuse_act,
            unconditional_pipeline=unconditional_pipeline,
        )
        cfgs2 = make_gmm_configs(
            mid_proxy,
            w2,
            w2_scale,
            w2_bias,
            group_sizes,
            group_offset,
            tile_info=tiles2,
            vmem_limit_bytes=vmem_limit_bytes,
            out_dtype=None,
            acc_dtype=acc_dtype,
            maybe_quantize_lhs=True,
            zero_initialize=zero_initialize,
            fuse_act=None,
            unconditional_pipeline=unconditional_pipeline,
        )
        if fused_vmem_estimate(cfgs1, cfgs2) <= vmem_limit_bytes:
            break
        sublane = cfgs2.dims.size_lhs_sublane
        if tile_m_arg is not None or tile_m <= sublane:
            raise ValueError(
                f"fused kernel does not fit VMEM: {tile_m=} needs "
                f"{fused_vmem_estimate(cfgs1, cfgs2)} bytes "
                f"(limit {vmem_limit_bytes})")
        tile_m = max(tile_m // 2, sublane)

    dims1, dims2 = cfgs1.dims, cfgs2.dims

    # Single-tile contract: each stage's k and n axes must fit ONE tile, since
    # there is no accumulator scratch for a partial k sum and no second n step.
    for stage, size_k, out_n, tiles in ((1, dims1.size_k, cfgs1.out_size_n,
                                         tiles1), (2, dims2.size_k,
                                                   cfgs2.out_size_n, tiles2)):
        if pl.cdiv(size_k, tiles.tile_k) != 1:
            raise ValueError(
                f"GMM{stage} contraction {size_k} must fit one k tile "
                f"(tile_k={tiles.tile_k})")
        if pl.cdiv(out_n, tiles.tile_n) != 1:
            raise ValueError(
                f"GMM{stage} output width {out_n} must fit one n tile "
                f"(tile_n={tiles.tile_n})")

    if tile_m % dims1.size_lhs_sublane != 0:
        raise ValueError(f"{tile_m=} must be a multiple of the sublane size "
                         f"({dims1.size_lhs_sublane})")
    if cfgs1.lhs_cfgs.quant_dtype is None or cfgs2.lhs_cfgs.quant_dtype is None:
        raise NotImplementedError(
            "gmm_fused requires the quantized (postscale) matmul path; "
            "got an unquantized config , check weight dtype/scales and "
            "hardware fp8/int8 support")
    return cfgs1, cfgs2, tiles1, tiles2, tile_m


def get_fused_cost_estimate(cfgs1: GmmConfigs,
                            cfgs2: GmmConfigs) -> pl.CostEstimate:
    """Cost of both matmuls minus the fused-away intermediate HBM trips."""
    c1 = get_cost_estimate(cfgs1)
    c2 = get_cost_estimate(cfgs2)
    mid_out_bytes = (cfgs1.dims.size_m * cfgs1.out_size_n *
                     jnp.dtype(cfgs1.out_dtype).itemsize)
    c2_lhs_dtype = cfgs2.lhs_cfgs.quant_dtype or cfgs2.lhs_cfgs.dtype
    mid_in_bytes = (cfgs2.dims.size_m * cfgs2.dims.size_k *
                    jnp.dtype(c2_lhs_dtype).itemsize)
    return pl.CostEstimate(
        flops=c1.flops + c2.flops,
        bytes_accessed=(c1.bytes_accessed + c2.bytes_accessed - mid_out_bytes -
                        mid_in_bytes),
        transcendentals=0,
    )


def get_fused_scope_name(cfgs1: GmmConfigs,
                         cfgs2: GmmConfigs,
                         bucket_base: int | None = None) -> str:
    dims1, dims2 = cfgs1.dims, cfgs2.dims
    # Each suffix appears only when its feature is on, so programs without
    # it keep the name every compiled-program fingerprint was taken against.
    suffix = "" if bucket_base is None else f"-bb_{bucket_base}"
    if cfgs1.rhs_cfgs.has_bias or cfgs2.rhs_cfgs.has_bias:
        suffix += "-bias"
    return (f"gmm_fused-g_{dims1.size_group}-m_{dims1.size_m}"
            f"-h_{dims1.size_k}-i_{dims2.size_k}-act_{cfgs1.fuse_act}"
            f"-tm_{cfgs1.tiles.tile_m}{suffix}")


def get_fused_metadata(cfgs1: GmmConfigs, cfgs2: GmmConfigs):
    ret = {f"gmm1.{k}": v for k, v in get_metadata(cfgs1).items()}
    ret.update({f"gmm2.{k}": v for k, v in get_metadata(cfgs2).items()})
    return ret


@jax.jit(static_argnames=[
    "fuse_act",
    "tile_m",
    "bucket_base",
    "vmem_limit_bytes",
    "acc_dtype",
    "zero_initialize",
    "unconditional_pipeline",
])
def gmm_fused(
    lhs: jax.Array,  # [size_m, hidden] bf16
    w1: jax.Array,  # [size_group, hidden, 2 * inter] (gate|up fused)
    w2: jax.Array,  # [size_group, inter, hidden]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    w1_scale: jax.Array,  # [size_group, 1, 1, 2 * inter] f32 channelwise
    w2_scale: jax.Array,  # [size_group, 1, 1, hidden] f32 channelwise
    group_offset: jax.Array | None = None,  # int32[1]
    w1_bias: jax.Array | None = None,  # [size_group, 1, 2 * inter]
    w2_bias: jax.Array | None = None,  # [size_group, 1, hidden]
    *,
    fuse_act: str = "silu",
    tile_m: int | None = None,
    bucket_base: int | None = None,
    vmem_limit_bytes: int | None = None,
    acc_dtype: jnp.dtype | None = None,
    zero_initialize: bool = True,
    unconditional_pipeline: bool = True,
) -> jax.Array:
    """Fused MoE FFN: GMM1 (gate/up) + activation + GMM2 (down).

    Args:
        lhs: Input rows [size_m, hidden], fp8-quantized in-kernel.
        w1: Fused gate+up projection weights [g, hidden, 2 * inter], fp8.
        w2: Down projection weights [g, inter, hidden], fp8.
        group_sizes: Rows per group, int32[size_lhs_group].
        w1_scale: f32 weight scales [g, nb, 1, 2 * inter], nb divides hidden.
        w2_scale: f32 weight scales [g, nb, 1, hidden], nb divides inter.
        group_offset: Optional first group to process, int32[1].
        w1_bias: Optional gate+up bias [g, 1, 2 * inter], added to the
            GMM1 accumulator before the activation.
        w2_bias: Optional down-projection bias [g, 1, hidden], added to the
            GMM2 accumulator. Under tensor parallelism the caller must have
            zeroed it on every shard but one; it is added once per shard.
        fuse_act: Activation between the matmuls. Required, because w1 is
            gate|up fused. Same options as gmm_v2.
        tile_m: gm-tile rows, default min(128, size_m); halved until the
            VMEM estimate fits the limit.
        bucket_base: Requested row-count bucket granularity, or None for no
            bucketing. Snapped to a rung the realized tile can host.
        vmem_limit_bytes: VMEM limit, default 90% of capacity.
        acc_dtype: Accumulator dtype for both matmuls, default bf16.
        zero_initialize: Zero output rows outside the computed range.
        unconditional_pipeline: See gmm_v2.

    Returns:
        Output of shape [size_m, hidden].
    """
    size_m, hidden = lhs.shape
    if size_m == 0:
        raise ValueError(
            "gmm_fused needs at least one row; got an empty lhs. The "
            "caller must skip the call when its token count is zero.")
    size_group = w1.shape[0]
    if w1.shape[1] != hidden or w1.shape[2] % 2 != 0:
        raise ValueError(f"w1 shape {w1.shape} incompatible with lhs "
                         f"[{size_m}, {hidden}] (want [g, hidden, 2*inter])")
    inter = w1.shape[2] // 2
    if w2.shape != (size_group, inter, hidden):
        raise ValueError(
            f"w2 shape {w2.shape} != {(size_group, inter, hidden)}")
    # Scale layout, mirroring gmm_v2.validate_inputs: channelwise (nb == 1)
    # or k-block-scaled (nb == size_k / quant_block), with nb dividing size_k.
    w1_nb = w1_scale.shape[1]
    if (w1_scale.shape != (size_group, w1_nb, 1, 2 * inter)
            or hidden % w1_nb != 0):
        raise ValueError(
            f"w1_scale must be [g, nb, 1, 2*inter] with nb | hidden "
            f"(nb=1 channelwise), got {w1_scale.shape} for hidden={hidden}")
    w2_nb = w2_scale.shape[1]
    if (w2_scale.shape != (size_group, w2_nb, 1, hidden)
            or inter % w2_nb != 0):
        raise ValueError(
            f"w2_scale must be [g, nb, 1, hidden] with nb | inter "
            f"(nb=1 channelwise), got {w2_scale.shape} for inter={inter}")
    for name, bias, size_n in (("w1_bias", w1_bias, 2 * inter),
                               ("w2_bias", w2_bias, hidden)):
        if bias is not None and bias.shape != (size_group, 1, size_n):
            raise ValueError(f"{name} must be [g, 1, {size_n}], got "
                             f"{bias.shape}")
    if fuse_act is None:
        raise ValueError("fuse_act is required (w1 is gate|up fused)")

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if jnp.isscalar(group_offset):
            group_offset = group_offset[None]

    if vmem_limit_bytes is None:
        vmem_limit_bytes = default_vmem_limit_bytes()

    cfgs1, cfgs2, tiles1, tiles2, tile_m = build_stage_configs(
        lhs,
        w1,
        w2,
        w1_scale,
        w2_scale,
        group_sizes,
        group_offset,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        fuse_act=fuse_act,
        tile_m=tile_m,
        vmem_limit_bytes=vmem_limit_bytes,
        acc_dtype=acc_dtype,
        zero_initialize=zero_initialize,
        unconditional_pipeline=unconditional_pipeline,
    )
    dims1, dims2 = cfgs1.dims, cfgs2.dims

    if bucket_base is not None:
        if bucket_base <= 0:
            raise ValueError(f"{bucket_base=} must be positive")
        # The realized tile height is only known after the VMEM clamp above,
        # so the request is snapped to a rung that tile can host, not refused.
        derived = _derive_bucket_base(tile_m, dims1.size_lhs_sublane,
                                      bucket_base)
        if derived != bucket_base:
            logger.info_once(
                "gmm_fused: bucket base %d is not a rung of the ladder a "
                "%d-row tile can host (sublane %d, at most %d arms); using "
                "%d instead. The output is unchanged either way.", bucket_base,
                tile_m, dims1.size_lhs_sublane, MAX_BUCKET_ARMS, derived)
        bucket_base = derived
    # Scales and biases stay in HBM, windowed by the pipeline.
    w1_scale = w1_scale.astype(jnp.float32)
    w2_scale = w2_scale.astype(jnp.float32)
    hbm_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    hbm_w1_bias_spec = hbm_w2_bias_spec = None
    if w1_bias is not None:
        w1_bias = w1_bias.astype(jnp.float32)
        hbm_w1_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if w2_bias is not None:
        w2_bias = w2_bias.astype(jnp.float32)
        hbm_w2_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    max_num_gm = dims2.size_group + pl.cdiv(dims2.size_m, tile_m) - 1
    scratch_shapes = [
        # partial_out_ref (final output columns = tile_n2 = hidden)
        pltpu.VMEM(
            (PARTIAL_OUT_SUBLANES * dims2.size_lhs_sublane, tiles2.tile_n),
            cfgs2.out_dtype),
        # metadata_ref: shared by both stages (identical tiling), with the
        # same padding as gmm_v2 for fill_metadata's unconditional stores.
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 2, ), jnp.int32),
            gm_id_to_row_start=pltpu.SMEM((max_num_gm + 2, ), jnp.int32),
            gm_id_to_row_size=pltpu.SMEM((max_num_gm + 1, ), jnp.int32),
        ),
    ]

    num_lanes = pltpu.get_tpu_info().num_lanes
    if cfgs2.zero_init:
        out_bytes = jnp.dtype(cfgs2.out_dtype).itemsize
        tile_zero_m = ZERO_REF_TARGET_BYTES // num_lanes // out_bytes
        tile_zero_m = min(tile_zero_m, dims2.size_m)
        scratch_shapes += [
            pltpu.VMEM((tile_zero_m, num_lanes), cfgs2.out_dtype),
            pltpu.SemaphoreType.DMA((1, )),
        ]
    else:
        scratch_shapes += [None, None]

    aligned_n = align_to(cfgs2.out_size_n, num_lanes)
    out_init = jax.ShapeDtypeStruct((dims2.size_m, aligned_n), cfgs2.out_dtype)
    lhs_weights = WeightsRef(weight=lhs, scale=None, bias=None)
    w1_weights = WeightsRef(weight=w1, scale=w1_scale, bias=w1_bias)
    w2_weights = WeightsRef(weight=w2, scale=w2_scale, bias=w2_bias)

    return pl.pallas_call(
        functools.partial(fused_kernel_main,
                          cfgs1=cfgs1,
                          cfgs2=cfgs2,
                          bucket_base=bucket_base),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=None,
                    bias=None,
                ),
                WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=hbm_scale_spec,
                    bias=hbm_w1_bias_spec,
                ),
                WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=hbm_scale_spec,
                    bias=hbm_w2_bias_spec,
                ),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=get_fused_scope_name(cfgs1, cfgs2, bucket_base),
        cost_estimate=get_fused_cost_estimate(cfgs1, cfgs2),
        metadata=get_fused_metadata(cfgs1, cfgs2),
    )(group_sizes, group_offset, lhs_weights, w1_weights,
      w2_weights)[:, :cfgs2.out_size_n]
