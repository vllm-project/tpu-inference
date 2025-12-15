# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel."""

import functools

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import (
    TunedValue, get_device_vmem_limit, get_tuned_block_sizes)
from tpu_inference.kernels.quantized_matmul.util import (get_kernel_name,
                                                         next_multiple,
                                                         unfold_args)


def quantize_array(
    x: jax.Array,  # [bs_block_size, in_block_size]
    x_abs_max: jax.Array,  # [1, bs_block_size]
    quant_dtype: jnp.dtype,
):
    is_float = jnp.issubdtype(quant_dtype, jnp.floating)
    dtype_info = jnp.finfo(quant_dtype) if is_float else jnp.iinfo(quant_dtype)
    dtype_max = float(dtype_info.max)

    # TODO(kyuyeunk): Investigate performance gain from non xlu transpose.
    scale = jnp.transpose(x_abs_max / dtype_max)
    return (x / scale).astype(quant_dtype), scale.astype(jnp.float32)


def get_vmem_limit(
    n_batch: int,
    n_out: int,
    n_in: int,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
    x_dtype: jnp.dtype,
    x_q_dtype: jnp.dtype,
    w_q_dtype: jnp.dtype,
    scale_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    save_acc: bool,
    save_x_q: bool,
    upper_limit_bytes: int,
):
    """Calculate VMEM limit for the kernel."""

    # Calculate in/out VMEM size.
    x_size = batch_block_size * in_block_size * dtypes.bit_width(x_dtype)
    x_abs_max_size = batch_block_size * dtypes.bit_width(scale_dtype)
    w_q_size = out_block_size * in_block_size * dtypes.bit_width(w_q_dtype)
    w_scale_size = out_block_size * dtypes.bit_width(scale_dtype)
    out_size = batch_block_size * out_block_size * dtypes.bit_width(out_dtype)

    vmem_in_out = x_size + x_abs_max_size + w_q_size + w_scale_size + out_size
    vmem_in_out *= 2  # Account for compute and vreg spills.

    # Account for double buffering.
    # Double buffering is used only if there are multiple blocks per in/out.
    vmem_in_out += x_size if (n_batch > 1 or n_in > 1) else 0
    vmem_in_out += x_abs_max_size if (n_batch > 1) else 0
    vmem_in_out += w_q_size if (n_out > 1 or n_in > 1) else 0
    vmem_in_out += w_scale_size if (n_out > 1) else 0
    vmem_in_out += out_size if (n_batch > 1 or n_out > 1) else 0

    # Calculate scratch VMEM size.
    acc_size = batch_block_size * out_block_size * dtypes.bit_width(acc_dtype)
    x_q_size = batch_block_size * in_block_size * dtypes.bit_width(x_q_dtype)
    x_scale_size = batch_block_size * dtypes.bit_width(scale_dtype)

    vmem_scratch = acc_size if save_acc else 0
    vmem_scratch += x_q_size + x_scale_size if save_x_q else 0
    vmem_scratch *= 2  # Account for compute and vreg spills.

    # Add in/out and scratch VMEM size.
    vmem_used = vmem_in_out + vmem_scratch
    vmem_used_bytes = vmem_used // 8  # Convert bits to bytes.
    # Specify upper limit. Defaults to 96MB.
    vmem_limit_bytes = min(vmem_used_bytes, upper_limit_bytes)

    return vmem_limit_bytes


def validate_inputs(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    x_abs_max: jax.Array,
    x_q_dtype: jnp.dtype,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
):
    """Verify inputs invoking the kernel."""

    if x.dtype != x_q_dtype:
        # If the input is quantized, then it should be the same subdtype as w_q
        if jnp.issubdtype(x_q_dtype, jnp.integer) != jnp.issubdtype(
                w_q.dtype, jnp.integer):
            raise ValueError(
                f'{x_q_dtype=} and {w_q.dtype=} must be the same int or float type.'
            )

    # Verify input shapes.
    if x.shape[1] != w_q.shape[0]:
        raise ValueError(f'{x.shape[1]=} must be equal to {w_q.shape[0]=}')
    if w_q.shape[1] != w_scale.shape[1]:
        raise ValueError(
            f'{w_q.shape[1]=} must be equal to {w_scale.shape[1]=}')
    if x_abs_max.shape != (1, x.shape[0]):
        raise ValueError(
            f'{x_abs_max.shape=} must be equal to (1, {x.shape[0]=})')
    if x.shape[0] % batch_block_size != 0:
        raise ValueError(
            f'{x.shape[0]=} must be a multiple of {batch_block_size=}')
    if w_q.shape[1] % out_block_size != 0:
        raise ValueError(
            f'{w_q.shape[1]=} must be a multiple of {out_block_size=}')
    if x.shape[1] % in_block_size != 0:
        raise ValueError(
            f'{x.shape[1]=} must be a multiple of {in_block_size=}')


def subchannel_matmul_kernel(lhs_ref, rhs_ref, w_scales_ref, out_ref,
                             acc_scratch, sc_size, dtype_lhs, k_dim, block_m,
                             block_k, steps_k, steps_n, compute_tile_n):
    print(
        "Printing k_dim, block_m, block_k, steps_k, steps_n, compute_tile_n:",
        k_dim, block_m, block_k, steps_k, steps_n, compute_tile_n)
    pid_k = pl.program_id(2)
    is_first_step = pid_k == 0
    is_last_step = pid_k == (k_dim // block_k - 1)

    @pl.when(is_first_step)
    def _init():
        acc_scratch[...] = jnp.zeros_like(acc_scratch)

    # Outer Loop: Iterate through K-dimension sub-channels
    acc_dtype = jnp.bfloat16

    # Pre-calculate all quantized blocks for this tile. Relieves register
    # pressure during the compute phase.
    lhs_q_list = []
    lhs_scale_list = []
    rhs_q_list = []
    rhs_scale_list = []

    for i in range(steps_k):
        k_start, k_end = i * sc_size, (i + 1) * sc_size

        lhs_sub = lhs_ref[:, k_start:k_end].astype(jnp.float32)
        lhs_sub_max = jnp.max(jnp.abs(lhs_sub), axis=-1, keepdims=False)
        lhs_sub_max = jnp.expand_dims(lhs_sub_max, axis=0)
        l_q, l_s = quantize_array(lhs_sub, lhs_sub_max, dtype_lhs)
        lhs_q_list.append(l_q)
        # cast scale to acc_dtype IMMEDIATELY to save register space
        lhs_scale_list.append(l_s.astype(acc_dtype))

        rhs_q_list.append(rhs_ref[k_start:k_end, :])
        rhs_scale_list.append(w_scales_ref[i, :, :].astype(acc_dtype))

    accumulators = [
        jnp.zeros((block_m, compute_tile_n), dtype=acc_dtype)
        for _ in range(steps_n)
    ]
    for i in range(steps_k):
        lhs_q = lhs_q_list[i]
        lhs_scale = lhs_scale_list[i]
        rhs_q_full = rhs_q_list[i]
        rhs_q_full = rhs_q_list[i]
        rhs_q_full = rhs_q_list[i]
        rhs_scale_full = rhs_scale_list[i]

        # Inner Loop: stripmine the N dimension to respect MXU_SIZE constraints
        for j in range(steps_n):
            n_start, n_end = j * compute_tile_n, (j + 1) * compute_tile_n

            rhs_q_slice = rhs_q_full[:, n_start:n_end]
            rhs_scale_slice = rhs_scale_full[:, n_start:n_end]
            if dtype_lhs == jnp.int4 or dtype_lhs == jnp.int8:
                preferred_element_type = jnp.int32
            else:
                preferred_element_type = jnp.float32
            dot_res = jnp.dot(lhs_q,
                              rhs_q_slice,
                              preferred_element_type=preferred_element_type)
            res = dot_res.astype(acc_dtype)
            # Broadcast multiply LHS scale (M, 1) -> (M, N)
            # This keeps the scale in a small register footprint
            res = res * lhs_scale
            # Broadcast multiply RHS scale (1, N) -> (M, N)
            res = res * rhs_scale_slice

            accumulators[j] += res
    acc_block = jnp.concatenate(accumulators, axis=1)
    print("Printing acc_block.shape:", acc_block.shape)
    acc_scratch[...] += acc_block

    @pl.when(is_last_step)
    def _write():
        out_ref[...] = acc_scratch[...].astype(out_ref.dtype)


def matmul_kernel(
    x_ref: jax.Array,  # (batch_block_size, in_block_size)
    w_q_ref: jax.Array,  # (in_block_size, out_block_size)
    w_scale_ref: jax.Array,  # (1, out_block_size)
    x_abs_max_ref: jax.Array,  # (1, batch_block_size)
    out_ref: jax.Array,  # (batch_block_size, out_block_size)
    acc_scratch: jax.Array,  # (batch_block_size, out_block_size)
    x_q_scratch: jax.Array,  # (batch_block_size, in_block_size)
    x_scale_scratch: jax.Array,  # (batch_block_size, 1)
    *,
    x_q_dtype: jnp.dtype,
    save_acc: bool,
    save_x_q: bool,
):
    out_idx, in_idx = pl.program_id(1), pl.program_id(2)
    n_in = pl.num_programs(2)
    x_ref_dtype = x_ref.dtype

    quantize_activation = x_q_dtype != x_ref_dtype

    # Initialize conditional logic.
    if save_x_q:
        assert quantize_activation
        assert x_q_scratch is not None
        assert x_scale_scratch is not None
        quant = out_idx == 0
    else:
        assert x_q_scratch is None
        assert x_scale_scratch is None
        quant = quantize_activation

    if save_acc:
        assert acc_scratch is not None
        is_first_step = in_idx == 0
        is_last_step = in_idx == (n_in - 1)
    else:
        assert acc_scratch is None
        is_first_step = True
        is_last_step = True

    acc_dtype = jnp.float32
    if quantize_activation and jnp.issubdtype(w_q_ref.dtype, jnp.integer):
        acc_dtype = jnp.int32

    # Start of actual computation logic.
    def matmul_body(quant: bool, is_first_step: bool, is_last_step: bool):
        if quantize_activation:
            if quant:
                x_q_tmp, x_scale_tmp = quantize_array(
                    x_ref[...],
                    x_abs_max_ref[...],
                    x_q_dtype,
                )

                if save_x_q:
                    x_q_scratch[...] = x_q_tmp
                    x_scale_scratch[...] = x_scale_tmp

            else:
                assert save_x_q
                x_q_tmp = x_q_scratch[...]
                if is_last_step:
                    x_scale_tmp = x_scale_scratch[...]

            acc = jnp.matmul(
                x_q_tmp,
                w_q_ref[...],
                preferred_element_type=acc_dtype,
            )
        else:
            acc = jnp.matmul(
                x_ref[...],
                w_q_ref[...],
                preferred_element_type=acc_dtype,
            )

        if not is_first_step:
            acc += acc_scratch[...]

        if is_last_step:
            acc *= w_scale_ref[...]
            if quantize_activation:
                # TODO(kyuyeunk): Investigate caching broadcast.
                acc *= x_scale_tmp
            out_ref[...] = acc.astype(x_ref_dtype)
        else:
            assert save_acc
            acc_scratch[...] = acc

    unfold_args((quant, is_first_step, is_last_step), (), matmul_body)


@functools.partial(
    jax.jit,
    static_argnames=[
        'x_q_dtype',
        'tuned_value',
        'sc_size',
    ],
)
def quantized_matmul_kernel(
    x: jax.Array,  # [bs, n_in]
    w_q: jax.Array,  # [n_in, n_out]
    w_scale: jax.Array,  # [n_blocks, n_out]
    w_zp: jax.Array | None = None,  # [n_out]
    x_q_dtype: jnp.dtype | None = None,
    *,
    tuned_value: TunedValue | None = None,
    sc_size: int | None = None,
) -> jax.Array:
    """Quantized matmul kernel.

  Args:
    x: Input unquantized array.
    w_q: Weight quantized array. [n_input_features, n_output_features]
    w_scale: Weight quantization scale. [n_blocks, n_output_features]
    w_zp: Weight zero point for asymmetric quantization.
    x_q_dtype: Quantization type of the input. If None or if the value is the
      same as x.dtype, then no quantization is applied.
    tuned_value: Kernel tuned values for optimal performance.

  Returns:
    Quantized matmul result.
  """

    if w_zp is not None:
        raise NotImplementedError('zero_point is not supported.')

    if x_q_dtype is None:
        x_q_dtype = x.dtype
    quantize_activation = x_q_dtype != x.dtype

    # Pallas kernel only has access to a single block of the input. Therefere,
    # for per-token quantization, abs max has to be computed outside of the
    # kernel.
    if not sc_size:
        x_abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=False)  # [bs]
        # Pallas requires minormost dim to be a multiple of sublane size 128.
        # Therefore, instead of using [bs, 1], we reshape this into [1, bs]
        x_abs_max = jnp.expand_dims(x_abs_max, axis=0)  # [1, bs]
        assert x_abs_max.shape == (1, x.shape[0])

    orig_n_batch, orig_n_in = x.shape
    _, orig_n_out = w_q.shape

    if tuned_value is None:
        tuned_value = get_tuned_block_sizes(
            n_batch=orig_n_batch,
            n_out=orig_n_out,
            n_in=orig_n_in,
            x_q_dtype=jnp.dtype(x_q_dtype).name,
            w_q_dtype=jnp.dtype(w_q.dtype).name,
        )
    print("Printing tuned_value:", tuned_value)
    batch_block_size = tuned_value.batch_block_size
    out_block_size = tuned_value.out_block_size
    in_block_size = tuned_value.in_block_size

    # Pad the inputs to be multiple of block size.
    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))
        if not sc_size:
            x_abs_max = jnp.pad(x_abs_max,
                                ((0, 0), (0, padded_n_batch - orig_n_batch)))
    padded_n_out = next_multiple(orig_n_out, out_block_size)
    if orig_n_out < padded_n_out:
        w_q = jnp.pad(w_q, ((0, 0), (0, padded_n_out - orig_n_out)))
        w_scale = jnp.pad(w_scale, ((0, 0), (0, padded_n_out - orig_n_out)))
    padded_n_in = next_multiple(orig_n_in, in_block_size)
    if orig_n_in < padded_n_in:
        x = jnp.pad(x, ((0, 0), (0, padded_n_in - orig_n_in)))
        w_q = jnp.pad(w_q, ((0, padded_n_in - orig_n_in), (0, 0)))

    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)

    n_batch = padded_n_batch // batch_block_size
    n_out = padded_n_out // out_block_size
    n_in = padded_n_in // in_block_size

    save_acc = n_in > 1
    # Remove redundant input quantization logic by caching quantized input. For
    # best performance, only enable this behavior when single input block is
    # used per batch.
    save_x_q = quantize_activation and n_in == 1 and n_out > 1

    acc_dtype = jnp.float32
    if quantize_activation and jnp.issubdtype(w_q.dtype, jnp.integer):
        acc_dtype = jnp.int32

    vmem_limit_bytes = get_vmem_limit(
        n_batch=n_batch,
        n_out=n_out,
        n_in=n_in,
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size,
        x_dtype=x.dtype,
        x_q_dtype=x_q_dtype,
        w_q_dtype=w_q.dtype,
        scale_dtype=jnp.float32,
        out_dtype=x.dtype,
        acc_dtype=acc_dtype,
        save_acc=save_acc,
        save_x_q=save_x_q,
        upper_limit_bytes=get_device_vmem_limit(),
    )

    x_block = pl.BlockSpec((batch_block_size, in_block_size), lambda b, o, i:
                           (b, i))
    w_q_block = pl.BlockSpec((in_block_size, out_block_size), lambda b, o, i:
                             (i, o))
    if sc_size:
        compute_tile_n = 256 * 4
        w_scale_block = pl.BlockSpec(
            (in_block_size // sc_size, 1, out_block_size), lambda b, o, i:
            (i, 0, o))
    else:
        w_scale_block = pl.BlockSpec((1, out_block_size), lambda b, o, i:
                                     (0, o))

    if sc_size:
        kernel = pl.pallas_call(
            functools.partial(
                subchannel_matmul_kernel,
                dtype_lhs=x_q_dtype,
                sc_size=sc_size,
                k_dim=orig_n_batch,
                block_m=batch_block_size,
                block_k=in_block_size,
                steps_k=in_block_size // sc_size,
                steps_n=out_block_size // compute_tile_n,
                compute_tile_n=compute_tile_n,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    x_block,
                    w_q_block,
                    w_scale_block,
                ],
                out_specs=pl.BlockSpec((batch_block_size, out_block_size),
                                       lambda b, o, i: (b, o)),
                scratch_shapes=[
                    pltpu.VMEM((batch_block_size, out_block_size),
                               jnp.bfloat16)
                ],
                grid=(n_batch, n_out, n_in),
            ),
            out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out),
                                           x.dtype),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=('parallel', 'parallel', 'arbitrary'),
                vmem_limit_bytes=vmem_limit_bytes,
            ),
        )
    else:
        kernel = pl.pallas_call(
            functools.partial(
                matmul_kernel,
                x_q_dtype=x_q_dtype,
                save_acc=save_acc,
                save_x_q=save_x_q,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((batch_block_size, in_block_size),
                                 lambda b, o, i: (b, i)),  # x
                    pl.BlockSpec((in_block_size, out_block_size),
                                 lambda b, o, i: (i, o)),  # w_q
                    pl.BlockSpec((1, out_block_size), lambda b, o, i:
                                 (0, o)),  # w_scale
                    pl.BlockSpec((1, batch_block_size), lambda b, o, i:
                                 (0, b)),  # x_abs_max
                ],
                out_specs=pl.BlockSpec((batch_block_size, out_block_size),
                                       lambda b, o, i: (b, o)),
                scratch_shapes=[
                    pltpu.VMEM((batch_block_size, out_block_size), acc_dtype)
                    if save_acc else None,  # acc_scratch
                    pltpu.VMEM((batch_block_size, in_block_size), x_q_dtype)
                    if save_x_q else None,  # x_q_scratch
                    pltpu.VMEM((batch_block_size, 1), jnp.float32)
                    if save_x_q else None,  # x_scale_scratch
                ],
                grid=(n_batch, n_out, n_in),
            ),
            out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out),
                                           x.dtype),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=('parallel', 'arbitrary', 'arbitrary'),
                vmem_limit_bytes=vmem_limit_bytes,
            ),
        )

    #validate_inputs(
    #x=x,
    #w_q=w_q,
    #w_scale=w_scale,
    #x_abs_max=x_abs_max,
    #x_q_dtype=x_q_dtype,
    #batch_block_size=batch_block_size,
    #out_block_size=out_block_size,
    #in_block_size=in_block_size,
    #)

    # The named_scope is used for autotune.
    kernel_name = get_kernel_name(tuned_value)
    with jax.named_scope(kernel_name):
        if sc_size:
            out = kernel(x, w_q, w_scale)
        else:
            out = kernel(x, w_q, w_scale, x_abs_max)

    return out[:orig_n_batch, :orig_n_out]
