# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel with blockwise quantization support."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.quantized_matmul import util
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import (
    TunedValue, get_device_vmem_limit, get_tuned_block_sizes)
from tpu_inference.kernels.quantized_matmul.util import (get_kernel_name,
                                                         next_multiple)

quantize_tensor = util.quantize_tensor


@functools.partial(
    jax.jit,
    static_argnames=[
        'block_size',
        'x_q_dtype',
        'tuned_value',
    ],
)
def quantized_matmul_kernel(
    x: jax.Array,  # [bs, n_in]
    w_q: jax.Array,  # [n_out, n_in]
    w_scale: jax.Array,  # [n_in // block_size, 1, n_out]
    w_zp: jax.Array | None = None,  # [n_out]
    block_size: int | None = None,
    x_q_dtype: jnp.dtype | None = None,
    *,
    tuned_value: TunedValue | None = None,
) -> jax.Array:
    """Quantized matmul kernel with blockwise support.

  Args:
    x: Input unquantized array.
    w_q: Weight quantized array. [n_output_features, n_input_features]
    w_scale: Weight quantization scale. [n_input_features // block_size, 1, n_output_features]
    w_zp: Weight zero point for asymmetric quantization.
    block_size: Block size for subchannel quantization.
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

    orig_n_batch, orig_n_in = x.shape
    orig_n_out, *_ = w_q.shape

    if tuned_value is None:
        tuned_value = get_tuned_block_sizes(
            n_batch=orig_n_batch,
            n_out=orig_n_out,
            n_in=orig_n_in,
            x_q_dtype=jnp.dtype(x_q_dtype).name,
            w_q_dtype=jnp.dtype(w_q.dtype).name,
        )
    batch_block_size = tuned_value.batch_block_size
    out_block_size = tuned_value.out_block_size
    in_block_size = tuned_value.in_block_size
    n_lane_multiplier = tuned_value.n_lane_multiplier
    block_size = tuned_value.in_block_size if block_size is None else block_size

    # Pad the inputs to be multiple of block size.
    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))

    padded_n_out = next_multiple(orig_n_out, out_block_size)
    if orig_n_out < padded_n_out:
        w_q = jnp.pad(w_q, ((0, padded_n_out - orig_n_out), (0, 0)))
        w_scale = jnp.pad(w_scale, (0, padded_n_out - orig_n_out))
    padded_n_in = next_multiple(orig_n_in, in_block_size)
    if orig_n_in < padded_n_in:
        x = jnp.pad(x, ((0, 0), (0, padded_n_in - orig_n_in)))
        w_q = jnp.pad(w_q, ((0, 0), (0, padded_n_in - orig_n_in)))

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

    acc_dtype = jnp.bfloat16
    if quantize_activation and jnp.issubdtype(w_q.dtype, jnp.integer):
        acc_dtype = jnp.int32

    vmem_limit_bytes = util.get_vmem_limit(
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

    steps_k = in_block_size // block_size
    # n_lane_multiplier > 1 could improve perf by reducing loop overhead and increasing instruction-level parallelism,
    # allowing the compiler to overlap output fusion and packing overhead with MXU computation
    compute_tile_n = pltpu.get_tpu_info().mxu_column_size * n_lane_multiplier
    steps_n = out_block_size // compute_tile_n

    def kernel(lhs_ref, rhs_ref, w_scales_ref, out_ref, acc_scratch):
        pid_k = pl.program_id(2)
        is_first_step = pid_k == 0
        is_last_step = pid_k == (orig_n_in // in_block_size - 1)

        @pl.when(is_first_step)
        def _init():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

        # Pre-calculate all quantized blocks for this tile. Relieves register
        # pressure during the compute phase.
        lhs_q_list = []
        lhs_scale_list = []
        rhs_q_list = []
        rhs_scale_list = []

        for i in range(steps_k):
            k_start, k_end = i * block_size, (i + 1) * block_size

            lhs_sub = lhs_ref[:, k_start:k_end].astype(jnp.float32)
            l_q, l_s = util.quantize_block(lhs_sub, 1, x_q_dtype)
            lhs_q_list.append(l_q)
            lhs_scale_list.append(l_s.astype(acc_dtype))
            rhs_q_list.append(rhs_ref[:, k_start:k_end])
            rhs_scale_list.append(w_scales_ref[i, :, :].astype(acc_dtype))

        accumulators = [
            jnp.zeros((batch_block_size, compute_tile_n), dtype=acc_dtype)
            for _ in range(steps_n)
        ]
        for i in range(steps_k):
            lhs_q = lhs_q_list[i]
            lhs_scale = lhs_scale_list[i]
            rhs_q_full = rhs_q_list[i]
            rhs_scale_full = rhs_scale_list[i]

            for j in range(steps_n):
                n_start, n_end = j * compute_tile_n, (j + 1) * compute_tile_n

                rhs_q_slice = (rhs_q_full[n_start:n_end, :])
                rhs_scale_slice = rhs_scale_full[:, n_start:n_end]
                if x_q_dtype == jnp.int4 or x_q_dtype == jnp.int8:
                    preferred_element_type = jnp.int32
                else:
                    preferred_element_type = jnp.float32
                dot_res = jax.lax.dot_general(
                    lhs_q,
                    rhs_q_slice.astype(lhs_q.dtype),
                    (((1, ), (1, )), ((), ())),
                    preferred_element_type=preferred_element_type,
                )
                res = dot_res.astype(acc_dtype)
                res = res * lhs_scale
                res = res * rhs_scale_slice

                accumulators[j] += res
        acc_block = jnp.concatenate(accumulators, axis=1)
        acc_scratch[...] += acc_block

        @pl.when(is_last_step)
        def _write():
            out_ref[...] = acc_scratch[...].astype(out_ref.dtype)

    kernel = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((batch_block_size, in_block_size),
                             lambda b, o, i: (b, i),
                             memory_space=pltpu.VMEM),  # x
                pl.BlockSpec((out_block_size, in_block_size),
                             lambda b, o, i: (o, i),
                             memory_space=pltpu.VMEM),  # w_q
                pl.BlockSpec((steps_k, 1, out_block_size),
                             lambda _, o, i: (i, 0, o),
                             memory_space=pltpu.VMEM)
            ],  # w_scale
            out_specs=pl.BlockSpec((batch_block_size, out_block_size),
                                   lambda b, o, i: (b, o)),
            scratch_shapes=[
                pltpu.VMEM((batch_block_size, out_block_size), jnp.bfloat16)
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

    util.validate_inputs(
        x=x,
        w_q=w_q,
        w_scale=w_scale,
        x_abs_max=None,
        x_q_dtype=x_q_dtype,
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size,
    )

    # The named_scope is used for autotune.
    kernel_name = get_kernel_name(tuned_value)
    with jax.named_scope(kernel_name):
        out = kernel(x, w_q, w_scale)

    return out[:orig_n_batch, :orig_n_out]
