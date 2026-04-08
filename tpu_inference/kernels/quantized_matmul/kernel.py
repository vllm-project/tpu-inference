# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.quantized_matmul import util
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import (
    TunedValue, get_device_vmem_limit, get_tuned_block_sizes)
from tpu_inference.kernels.quantized_matmul.util import (get_kernel_name,
                                                         next_multiple,
                                                         unfold_args)

quantize_tensor = util.quantize_tensor


def matmul_kernel(
    x_ref: jax.Array,  # (batch_block_size, in_block_size)
    w_q_ref: jax.Array,  # (out_block_size, in_block_size)
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
                x_q_tmp, x_scale_tmp = util.quantize_array(
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

            acc = jax.lax.dot_general(
                x_q_tmp,
                w_q_ref[...],
                (((1, ), (1, )), ((), ())),
                preferred_element_type=acc_dtype,
            )
        else:
            acc = jax.lax.dot_general(
                x_ref[...],
                w_q_ref[...],
                (((1, ), (1, )), ((), ())),
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


@jax.jit(static_argnames=[
    "x_q_dtype",
    "tuned_value",
])
def quantized_matmul_kernel(
    x: jax.Array,  # [bs, n_in]
    w_q: jax.Array,  # [n_out, n_in]
    w_scale: jax.Array,  # [n_out]
    w_zp: jax.Array | None = None,  # [n_out]
    block_size: int | None = None,
    x_q_dtype: jnp.dtype | None = None,
    *,
    tuned_value: TunedValue | None = None,
) -> jax.Array:
    """Quantized matmul kernel."""

    if w_zp is not None:
        raise NotImplementedError("zero_point is not supported.")
    if block_size is not None:
        raise NotImplementedError("block_size is not supported.")

    if x_q_dtype is None:
        x_q_dtype = x.dtype
    quantize_activation = x_q_dtype != x.dtype

    x_abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=False)  # [bs]
    x_abs_max = jnp.expand_dims(x_abs_max, axis=0)  # [1, bs]

    orig_n_batch, orig_n_in = x.shape
    orig_n_out, _ = w_q.shape

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

    # Pad the inputs to be multiple of block size.
    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))
        x_abs_max = jnp.pad(x_abs_max,
                            ((0, 0), (0, padded_n_batch - orig_n_batch)))
    padded_n_out = next_multiple(orig_n_out, out_block_size)
    if orig_n_out < padded_n_out:
        w_q = jnp.pad(w_q, ((0, padded_n_out - orig_n_out), (0, 0)))
        w_scale = jnp.pad(w_scale, (0, padded_n_out - orig_n_out))
    padded_n_in = next_multiple(orig_n_in, in_block_size)
    if orig_n_in < padded_n_in:
        x = jnp.pad(x, ((0, 0), (padded_n_in - orig_n_in)))
        w_q = jnp.pad(w_q, ((0, 0), (padded_n_in - orig_n_in)))

    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)
    w_scale = jnp.expand_dims(w_scale, axis=0)  # [1, n_output_features]

    n_batch = padded_n_batch // batch_block_size
    n_out = padded_n_out // out_block_size
    n_in = padded_n_in // in_block_size

    save_acc = n_in > 1
    save_x_q = quantize_activation and n_in == 1 and n_out > 1

    acc_dtype = jnp.float32
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
                pl.BlockSpec((batch_block_size, in_block_size), lambda b, o, i:
                             (b, i)),  # x
                pl.BlockSpec((out_block_size, in_block_size), lambda b, o, i:
                             (o, i)),  # w_q
                pl.BlockSpec((1, out_block_size), lambda b, o, i:
                             (0, o)),  # w_scale
                pl.BlockSpec((1, batch_block_size), lambda b, o, i:
                             (0, b)),  # x_abs_max
            ],
            out_specs=pl.BlockSpec((batch_block_size, out_block_size),
                                   lambda b, o, i: (b, o)),
            scratch_shapes=[
                (pltpu.VMEM((batch_block_size, out_block_size), acc_dtype)
                 if save_acc else None),  # acc_scratch
                (pltpu.VMEM((batch_block_size, in_block_size), x_q_dtype)
                 if save_x_q else None),  # x_q_scratch
                (pltpu.VMEM((batch_block_size, 1), jnp.float32)
                 if save_x_q else None),  # x_scale_scratch
            ],
            grid=(n_batch, n_out, n_in),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out),
                                       x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )

    # The named_scope is used for autotune.
    kernel_name = get_kernel_name(tuned_value)
    with jax.named_scope(kernel_name):
        out = kernel(x, w_q, w_scale, x_abs_max)

    return out[:orig_n_batch, :orig_n_out]


def _batched_matmul_body(
    x_ref: jax.Array,           # (batch_block_size, heads_block_size, in_block_size)
    w_ref: jax.Array,           # (heads_block_size, in_block_size, out_block_size)
    out_ref: jax.Array,         # (batch_block_size, heads_block_size, out_block_size)
    acc_scratch: jax.Array,     # (batch_block_size, heads_block_size, out_block_size)
    *,
    save_acc: bool,
):
    """Shared body for simplified 3D batched matmul kernel.

    Grid axes: (batch_idx=0, out_idx=1, in_idx=2).
    """
    in_idx  = pl.program_id(2)
    n_in = pl.num_programs(2)
    x_ref_dtype = x_ref.dtype
    

    if save_acc:
        is_first_step = in_idx == 0
        is_last_step  = in_idx == (n_in - 1)
    else:
        is_first_step = True
        is_last_step  = True

    def matmul_body(is_first_step: bool, is_last_step: bool):
        x_block = x_ref[...] # (batch_block_size, heads_block_size, in_block_size)
        w_block = w_ref[...] # (heads_block_size, in_block_size, out_block_size)

        # Compute [batch, heads, out] = [batch, heads, in] @ [heads, in, out]
        acc_block = jnp.einsum('tnp,npl->tnl', x_block, w_block, preferred_element_type=jnp.float32)

        if not is_first_step:
            acc_block += acc_scratch[...]

        if is_last_step:
            out_ref[...] = acc_block.astype(x_ref_dtype)
        else:
            acc_scratch[...] = acc_block

    unfold_args((is_first_step, is_last_step), (), matmul_body)


def batched_matmul_kernel(
    x_ref: jax.Array,
    w_ref: jax.Array,
    out_ref: jax.Array,
    acc_scratch: jax.Array,
    *,
    save_acc: bool,
):
    """Wrapper to maintain compatibility with existing call site."""
    _batched_matmul_body(
        x_ref, w_ref, out_ref, acc_scratch,
        save_acc=save_acc,
    )


@jax.jit(static_argnames=[
    "x_q_dtype",
    "tuned_value",
])
def batched_quantized_matmul_kernel(
    x: jax.Array,       # [T, N, P]  tokens first
    w_q: jax.Array,     # [N, L, P]  output-features first per head, quantized
    w_scale: jax.Array, # [N, L] per-output-feature scale, or scalar (shape ())
    x_q_dtype: jnp.dtype | None = None,
    *,
    tuned_value: TunedValue | None = None,
) -> jax.Array:  # [T, N, L]  tokens-major physical layout
    """Batched matmul: out[t, n, :] = x[t, n, :] @ w_q[n].T. Simplified: no quantization."""
    
    orig_n_batch, n_heads, orig_n_in = x.shape
    _, _, orig_n_out                 = w_q.shape

    # TODO: update the hardcodings to be configurable.
    heads_block_size = 128 
    batch_block_size = 128
    out_block_size   = 128
    in_block_size    = 128

    if tuned_value is None:
        tuned_value = TunedValue(batch_block_size, out_block_size, in_block_size)

    # Pad N to block-size multiples.
    padded_n_heads = next_multiple(n_heads, heads_block_size)
    if n_heads < padded_n_heads:
        pad = padded_n_heads - n_heads
        x   = jnp.pad(x,   ((0, 0), (0, pad), (0, 0))) # pad N dim
        w_q = jnp.pad(w_q, ((0, pad), (0, 0), (0, 0))) # pad N dim

    # Pad T (Tokens) dimension in x.
    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        pad = padded_n_batch - orig_n_batch
        x   = jnp.pad(x,   ((0, pad), (0, 0), (0, 0))) # pad T dim

    # Pad L (dim 2) for out, pad P (dim 1) for in.
    padded_n_out = next_multiple(orig_n_out, out_block_size)
    if orig_n_out < padded_n_out:
        pad = padded_n_out - orig_n_out
        w_q = jnp.pad(w_q, ((0, 0), (0, 0), (0, pad)))  # pad L dim

    padded_n_in = next_multiple(orig_n_in, in_block_size)
    if orig_n_in < padded_n_in:
        pad = padded_n_in - orig_n_in
        x   = jnp.pad(x,   ((0, 0), (0, 0), (0, pad)))  # pad P dim in x
        w_q = jnp.pad(w_q, ((0, 0), (0, pad), (0, 0)))  # pad P dim in w_q

    n_batch_blocks = padded_n_batch // batch_block_size
    n_out_blocks   = padded_n_out   // out_block_size
    n_in_blocks    = padded_n_in    // in_block_size

    save_acc = n_in_blocks > 1
    acc_dtype = jnp.float32

    # TODO update to have accuracy bounds.
    vmem_limit_bytes = get_device_vmem_limit()

    x_spec = pl.BlockSpec(
        # x: [T, N, P]
        (batch_block_size, heads_block_size, in_block_size),
        lambda b, o, i: (b, 0, i),
    )
    w_q_spec = pl.BlockSpec(
        # w_q: [N, P, L]
        (heads_block_size, in_block_size, out_block_size),
        lambda b, o, i: (0, i, o),
    )
    out_spec = pl.BlockSpec(
        # out: [T, N, L]
        (batch_block_size, heads_block_size, out_block_size),
        lambda b, o, i: (b, 0, o),
    )
    scratch_shapes = [
        (pltpu.VMEM((batch_block_size, heads_block_size, out_block_size), acc_dtype)
         if save_acc else None),
    ]
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel", "arbitrary", "arbitrary"),
        vmem_limit_bytes=vmem_limit_bytes,
    )
    out_shape = jax.ShapeDtypeStruct((padded_n_batch, padded_n_heads, padded_n_out), x.dtype)

    kernel_name = get_kernel_name(tuned_value)
    with jax.named_scope(kernel_name):
        kernel = pl.pallas_call(
            functools.partial(
                batched_matmul_kernel,
                save_acc=save_acc,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[x_spec, w_q_spec], 
                out_specs=out_spec,
                scratch_shapes=scratch_shapes,
                grid=(n_batch_blocks, n_out_blocks, n_in_blocks),
            ),
            out_shape=out_shape,
            compiler_params=compiler_params,
        )
        out = kernel(x, w_q)

    return out[:orig_n_batch, :n_heads, :orig_n_out]
