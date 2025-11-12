import functools
import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.quantized_matmul.util import unfold_args, next_multiple
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import (
    TunedValue,
    get_tuned_block_sizes,
    get_device_vmem_limit
)

def _quantize_array(
    x: jax.Array,
    x_abs_max: jax.Array,
    quant_dtype: jnp.dtype,
):
    """Quantizes an input array using per-row scaling (1D quantization).
    
    Args:
        x: Input array of shape [batch_block_size, in_block_size]
        x_abs_max: Maximum absolute values of shape [1, batch_block_size]
        quant_dtype: Target quantization dtype (int8, float8_e4m3fn, etc.)
        
    Returns:
        quantized_array: Quantized array in quant_dtype
        scale: Dequantization scale of shape [batch_block_size, 1]
    """
    is_float = jnp.issubdtype(quant_dtype, jnp.floating)
    
    if is_float:
        dtype_info = jnp.finfo(quant_dtype)
        dtype_max = float(dtype_info.max)
        if quant_dtype == jnp.float8_e4m3fn:
            dtype_max = 448.0
        dtype_min = float(dtype_info.min)
    else:
        dtype_info = jnp.iinfo(quant_dtype)
        dtype_max = float(dtype_info.max)
        dtype_min = float(dtype_info.min)
    
    scale_basis = jnp.transpose(x_abs_max)
    scale_basis = jnp.maximum(scale_basis, jnp.finfo(jnp.float32).tiny)
    scale = scale_basis / dtype_max

    x_scaled = x.astype(jnp.float32) / scale

    if not is_float:
        x_scaled = jnp.round(x_scaled)
        quantized_array = jnp.clip(x_scaled, dtype_min, dtype_max).astype(quant_dtype)
    else:
        quantized_array = x_scaled.astype(quant_dtype)

    return quantized_array, scale.astype(jnp.float32)


def _quantize_array_2d(
    x: jax.Array,
    x_abs_max: jax.Array,
    quant_dtype: jnp.dtype,
):
    """Quantizes an input array using block-wise scaling (2D quantization).
    
    Args:
        x: Input array of shape [batch_block_size, quant_block_size]
        x_abs_max: Maximum absolute values per block, shape [batch_block_size, 1]
        quant_dtype: Target quantization dtype (int8, float8_e4m3fn, etc.)
        
    Returns:
        quantized_array: Quantized array in quant_dtype
        scale: Dequantization scale of shape [batch_block_size, 1]
    """
    is_float = jnp.issubdtype(quant_dtype, jnp.floating)
    
    if is_float:
        dtype_info = jnp.finfo(quant_dtype)
        dtype_max = float(dtype_info.max)
        if quant_dtype == jnp.float8_e4m3fn:
            dtype_max = 448.0
        dtype_min = float(dtype_info.min)
    else:
        dtype_info = jnp.iinfo(quant_dtype)
        dtype_max = float(dtype_info.max)
        dtype_min = float(dtype_info.min)

    scale_basis = jnp.maximum(x_abs_max, jnp.finfo(jnp.float32).tiny)
    scale = scale_basis / dtype_max

    x_scaled = x.astype(jnp.float32) / scale

    if not is_float:
        x_scaled = jnp.round(x_scaled)
        quantized_array = jnp.clip(x_scaled, dtype_min, dtype_max).astype(quant_dtype)
    else:
        quantized_array = x_scaled.astype(quant_dtype)

    return quantized_array, scale.astype(jnp.float32)

def _get_vmem_limit(
    n_batch: int, n_out: int, n_in: int,
    batch_block_size: int, out_block_size: int, in_block_size: int,
    x_dtype: jnp.dtype, x_q_dtype: jnp.dtype, w_q_dtype: jnp.dtype,
    scale_dtype: jnp.dtype, out_dtype: jnp.dtype, acc_dtype: jnp.dtype,
    save_acc: bool, save_x_q: bool, upper_limit_bytes: int,
):
    """Calculates VMEM usage for TPU kernel compilation.
    
    Args:
        n_batch, n_out, n_in: Number of blocks in each grid dimension
        batch_block_size, out_block_size, in_block_size: Size of each block
        x_dtype, x_q_dtype, w_q_dtype: Data types for inputs
        scale_dtype, out_dtype, acc_dtype: Data types for scales and accumulators
        save_acc: Whether accumulator scratch buffer is needed
        save_x_q: Whether quantized activation scratch buffer is needed
        upper_limit_bytes: Device VMEM limit in bytes
        
    Returns:
        Estimated VMEM usage in bytes, capped at upper_limit_bytes
    """
    x_size = batch_block_size * in_block_size * dtypes.bit_width(x_dtype)
    x_abs_max_size = batch_block_size * dtypes.bit_width(scale_dtype)
    w_q_size = out_block_size * in_block_size * dtypes.bit_width(w_q_dtype)
    w_scale_size = out_block_size * dtypes.bit_width(scale_dtype)
    out_size = batch_block_size * out_block_size * dtypes.bit_width(out_dtype)

    vmem_in_out = x_size + x_abs_max_size + w_q_size + w_scale_size + out_size
    vmem_in_out *= 2

    vmem_in_out += x_size if (n_batch > 1 or n_in > 1) else 0
    vmem_in_out += x_abs_max_size if (n_batch > 1) else 0
    vmem_in_out += w_q_size if (n_out > 1 or n_in > 1) else 0
    vmem_in_out += w_scale_size if (n_out > 1) else 0
    vmem_in_out += out_size if (n_batch > 1 or n_out > 1) else 0

    acc_size = batch_block_size * out_block_size * dtypes.bit_width(acc_dtype)
    x_q_size = batch_block_size * in_block_size * dtypes.bit_width(x_q_dtype)
    x_scale_size = batch_block_size * dtypes.bit_width(scale_dtype)

    vmem_scratch = acc_size if save_acc else 0
    vmem_scratch += x_q_size + x_scale_size if save_x_q else 0
    vmem_scratch *= 2

    vmem_used = vmem_in_out + vmem_scratch
    vmem_used_bytes = vmem_used // 8
    return min(vmem_used_bytes, upper_limit_bytes)


def _validate_inputs(x, w_q, w_scale, x_abs_max, x_q_dtype, batch_block_size, out_block_size, in_block_size):
    """Validates input shapes and dtypes for 1D quantized matmul.
    
    Args:
        x: Input activations [batch_size, in_features]
        w_q: Quantized weights [out_features, in_features]
        w_scale: Weight dequantization scales [out_features] or [1, out_features]
        x_abs_max: Activation max values [1, batch_size]
        x_q_dtype: Target dtype for activation quantization
        batch_block_size, out_block_size, in_block_size: Block sizes
        
    Raises:
        ValueError: If shapes are incompatible or dtypes mismatch
    """
    if x.dtype != x_q_dtype:
        if jnp.issubdtype(x_q_dtype, jnp.integer) != jnp.issubdtype(w_q.dtype, jnp.integer):
            raise ValueError(f'{x_q_dtype=} and {w_q.dtype=} must be same int/float type.')
    if x.shape[1] != w_q.shape[1]:
        raise ValueError("x and w_q shape mismatch")
    if w_q.shape[0] != w_scale.shape[1]:
        raise ValueError("w_q and w_scale shape mismatch")
    if x_abs_max.shape != (1, x.shape[0]):
        raise ValueError("x_abs_max shape mismatch")


def _validate_inputs_2d(x, w_q, w_scale, x_abs_max, x_q_dtype, batch_block_size, out_block_size, quant_block_size):
    """Validates input shapes and dtypes for 2D quantized matmul.
    
    Args:
        x: Input activations [batch_size, in_features]
        w_q: Quantized weights [out_features, in_features]
        w_scale: Weight dequantization scales [out_features, n_in_blocks]
        x_abs_max: Activation max values per block [batch_size, n_in_blocks]
        x_q_dtype: Target dtype for activation quantization
        batch_block_size, out_block_size: Block sizes
        quant_block_size: Size of each quantization block
        
    Raises:
        ValueError: If shapes are incompatible or dtypes mismatch
    """
    if x.dtype != x_q_dtype:
        if jnp.issubdtype(x_q_dtype, jnp.integer) != jnp.issubdtype(w_q.dtype, jnp.integer):
            raise ValueError(f'{x_q_dtype=} and {w_q.dtype=} must be same int/float type.')
    
    if x.shape[1] != w_q.shape[1]:
        raise ValueError(f'{x.shape[1]=} must be equal to {w_q.shape[1]=}')
    n_in_blocks = x.shape[1] // quant_block_size

    if w_q.shape[0] != w_scale.shape[0]:
        raise ValueError(f'{w_q.shape[0]=} must be equal to {w_scale.shape[0]=}')
    if n_in_blocks != w_scale.shape[1]:
        raise ValueError(f'{n_in_blocks=} must be equal to {w_scale.shape[1]=}')

    if x_abs_max.shape != (x.shape[0], n_in_blocks):
        raise ValueError(f'{x_abs_max.shape=} must be equal to ({x.shape[0]}, {n_in_blocks})')


def _quantized_matmul_kernel_1d(
    x_ref, w_q_ref, w_scale_ref, x_abs_max_ref, out_ref,
    acc_scratch, x_q_scratch, x_scale_scratch,
    *, x_q_dtype, dot_dtype, save_acc, save_x_q
):
    """Pallas kernel for 1D (per-row) quantized matrix multiplication.
    
    Grid: (n_batch, n_out, n_in)
    - Parallel over batch dimension
    - Arbitrary scheduling over out and in dimensions
    
    Args:
        x_ref: Input activation block
        w_q_ref: Pre-quantized weight block
        w_scale_ref: Weight dequantization scales
        x_abs_max_ref: Activation max values for quantization
        out_ref: Output block to write results
        acc_scratch: Scratch buffer for partial accumulations (if save_acc)
        x_q_scratch: Scratch buffer for quantized activations (if save_x_q)
        x_scale_scratch: Scratch buffer for activation scales (if save_x_q)
        x_q_dtype: Target dtype for activation quantization
        dot_dtype: Accumulator dtype for dot product (int32 or float32)
        save_acc: Whether to accumulate across in_blocks
        save_x_q: Whether to save quantized activations (currently disabled)
    """
    out_idx, in_idx = pl.program_id(1), pl.program_id(2)
    n_in = pl.num_programs(2)
    x_ref_dtype = x_ref.dtype
    quantize_activation = x_q_dtype != x_ref.dtype

    if save_x_q:
        quant = out_idx == 0
    else:
        quant = quantize_activation

    if save_acc:
        is_first_step = (in_idx == 0)
        is_last_step = (in_idx == (n_in - 1))
    else:
        is_first_step, is_last_step = True, True

    def matmul_body(quant, is_first_step, is_last_step):
        if quantize_activation:
            if quant:
                x_q_tmp, x_scale_tmp = _quantize_array(x_ref[...], x_abs_max_ref[...], x_q_dtype)
                if save_x_q:
                    x_q_scratch[...] = x_q_tmp
                    x_scale_scratch[...] = x_scale_tmp
            else:
                x_q_tmp = x_q_scratch[...]
                if is_last_step:
                    x_scale_tmp = x_scale_scratch[...]
        else:
            x_q_tmp = x_ref[...]

        acc = jax.lax.dot_general(
            x_q_tmp, w_q_ref[...], (((1, ), (1, )), ((), ())),
            preferred_element_type=dot_dtype
        )

        if not is_first_step:
            acc += acc_scratch[...]

        if is_last_step:
            acc = acc.astype(jnp.float32)
            acc *= w_scale_ref[...]
            if quantize_activation:
                acc *= x_scale_tmp
            out_ref[...] = acc.astype(x_ref_dtype)
        else:
            acc_scratch[...] = acc

    unfold_args((quant, is_first_step, is_last_step), (), matmul_body)


@functools.partial(jax.jit, static_argnames=['x_q_dtype', 'tuned_value'])
def quantized_matmul_1d(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    w_zp=None,
    block_size=None,
    x_q_dtype=None,
    *,
    tuned_value=None
):
    """Performs 1D (per-row) quantized matrix multiplication.
    s
    Args:
        x: Input activations [batch_size, in_features]
        w_q: Pre-quantized weights [out_features, in_features]
        w_scale: Weight dequantization scales [out_features]
        w_zp: Weight zero points (not currently supported)
        block_size: Quantization block size (not used in 1D)
        x_q_dtype: Target dtype for activation quantization. If None, uses x.dtype
        tuned_value: Pre-tuned block sizes. If None, auto-selects based on shape
        
    Returns:
        Output activations [batch_size, out_features] in x.dtype
    """
    if w_zp is not None:
        raise NotImplementedError('zero_point not supported')
    if block_size is not None:
        raise NotImplementedError('block_size not supported')
    if x_q_dtype is None:
        x_q_dtype = x.dtype
    
    x_abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=False)
    x_abs_max = jnp.expand_dims(x_abs_max, axis=0)

    orig_n_batch, orig_n_in = x.shape
    orig_n_out, _ = w_q.shape

    if tuned_value is None:
        tuned_value = get_tuned_block_sizes(
            orig_n_batch, orig_n_out, orig_n_in,
            jnp.dtype(x_q_dtype).name, jnp.dtype(w_q.dtype).name
        )
    
    batch_block_size = tuned_value.batch_block_size
    out_block_size = tuned_value.out_block_size
    in_block_size = tuned_value.in_block_size

    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))
        x_abs_max = jnp.pad(x_abs_max, ((0, 0), (0, padded_n_batch - orig_n_batch)))
    
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
    if w_scale.ndim == 1:
        w_scale = jnp.expand_dims(w_scale, axis=0)

    n_batch = padded_n_batch // batch_block_size
    n_out = padded_n_out // out_block_size
    n_in = padded_n_in // in_block_size
    save_acc = n_in > 1
    
    save_x_q = False

    dot_dtype = jnp.float32
    acc_dtype = jnp.float32
    if x_q_dtype != x.dtype and jnp.issubdtype(w_q.dtype, jnp.integer):
        dot_dtype = jnp.int32
        acc_dtype = jnp.int32

    vmem_limit_bytes = _get_vmem_limit(
        n_batch, n_out, n_in, batch_block_size, out_block_size, in_block_size,
        x.dtype, x_q_dtype, w_q.dtype, jnp.float32, x.dtype, acc_dtype,
        save_acc, save_x_q, get_device_vmem_limit()
    )

    kernel = pl.pallas_call(
        functools.partial(
            _quantized_matmul_kernel_1d,
            x_q_dtype=x_q_dtype,
            dot_dtype=dot_dtype,
            save_acc=save_acc,
            save_x_q=save_x_q,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((batch_block_size, in_block_size), lambda b, o, i: (b, i)),
                pl.BlockSpec((out_block_size, in_block_size), lambda b, o, i: (o, i)),
                pl.BlockSpec((1, out_block_size), lambda b, o, i: (0, o)),
                pl.BlockSpec((1, batch_block_size), lambda b, o, i: (0, b)),
            ],
            out_specs=pl.BlockSpec((batch_block_size, out_block_size), lambda b, o, i: (b, o)),
            scratch_shapes=[
                pltpu.VMEM((batch_block_size, out_block_size), acc_dtype) if save_acc else None,
                pltpu.VMEM((batch_block_size, in_block_size), x_q_dtype) if save_x_q else None,
                pltpu.VMEM((batch_block_size, 1), jnp.float32) if save_x_q else None,
            ],
            grid=(n_batch, n_out, n_in),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=('parallel', 'arbitrary', 'arbitrary'),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )
    
    _validate_inputs(x, w_q, w_scale, x_abs_max, x_q_dtype,
                     batch_block_size, out_block_size, in_block_size)
    
    out = kernel(x, w_q, w_scale, x_abs_max)
    return out[:orig_n_batch, :orig_n_out]


def _quantized_matmul_kernel_2d(
    x_ref, w_q_ref, w_scale_ref, x_abs_max_ref, out_ref,
    acc_scratch, x_q_scratch, x_scale_scratch,
    *, x_q_dtype, dot_dtype, save_acc, save_x_q
):
    """Pallas kernel for 2D (block-wise) quantized matrix multiplication.

    Grid: (n_batch, n_out, n_in_blocks)
    - Parallel over batch dimension
    - Arbitrary scheduling over out and in_blocks dimensions
    
    Args:
        x_ref: Input activation block [batch_block_size, quant_block_size]
        w_q_ref: Pre-quantized weight block [out_block_size, quant_block_size]
        w_scale_ref: Full weight scales [n_in_blocks, out_block_size]
        x_abs_max_ref: Full activation max values [n_in_blocks, batch_block_size]
        out_ref: Output block to write results
        acc_scratch: Scratch buffer for float32 partial sums (if save_acc)
        x_q_scratch: Scratch buffer for quantized activations (if save_x_q)
        x_scale_scratch: Scratch buffer for activation scales (if save_x_q)
        x_q_dtype: Target dtype for activation quantization
        dot_dtype: Accumulator dtype for dot product
        save_acc: Whether to accumulate across in_blocks
        save_x_q: Whether to save quantized activations (currently disabled)
    """
    out_idx, in_block_idx = pl.program_id(1), pl.program_id(2)
    n_in_blocks = pl.num_programs(2)
    quantize_activation = x_q_dtype != x_ref.dtype
    
    if save_x_q:
        quant = out_idx == 0
    else:
        quant = quantize_activation

    if save_acc:
        is_first_step = (in_block_idx == 0)
        is_last_step = (in_block_idx == (n_in_blocks - 1))
    else:
        is_first_step, is_last_step = True, True

    def matmul_body(quant, is_first_step, is_last_step):
        # Extract current block's scale from full loaded array
        x_abs_max_row = x_abs_max_ref[in_block_idx]
        x_abs_max_current = x_abs_max_row[:, None]

        if quant:
             x_q_tmp, x_scale_tmp = _quantize_array_2d(
                 x_ref[...], x_abs_max_current, x_q_dtype
             )
             if save_x_q:
                 x_q_scratch[...] = x_q_tmp
                 x_scale_scratch[...] = x_scale_tmp
        else:
             x_q_tmp = x_q_scratch[...]
             x_scale_tmp = x_scale_scratch[...]

        acc = jax.lax.dot_general(
            x_q_tmp, w_q_ref[...], (((1,), (1,)), ((), ())),
            preferred_element_type=dot_dtype,
        )

        # IMMEDIATE DEQUANTIZATION (unique to 2D)
        # Extract current block's weight scale from full loaded array
        w_scale_current = w_scale_ref[in_block_idx][None, :]
        
        acc = acc.astype(jnp.float32)
        acc *= x_scale_tmp
        acc *= w_scale_current

        if not is_first_step:
            acc += acc_scratch[...]
            
        if is_last_step:
            out_ref[...] = acc.astype(x_ref.dtype)
        elif save_acc:
            acc_scratch[...] = acc

    unfold_args((quant, is_first_step, is_last_step), (), matmul_body)


@functools.partial(jax.jit, static_argnames=[
    'x_q_dtype', 'quant_block_size', 'batch_block_size', 'out_block_size'
])
def quantized_matmul_2d(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quant_block_size: int,
    x_q_dtype=None,
    *,
    batch_block_size: int = 128,
    out_block_size: int = 128
) -> jax.Array:
    """Performs 2D (block-wise) quantized matrix multiplication.
    
    Args:
        x: Input activations [batch_size, in_features]
        w_q: Pre-quantized weights [out_features, in_features]
        w_scale: Weight dequantization scales [out_features, n_in_blocks]
        quant_block_size: Size of each quantization block along in_features
        x_q_dtype: Target dtype for activation quantization. If None, uses x.dtype
        batch_block_size: Pallas block size for batch dimension (default: 128)
        out_block_size: Pallas block size for output dimension (default: 128)
        
    Returns:
        Output activations [batch_size, out_features] in x.dtype
    """
    if x_q_dtype is None:
        x_q_dtype = x.dtype
    
    orig_n_batch, orig_n_in = x.shape
    orig_n_out, _ = w_q.shape
    n_in_blocks = orig_n_in // quant_block_size

    x_blocked = x.reshape(orig_n_batch, n_in_blocks, quant_block_size)
    x_abs_max = jnp.max(jnp.abs(x_blocked), axis=-1)
    x_abs_max = x_abs_max.astype(jnp.float32)

    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))
        x_abs_max = jnp.pad(x_abs_max, ((0, padded_n_batch - orig_n_batch), (0, 0)))

    padded_n_out = next_multiple(orig_n_out, out_block_size)
    if orig_n_out < padded_n_out:
        w_q = jnp.pad(w_q, ((0, padded_n_out - orig_n_out), (0, 0)))
        w_scale = jnp.pad(w_scale, ((0, padded_n_out - orig_n_out), (0, 0)))

    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)

    # Transpose for (8, ...) tiling efficiency on TPU
    x_abs_max_t = jnp.transpose(x_abs_max)
    w_scale_t = jnp.transpose(w_scale)

    n_batch = padded_n_batch // batch_block_size
    n_out = padded_n_out // out_block_size
    save_acc = n_in_blocks > 1
    
    # save_x_q disabled: VMEM synchronization across grid instances is unsafe
    save_x_q = False

    dot_dtype = jnp.float32
    acc_dtype = jnp.float32
    if x_q_dtype != x.dtype and jnp.issubdtype(w_q.dtype, jnp.integer):
        dot_dtype = jnp.int32
        # acc_dtype remains float32: 2D kernel accumulates dequantized floats
    
    vmem_limit_bytes = get_device_vmem_limit()

    kernel = pl.pallas_call(
        functools.partial(
            _quantized_matmul_kernel_2d,
            x_q_dtype=x_q_dtype,
            dot_dtype=dot_dtype,
            save_acc=save_acc,
            save_x_q=save_x_q,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((batch_block_size, quant_block_size), lambda b, o, i: (b, i)),
                pl.BlockSpec((out_block_size, quant_block_size), lambda b, o, i: (o, i)),
                # Load full dimension to satisfy TPU static analysis constraints
                pl.BlockSpec((n_in_blocks, out_block_size), lambda b, o, i: (0, o)),
                pl.BlockSpec((n_in_blocks, batch_block_size), lambda b, o, i: (0, b)),
            ],
            out_specs=pl.BlockSpec((batch_block_size, out_block_size), lambda b, o, i: (b, o)),
            scratch_shapes=[
                pltpu.VMEM((batch_block_size, out_block_size), acc_dtype) if save_acc else None,
                pltpu.VMEM((batch_block_size, quant_block_size), x_q_dtype) if save_x_q else None,
                pltpu.VMEM((batch_block_size, 1), jnp.float32) if save_x_q else None,
            ],
            grid=(n_batch, n_out, n_in_blocks),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=('parallel', 'arbitrary', 'arbitrary'),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )

    _validate_inputs_2d(x, w_q, w_scale, x_abs_max, x_q_dtype,
                        batch_block_size, out_block_size, quant_block_size)

    out = kernel(x, w_q, w_scale_t, x_abs_max_t)
    return out[:orig_n_batch, :orig_n_out]