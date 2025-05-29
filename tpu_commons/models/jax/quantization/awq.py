from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import Mesh
from transformers import modeling_flax_utils

from tpu_commons.models.jax.param_init import sharding_init

AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack(matrix: jnp.ndarray, bits: int):
    """Unpacks AWQ weights from int32 to int4 values inside int8 datatype.
    Converted to JAX from https://github.com/casper-hansen/AutoAWQ/blob/f0321eedca887c12680553fc561d176b03b1b9a5/awq/utils/packing_utils.py#L8
    """

    shifts = jnp.arange(0, 32, bits)
    # zero-point quantization will need uint4 otherwise int4.
    # https://github.com/casper-hansen/AutoAWQ/blob/f0321eedca887c12680553fc561d176b03b1b9a5/awq/quantize/quantizer.py#L76
    iweights = jnp.right_shift(matrix[..., None],
                               shifts[None, None, :]).astype(jnp.uint4)
    return iweights.reshape(iweights.shape[0], -1)


def reverse_awq_order(matrix: jnp.ndarray, bits: int):
    """Reorders the packed int4 values.
    Converted to JAX from https://github.com/casper-hansen/AutoAWQ/blob/f0321eedca887c12680553fc561d176b03b1b9a5/awq/utils/packing_utils.py#L29
    """
    reverse_order_tensor = jnp.arange(matrix.shape[-1], dtype=jnp.int32)
    reverse_order_tensor = reverse_order_tensor.reshape(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:,
                                                jnp.array(AWQ_REVERSE_ORDER)]
    reverse_order_tensor = reverse_order_tensor.reshape(-1)

    return matrix[:, reverse_order_tensor]


def dequantize(
    imatrix: jnp.ndarray,
    scales: jnp.ndarray,
    zeros: jnp.ndarray,
    group_size: int,
    axis_to_repeat: int = 0,
    out_dtype: jnp.dtype = None,
):
    """Dequantizes a 4-bit integer matrix into a float matrix using.
    Converted to jax from https://github.com/casper-hansen/AutoAWQ/blob/f0321eedca887c12680553fc561d176b03b1b9a5/awq/utils/packing_utils.py#L87
    Args:
        imatrix: JAX array of 4-bit integers.
        scales: JAX array of 16-bit floats (scale factors).
        zeros: JAX array of 4-bit integers (zero points).
        group_size: The group size for repetition.
        axis_to_repeat: Axis to repeat the groups.
        out_dtype: Preferred dtype of output.

    Returns:
        A JAX array of dequantized 16-bit floats.
    """
    repeated_scales = jnp.repeat(scales, group_size, axis=axis_to_repeat)
    repeated_zeros = jnp.repeat(zeros, group_size, axis=axis_to_repeat)

    # Dequantize the matrix
    # Prevent overflow by upcasting before subtracting.
    # TODO: Will it make it faster (and still accurate) if we cast to bf16 here for faster matmul, instead of cast at the end?
    fmatrix = (imatrix.astype(repeated_scales.dtype) -
               repeated_zeros.astype(repeated_scales.dtype)) * repeated_scales

    if out_dtype is not None:
        fmatrix = fmatrix.astype(out_dtype)

    return fmatrix


class Int4Einsum(nn.Module):
    shape: Tuple[int, ...]
    dtype: jnp.dtype
    named_axes: Tuple[str, ...]
    mesh: Mesh
    hidden_dim: int
    group_size: int = 128

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        w = self.param(
            "qweight",
            sharding_init(self.named_axes, self.mesh, use_constant=True),
            self.shape,
            jnp.uint4,
        )

        z_shape = (self.shape[0], self.shape[1] // self.group_size,
                   self.shape[2])
        z = self.param(
            "qzeros",
            sharding_init(self.named_axes, self.mesh, use_constant=True),
            z_shape,
            jnp.uint4,
        )
        s = self.param(
            "scales",
            sharding_init(self.named_axes, self.mesh, use_constant=True),
            z_shape,
            self.dtype,
        )

        scaled_weights = dequantize(w,
                                    s,
                                    z,
                                    self.group_size,
                                    axis_to_repeat=1,
                                    out_dtype=self.dtype)
        result = jnp.einsum(eqn, x, scaled_weights)

        assert result.dtype == self.dtype

        return result


class Int4EinsumBias(nn.Module):
    shape: Tuple[int, ...]
    dtype: jnp.dtype
    named_axes: Tuple[str, ...]
    mesh: Mesh
    hidden_dim: int
    group_size: int = 128

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        w = self.param(
            "qweight",
            sharding_init(self.named_axes, self.mesh, use_constant=True),
            self.shape,
            jnp.uint4,
        )

        b = self.param(
            "bias",
            sharding_init((self.named_axes[0], self.named_axes[2]), self.mesh),
            (self.shape[0], self.shape[2]),
            self.dtype,
        )

        z_shape = (self.shape[0], self.shape[1] // self.group_size,
                   self.shape[2])
        z = self.param(
            "qzeros",
            sharding_init(self.named_axes, self.mesh, use_constant=True),
            z_shape,
            jnp.uint4,
        )
        s = self.param(
            "scales",
            sharding_init(self.named_axes, self.mesh, use_constant=True),
            z_shape,
            self.dtype,
        )

        scaled_weights = dequantize(w,
                                    s,
                                    z,
                                    self.group_size,
                                    axis_to_repeat=1,
                                    out_dtype=self.dtype)
        result = jnp.einsum(eqn, x, scaled_weights)

        assert result.dtype == self.dtype

        return result, jnp.asarray(b)


class Int4MLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    dtype: jnp.dtype
    mesh: Mesh
    act: str
    group_size: int = 128

    @nn.compact
    def __call__(self, x) -> jax.Array:
        # gate proj
        gate_proj_named_axes = (None, "model")
        gate_proj_shape = (self.hidden_size, self.intermediate_size)
        gate_proj_grouped_shape = (
            self.hidden_size // self.group_size,
            self.intermediate_size,
        )

        gate_proj_w = self.param(
            "gate_proj_qweight",
            sharding_init(gate_proj_named_axes, self.mesh, use_constant=True),
            gate_proj_shape,
            jnp.uint4,
        )
        gate_proj_z = self.param(
            "gate_proj_qzeros",
            sharding_init(gate_proj_named_axes, self.mesh, use_constant=True),
            gate_proj_grouped_shape,
            jnp.uint4,
        )
        gate_proj_s = self.param(
            "gate_proj_scales",
            sharding_init(gate_proj_named_axes, self.mesh, use_constant=True),
            gate_proj_grouped_shape,
            self.dtype,
        )

        # up proj
        up_proj_named_axes = gate_proj_named_axes
        up_proj_shape = gate_proj_shape
        gate_proj_grouped_shape = gate_proj_grouped_shape
        up_proj_w = self.param(
            "up_proj_qweight",
            sharding_init(up_proj_named_axes, self.mesh, use_constant=True),
            up_proj_shape,
            jnp.uint4,
        )
        up_proj_z = self.param(
            "up_proj_qzeros",
            sharding_init(up_proj_named_axes, self.mesh, use_constant=True),
            gate_proj_grouped_shape,
            jnp.uint4,
        )
        up_proj_s = self.param(
            "up_proj_scales",
            sharding_init(up_proj_named_axes, self.mesh, use_constant=True),
            gate_proj_grouped_shape,
            self.dtype,
        )

        # down proj
        down_proj_named_axes = ("model", None)
        down_proj_shape = (self.intermediate_size, self.hidden_size)
        down_proj_grouped_shape = (
            self.intermediate_size // self.group_size,
            self.hidden_size,
        )
        down_proj_w = self.param(
            "down_proj_qweight",
            sharding_init(down_proj_named_axes, self.mesh, use_constant=True),
            down_proj_shape,
            jnp.uint4,
        )
        down_proj_z = self.param(
            "down_proj_qzeros",
            sharding_init(down_proj_named_axes, self.mesh, use_constant=True),
            down_proj_grouped_shape,
            jnp.uint4,
        )
        down_proj_s = self.param(
            "down_proj_scales",
            sharding_init(down_proj_named_axes, self.mesh, use_constant=True),
            down_proj_grouped_shape,
            self.dtype,
        )

        gate_proj = dequantize(
            gate_proj_w,
            gate_proj_s,
            gate_proj_z,
            self.group_size,
            axis_to_repeat=0,
            out_dtype=self.dtype,
        )
        up_proj = dequantize(
            up_proj_w,
            up_proj_s,
            up_proj_z,
            self.group_size,
            axis_to_repeat=0,
            out_dtype=self.dtype,
        )
        down_proj = dequantize(
            down_proj_w,
            down_proj_s,
            down_proj_z,
            self.group_size,
            axis_to_repeat=0,
            out_dtype=self.dtype,
        )

        gate = jnp.dot(x, gate_proj)
        gate = modeling_flax_utils.ACT2FN[self.act](gate)
        up = jnp.dot(x, up_proj)
        fuse = gate * up
        outputs = jnp.dot(fuse, down_proj)

        assert outputs.dtype == self.dtype

        return outputs
