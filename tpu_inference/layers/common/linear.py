# Copyright 2025 Google LLC
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

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference import envs
from tpu_inference.kernels.quantized_matmul.blockwise_kernel import \
    quantized_matmul_kernel as blockwise_quantized_matmul_kernel
from tpu_inference.kernels.quantized_matmul.kernel import \
    quantized_matmul_kernel
from tpu_inference.kernels.quantized_matmul.util import xla_quantized_matmul


def _get_x_q_dtype(w_q_dtype: jnp.dtype) -> jnp.dtype:
    """Return 8-bit float or integer dtype depending on w_q_dtype."""
    if jnp.issubdtype(w_q_dtype, jnp.integer):
        return jnp.int8
    elif jnp.issubdtype(w_q_dtype, jnp.floating):
        return jnp.float8_e4m3fn
    # TODO: we need a new flag for 4bit activation later such as w4a4.
    else:
        raise ValueError(
            f"Unsupported quantized dtype: {w_q_dtype}, it should be integer or float"
        )


def sharded_quantized_matmul(x: jax.Array,
                             w_q: jax.Array,
                             w_s: jax.Array,
                             weight_sharding: P,
                             *,
                             mesh: Mesh | None = None,
                             block_size: i    nt | None = None) -> jax.Array:
    """
    Wrapper around the quantized matmul kernel.

    Args:
        x:  Activation.
        w_q: Weight quantized array. [n_output_features, n_input_features]
        w_s: Weight quantization scale. [n_output_features]
        weight_sharding: PartitionSpec for the weight tensor.
        mesh: (Optional) Mesh to shard on. If None, mesh from current context is used, similar to jax.shard_map().
        block_size: Block size for blockwise quantized matmul kernel.

    Returns:
        Output of the quantized matmul.
    """

    # NOTE (jacobplatin/kyuyeunk) there have been numeric issues (concerning) NaNs
    # with the kernel and thus we disable it for now.
    out_axis, in_axis = weight_sharding
    x_sharding = P(None, in_axis)
    if block_size:
        scale_sharding = P(
            in_axis,
            None,
            out_axis,
        )
    else:
        scale_sharding = P(out_axis, )
    out_sharding = P(None, out_axis)

    x_q_dtype = _get_x_q_dtype(w_q.dtype)
    x = jax.lax.with_sharding_constraint(
        x,
        NamedSharding(mesh, x_sharding) if mesh else x_sharding)

    def wrapper(x, w_q, w_s):
        if envs.ENABLE_QUANTIZED_MATMUL_KERNEL:
            if block_size:
                output = blockwise_quantized_matmul_kernel(
                    x, w_q, w_s, x_q_dtype=x_q_dtype, block_size=block_size)
            else:
                output = quantized_matmul_kernel(x, w_q, w_s, x_q_dtype=w_q.dtype)
        else:
            output = xla_quantized_matmul(x, w_q, w_s)
        if in_axis:
            output = jax.lax.psum(output, axis_name=in_axis)
        return output

    return jax.shard_map(
        wrapper,
        mesh=mesh,
        in_specs=(x_sharding, weight_sharding, scale_sharding),
        out_specs=(out_sharding),
        check_vma=False,
    )(x, w_q, w_s)
