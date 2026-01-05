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
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference import envs
from tpu_inference.kernels.quantized_matmul.kernel import (
    quantized_matmul_kernel, xla_quantized_matmul)


def sharded_quantized_matmul(x: jax.Array, w_q: jax.Array, w_s: jax.Array,
                             mesh: Mesh, weight_sharding: P) -> jax.Array:
    """
    Wrapper around the quantized matmul kernel.

    Args:
        x:  Activation.
        w_q: Weight quantized array. [n_output_features, n_input_features]
        w_s: Weight quantization scale. [n_output_features]
        mesh: Mesh to shard on.
        weight_sharding: PartitionSpec for the weight tensor.

    Returns:
        Output of the quantized matmul.
    """

    # NOTE (jacobplatin/kyuyeunk) there have been numeric issues (concerning) NaNs
    # with the kernel and thus we disable it for now.
    if envs.ENABLE_QUANTIZED_MATMUL_KERNEL:
        out_axis, in_axis = weight_sharding
        x_sharding = P(None, in_axis)
        scale_sharding = P(out_axis, )
        out_sharding = P(None, out_axis)

        x = jax.lax.with_sharding_constraint(x,
                                             NamedSharding(mesh, x_sharding))

        def wrapper(x, w_q, w_s):
            output = quantized_matmul_kernel(x, w_q, w_s, x_q_dtype=w_q.dtype)
            if in_axis:
                output = jax.lax.psum(output, axis_name=in_axis)
            return output

        return jax.shard_map(wrapper,
                             mesh=mesh,
                             in_specs=(x_sharding, weight_sharding,
                                       scale_sharding),
                             out_specs=(out_sharding),
                             check_vma=False)(x, w_q, w_s)
    else:
        return xla_quantized_matmul(x, w_q, w_s)
