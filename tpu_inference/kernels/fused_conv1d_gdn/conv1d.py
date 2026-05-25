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

import jax
import jax.numpy as jnp

from tpu_inference.kernels.fused_conv1d_gdn import configs, ref_classes


def causal_conv1d(
    lhs: jax.Array,
    conv_weights_ref: ref_classes.ConvWeightsRef,
    cfgs: configs.GDNConfigs,
) -> jax.Array:
    assert lhs.ndim == 4

    out_list = []

    for t_idx in range(cfgs.tok_tile_size):
        out = jnp.zeros((cfgs.seq_tile_size, 1, cfgs.dim_size), jnp.float32)

        end_idx = t_idx + cfgs.prev_kernel_size
        start_idx = 1 + end_idx - cfgs.kernel_size
        for k in range(cfgs.kernel_size):
            lhs_curr = lhs[:, start_idx + k]
            rhs = conv_weights_ref.weight[k:k + 1]
            out += lhs_curr * rhs

        if (bias_ref := conv_weights_ref.bias) is not None:
            out += bias_ref[...].reshape(1, 1, -1)

        out_list.append(out)

    return jnp.stack(out_list, axis=1)
