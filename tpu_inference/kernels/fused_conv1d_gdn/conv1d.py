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
    metadata_ref: ref_classes.MetadataRef,
    b_start: jax.Array,
    lhs: jax.Array,
    states_ref: jax.Array,
    conv_weights_ref: ref_classes.ConvWeightsRef,
    cfgs: configs.GDNConfigs,
):
    out_list = []
    for idx in range(cfgs.tile_size):
        b_idx = b_start + idx

        s_idx = metadata_ref.b_idx_to_s_idx[b_idx]
        sz_from_old = metadata_ref.b_idx_to_sz_from_old[b_idx]
        has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]

        out = jnp.zeros((1, cfgs.dim_size), jnp.float32)

        end_idx = idx + cfgs.prev_kernel_size
        start_idx = 1 + end_idx - cfgs.kernel_size
        for k in range(cfgs.kernel_size):
            lhs_curr = lhs[start_idx + k]

            if k < cfgs.prev_kernel_size:
                conv_state = states_ref[idx, k]
                conv_state = jnp.where(has_initial_state, conv_state, 0)
                lhs_curr = jnp.where(k < sz_from_old, conv_state, lhs_curr)

            if k > 0:
                states_ref[idx, k - 1] = lhs_curr

            rhs = conv_weights_ref.weight[k]
            out += lhs_curr * rhs

        if (bias_ref := conv_weights_ref.bias) is not None:
            out += bias_ref[...].reshape(1, -1)
        out_list.append(out)

    return jnp.stack(out_list, axis=0)
