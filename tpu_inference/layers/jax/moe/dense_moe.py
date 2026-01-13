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

from flax import nnx
from jaxtyping import Float

class DenseMoEEngine(nnx.Module):
    """Encapsulates logic for Dense/Standard MoE forward passes."""
    def __init__(self, moe_instance: 'MoE'):
        self.m = moe_instance

    def moe_fwd(self, x_TD: Float, weights):
        x_TD = jnp.asarray(x_TD, self.m.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.m.activation_ffw_td)
        with jax.named_scope("gating"):
            gating_TEF = jnp.einsum('TD,EDF -> TEF', x_TD, self.m.kernel_gating_EDF.value)
            activated_gating_TEF = modeling_flax_utils.ACT2FN[self.m.hidden_act](gating_TEF)
        with jax.named_scope("up_projection"):
            up_proj_TEF = jnp.einsum('TD,EDF -> TEF', x_TD, self.m.kernel_up_proj_EDF.value)
        fuse_TEF = activated_gating_TEF * up_proj_TEF
        with jax.named_scope("down_projection"):
            down_proj_TED = jnp.einsum('TEF,EFD -> TED', fuse_TEF, self.m.kernel_down_proj_EFD.value)
        with jax.named_scope("sum"):
            output_TD = jnp.einsum('TED,TE -> TD', down_proj_TED, weights)
        return output_TD.astype(self.m.dtype)

    def moe_fwd_preapply_router_weights(self, x_TD: jax.Array, weights_TE):
        num_experts = weights_TE.shape[-1]
        x_TED = jnp.repeat(x_TD[:, None, :], num_experts, 1)
        x_TED = jnp.asarray(x_TED, self.m.dtype) * weights_TE[..., None]
        x_TED = nnx.with_sharding_constraint(x_TED, self.m.activation_ffw_ted)
        
        with jax.named_scope("gating"):
            gating_TEF = jnp.einsum('TED,EDF -> TEF', x_TED, self.m.kernel_gating_EDF.value)
            activated_gating_TEF = modeling_flax_utils.ACT2FN[self.m.hidden_act](gating_TEF)
        with jax.named_scope("up_projection"):
            up_proj_TEF = jnp.einsum('TED,EDF -> TEF', x_TED, self.m.kernel_up_proj_EDF.value)
        
        fuse_TEF = activated_gating_TEF * up_proj_TEF
        with jax.named_scope("down_projection"):
            down_proj_TED = jnp.einsum('TEF,EFD -> TED', fuse_TEF, self.m.kernel_down_proj_EFD.value)
        return down_proj_TED.sum(axis=1).astype(self.m.dtype)
