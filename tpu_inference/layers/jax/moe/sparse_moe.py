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
# yapf: disable
import jax
import jax.numpy as jnp
from flax import nnx

from tpu_inference.layers.jax.moe.utils import (get_all_to_all_params_fn,
                                                global_permute_fn, gmm_fn,
                                                local_permute_fn,
                                                modeling_flax_utils,
                                                sort_activations_fn,
                                                unpermute_fn)

# yapf: enable


class SparseMoEEngine(nnx.Module):
    """Encapsulates logic for Sparse/Grouped Matrix Multiply MoE."""

    def __init__(self, moe_instance):
        self.m = moe_instance

    def distributed_fwd(
        self,
        x_TD: jax.Array,
        router_weights_TX: jax.Array,
        selected_experts_TX: jax.Array,
        kernel_gating: jax.Array,
        kernel_up_proj: jax.Array,
        kernel_down_proj: jax.Array,
    ):
        """
        The sparse MoE forward pass with fully distributed logic.
        This assumes it is running within a distributed TPU.
        """

        # 1. Global Permute
        (
            sorted_inputs,
            global_sort_indices,
            global_group_sizes,
            global_sorted_experts,
        ) = global_permute_fn(x_TD, selected_experts_TX,
                              self.m.num_experts_per_tok,
                              self.m.num_local_experts)

        expert_shard_id = jax.lax.axis_index(self.m.expert_axis_name)
        local_expert_size = self.m.num_local_experts // self.m.num_expert_parallelism

        if self.m.num_expert_parallelism > 1:
            if self.m.is_batch_sharded_by_expert:
                # 2a. Send Tokens To Experts (All-to-All)
                all_shards_group_sizes = jax.lax.all_gather(
                    global_group_sizes, axis_name=self.m.data_axis_name)

                all_shards_group_sizes_per_expert_shard = jnp.sum(
                    all_shards_group_sizes.reshape(
                        self.m.num_expert_parallelism,
                        self.m.num_expert_parallelism, local_expert_size),
                    axis=2)

                input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params_fn(
                    all_shards_group_sizes_per_expert_shard, expert_shard_id,
                    self.m.num_expert_parallelism)

                local_total_assignments = x_TD.shape[
                    0] * self.m.num_experts_per_tok
                global_total_assignments = local_total_assignments * self.m.num_expert_parallelism
                output_shape_est = jnp.zeros(
                    (global_total_assignments, self.m.hidden_size),
                    dtype=sorted_inputs.dtype)

                inputs_after_all2all = jax.lax.ragged_all_to_all(
                    sorted_inputs,
                    output_shape_est,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name=self.m.expert_axis_name)

                # 3a. Local Permute
                full_global_group_sizes = jax.lax.all_gather(
                    global_group_sizes, axis_name=self.m.expert_axis_name)
                (
                    compute_inputs,
                    local_sorted_indices,
                    compute_group_sizes,
                    compute_expert_ids,
                ) = local_permute_fn(
                    inputs_after_all2all,
                    full_global_group_sizes,
                    local_expert_size,
                    shard_index=expert_shard_id,
                    is_offset=False,
                )

            else:
                # 3b. Local "Permute" for Replicated tokens
                (
                    compute_inputs,
                    local_sorted_indices,
                    compute_group_sizes,
                    compute_expert_ids,
                ) = local_permute_fn(
                    sorted_inputs,
                    global_group_sizes[None, :],
                    local_expert_size,
                    shard_index=expert_shard_id,
                    is_offset=True,
                    global_sorted_experts=global_sorted_experts,
                )

                reshaped_group_sizes = jnp.sum(global_group_sizes.reshape(
                    -1, local_expert_size),
                                               axis=1)
                mask = compute_expert_ids < local_expert_size
                compute_inputs = compute_inputs * mask[..., None]

        else:
            # --- NO EXPERT PARALLELISM ---
            compute_inputs = sorted_inputs
            compute_group_sizes = global_group_sizes
            compute_expert_ids = global_sorted_experts
            local_sorted_indices = jnp.arange(sorted_inputs.shape[0])

        # 4. Compute: Apply experts using Grouped Matrix Multiply
        with jax.named_scope("gating"):
            gating_TEF = gmm_fn(compute_inputs, kernel_gating,
                                compute_group_sizes, self.m.tile_size,
                                self.m.moe_backend, self.m.dtype,
                                self.m.quantized_dtype)
            activated_gating_TEF = modeling_flax_utils.ACT2FN[
                self.m.hidden_act](gating_TEF)

        with jax.named_scope("up_projection"):
            up_proj_TEF = gmm_fn(compute_inputs, kernel_up_proj,
                                 compute_group_sizes, self.m.tile_size,
                                 self.m.moe_backend, self.m.dtype,
                                 self.m.quantized_dtype)

        fuse_TEF = activated_gating_TEF * up_proj_TEF

        with jax.named_scope("down_projection"):
            intermediate_output = gmm_fn(fuse_TEF, kernel_down_proj,
                                         compute_group_sizes, self.m.tile_size,
                                         self.m.moe_backend, self.m.dtype,
                                         self.m.quantized_dtype)

        # 5. Return Results (All-to-All)
        if self.m.num_expert_parallelism > 1:
            local_total_assignments = x_TD.shape[0] * self.m.num_experts_per_tok
            output_shape = jnp.zeros(
                (local_total_assignments, self.m.hidden_size),
                dtype=intermediate_output.dtype)

            if self.m.is_batch_sharded_by_expert:
                local_output = sort_activations_fn(
                    intermediate_output, jnp.argsort(local_sorted_indices))

                input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params_fn(
                    jnp.transpose(all_shards_group_sizes),
                    expert_shard_id,
                    self.m.num_expert_parallelism,
                )
                final_intermediate_output = jax.lax.ragged_all_to_all(
                    local_output,
                    output_shape,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name=self.m.expert_axis_name)
            else:
                input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params_fn(
                    reshaped_group_sizes,
                    expert_shard_id,
                    self.m.num_expert_parallelism,
                    is_batch_sharded=False,
                )
                final_intermediate_output = jax.lax.ragged_all_to_all(
                    intermediate_output,
                    output_shape,
                    input_offsets,
                    send_sizes,
                    output_offsets,
                    recv_sizes,
                    axis_name=self.m.expert_axis_name)
        else:
            final_intermediate_output = intermediate_output

        # 6. Global Unpermute
        with jax.named_scope("unpermute"):
            output_TD = unpermute_fn(final_intermediate_output,
                                     global_sort_indices, router_weights_TX,
                                     self.m.num_experts_per_tok,
                                     self.m.hidden_size, self.m.dtype)

        return output_TD
