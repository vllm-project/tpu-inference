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

from tpu_inference.layers.jax.moe.utils import (get_all_to_all_params_fn,
                                                global_permute_fn, gmm_fn,
                                                local_permute_fn,
                                                modeling_flax_utils,
                                                sort_activations_fn,
                                                unpermute_fn)

# yapf: enable


def sparse_moe_distributed_fwd(
    moe_instance,
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
                          moe_instance.num_experts_per_tok,
                          moe_instance.num_local_experts)

    if moe_instance.num_expert_parallelism > 1:
        expert_shard_id = jax.lax.axis_index(moe_instance.expert_axis_name)
        local_expert_size = moe_instance.num_local_experts // moe_instance.num_expert_parallelism

        if moe_instance.is_batch_sharded_by_expert:
            # 2a. Send Tokens To Experts (All-to-All)
            all_shards_group_sizes = jax.lax.all_gather(
                global_group_sizes, axis_name=moe_instance.data_axis_name)

            all_shards_group_sizes_per_expert_shard = jnp.sum(
                all_shards_group_sizes.reshape(
                    moe_instance.num_expert_parallelism,
                    moe_instance.num_expert_parallelism, local_expert_size),
                axis=2)

            input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params_fn(
                all_shards_group_sizes_per_expert_shard, expert_shard_id,
                moe_instance.num_expert_parallelism)

            local_total_assignments = x_TD.shape[
                0] * moe_instance.num_experts_per_tok
            global_total_assignments = local_total_assignments * moe_instance.num_expert_parallelism
            output_shape_est = jnp.zeros(
                (global_total_assignments, moe_instance.hidden_size),
                dtype=sorted_inputs.dtype)

            inputs_after_all2all = jax.lax.ragged_all_to_all(
                sorted_inputs,
                output_shape_est,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name=moe_instance.expert_axis_name)

            # 3a. Local Permute
            full_global_group_sizes = jax.lax.all_gather(
                global_group_sizes, axis_name=moe_instance.expert_axis_name)
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
        gating_TEF = gmm_fn(compute_inputs, kernel_gating, compute_group_sizes,
                            moe_instance.tile_size, moe_instance.moe_backend,
                            moe_instance.dtype, moe_instance.quantized_dtype)
        activated_gating_TEF = modeling_flax_utils.ACT2FN[
            moe_instance.hidden_act](gating_TEF)

    with jax.named_scope("up_projection"):
        up_proj_TEF = gmm_fn(compute_inputs, kernel_up_proj,
                             compute_group_sizes, moe_instance.tile_size,
                             moe_instance.moe_backend, moe_instance.dtype,
                             moe_instance.quantized_dtype)

    fuse_TEF = activated_gating_TEF * up_proj_TEF

    with jax.named_scope("down_projection"):
        intermediate_output = gmm_fn(fuse_TEF, kernel_down_proj,
                                     compute_group_sizes,
                                     moe_instance.tile_size,
                                     moe_instance.moe_backend,
                                     moe_instance.dtype,
                                     moe_instance.quantized_dtype)

    # 5. Return Results (All-to-All)
    if moe_instance.num_expert_parallelism > 1:
        local_total_assignments = x_TD.shape[
            0] * moe_instance.num_experts_per_tok
        output_shape = jnp.zeros(
            (local_total_assignments, moe_instance.hidden_size),
            dtype=intermediate_output.dtype)

        if moe_instance.is_batch_sharded_by_expert:
            local_output = sort_activations_fn(
                intermediate_output, jnp.argsort(local_sorted_indices))

            input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params_fn(
                jnp.transpose(all_shards_group_sizes),
                expert_shard_id,
                moe_instance.num_expert_parallelism,
            )
            final_intermediate_output = jax.lax.ragged_all_to_all(
                local_output,
                output_shape,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name=moe_instance.expert_axis_name)
        else:
            input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params_fn(
                reshaped_group_sizes,
                expert_shard_id,
                moe_instance.num_expert_parallelism,
                is_batch_sharded=False,
            )
            final_intermediate_output = jax.lax.ragged_all_to_all(
                intermediate_output,
                output_shape,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name=moe_instance.expert_axis_name)
    else:
        final_intermediate_output = intermediate_output

    # 6. Global Unpermute
    with jax.named_scope("unpermute"):
        output_TD = unpermute_fn(final_intermediate_output,
                                 global_sort_indices, router_weights_TX,
                                 moe_instance.num_experts_per_tok,
                                 moe_instance.dtype)

    return output_TD
