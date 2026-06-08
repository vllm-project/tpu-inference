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
""" TPU implementation of the StreamIndex algorithm for the 
DeepSeek-V4 Lightning Indexer. Optimized for vLLM, TorchAX, and JAX compilation.
"""

import jax
import jax.numpy as jnp


@jax.jit(static_argnames=["k", "compression_ratio", "c_s", "c_t"])
def streamindex_chunked_topk(
    query_projection: jax.Array,  # Shape: [num_tokens, H_I, D] (Flattened Q)
    kv_cache: jax.Array,          # Shape: [num_pages, page_size, H_KV, D]
    block_table: jax.Array,       # Shape: [B, max_blocks]
    seq_lens: jax.Array,          # Shape: [B] (Number of tokens currently in context)
    cu_seq_lens: jax.Array,       # Shape: [B+1] (Cumulative sum of query lengths)
    indexer_weights: jax.Array,   # Shape: [num_tokens, H_I]
    k: int,
    compression_ratio: int,
    c_s: int = 8192,
    c_t: int = 2048,
) -> jax.Array:
    """
    Executes a chunked, streaming top-k selection natively in JAX.
    Replaces vmap with scan to avoid TPU OOM, and eliminates row-by-row scatter loops.
    """
    num_tokens, H_I, D = query_projection.shape
    page_size = kv_cache.shape[1]
    seq_lens = seq_lens.astype(jnp.int32)
    cu_seq_lens = cu_seq_lens.astype(jnp.int32)
    block_table = block_table.astype(jnp.int32)

    assert c_s % page_size == 0, (
        f"c_s ({c_s}) must be a multiple of page_size ({page_size}) for bulk KV"
        " gathering"
    )
    B = seq_lens.shape[0]

    num_t_chunks_global = (num_tokens + c_t - 1) // c_t
    num_tokens_padded = num_t_chunks_global * c_t

    pad_len = num_tokens_padded

    q_proj_padded = jnp.pad(
        query_projection, ((0, pad_len), (0, 0), (0, 0)), constant_values=0.0
    )
    w_proj_padded = jnp.pad(
        indexer_weights, ((0, pad_len), (0, 0)), constant_values=0.0
    )

    final_indices_init = jnp.full((num_tokens, k), -1, dtype=jnp.int32)

    def batch_step(carry_output, b_idx):
        q_start = cu_seq_lens[b_idx]
        q_end = cu_seq_lens[b_idx + 1]
        T_seq = q_end - q_start
        S_valid = seq_lens[b_idx] // compression_ratio

        q_seq = jax.lax.dynamic_slice(
            q_proj_padded, (q_start, 0, 0), (num_tokens_padded, H_I, D)
        )
        w_seq = jax.lax.dynamic_slice(
            w_proj_padded, (q_start, 0), (num_tokens_padded, H_I)
        )

        init_values = jnp.full((num_tokens_padded, k), -jnp.inf, dtype=jnp.float32)
        init_indices = jnp.full((num_tokens_padded, k), -1, dtype=jnp.int32)

        def t_loop_body(t_idx, t_state):
            val_acc_global, idx_acc_global = t_state

            queries_chunk = jax.lax.dynamic_slice_in_dim(
                q_seq, t_idx * c_t, c_t, axis=0
            )
            weights_chunk = jax.lax.dynamic_slice_in_dim(
                w_seq, t_idx * c_t, c_t, axis=0
            )

            t_absolute = jnp.arange(c_t, dtype=jnp.int32)[:, None] + (t_idx * c_t)
            valid_t_mask = t_absolute < T_seq
            q_pos_absolute = (seq_lens[b_idx] - T_seq) + t_absolute

            local_val_acc = jnp.full((c_t, k), -jnp.inf, dtype=jnp.float32)
            local_idx_acc = jnp.full((c_t, k), -1, dtype=jnp.int32)

            def s_loop_body(s_idx, s_state):
                val_acc, idx_acc = s_state

                logical_indices = jnp.arange(c_s, dtype=jnp.int32) + (s_idx * c_s)
                valid_s_mask = logical_indices < S_valid
                safe_logical = jnp.where(valid_s_mask, logical_indices, 0)

                block_indices = safe_logical[::page_size] // page_size

                physical_pages = block_table[b_idx, block_indices]
                keys_chunk_blocks = kv_cache[physical_pages]

                keys_chunk = keys_chunk_blocks.reshape((c_s, kv_cache.shape[2], D))
                keys_chunk = jnp.where(valid_s_mask[:, None, None], keys_chunk, 0.0)

                H_KV = keys_chunk.shape[1]
                if H_KV != H_I:
                    keys_chunk = jnp.repeat(keys_chunk, H_I // H_KV, axis=1)

                inner_dot = jnp.einsum("thd,shd->ths", queries_chunk, keys_chunk)

                scores_multihead = jax.nn.relu(inner_dot) * jnp.expand_dims(
                    weights_chunk, axis=-1
                )
                scores_tile = jnp.sum(scores_multihead, axis=1)

                s_absolute = jnp.arange(c_s, dtype=jnp.int32)[None, :] + (s_idx * c_s)
                causal_mask = (s_absolute * compression_ratio) <= q_pos_absolute

                combined_mask = causal_mask & valid_t_mask & valid_s_mask[None, :]
                scores_tile = jnp.where(combined_mask, scores_tile, -jnp.inf)

                local_vals, local_ids = jax.lax.top_k(scores_tile, k)

                global_ids_tile = jnp.where(
                    local_vals > -jnp.inf, local_ids + (s_idx * c_s), -1
                )

                merged_vals = jnp.concatenate([val_acc, local_vals], axis=-1)
                merged_ids = jnp.concatenate([idx_acc, global_ids_tile], axis=-1)

                new_vals_tile, local_topk_ids = jax.lax.top_k(merged_vals, k)
                new_ids_tile = jnp.take_along_axis(merged_ids, local_topk_ids, axis=-1)

                return new_vals_tile, new_ids_tile

            # Calculate dynamic loop bound for the valid KV cache size of this sequence
            num_s_chunks_local = (S_valid + c_s - 1) // c_s

            final_local_vals, final_local_ids = jax.lax.fori_loop(
                0, num_s_chunks_local, s_loop_body, (local_val_acc, local_idx_acc)
            )

            new_val_acc_global = jax.lax.dynamic_update_slice_in_dim(
                val_acc_global, final_local_vals, t_idx * c_t, axis=0
            )
            new_idx_acc_global = jax.lax.dynamic_update_slice_in_dim(
                idx_acc_global, final_local_ids, t_idx * c_t, axis=0
            )

            return new_val_acc_global, new_idx_acc_global

        # Calculate dynamic loop bound for the queries in this sequence
        num_t_chunks_local = (T_seq + c_t - 1) // c_t

        _, final_batch_indices = jax.lax.fori_loop(
            0, num_t_chunks_local, t_loop_body, (init_values, init_indices)
        )

        global_indices = jnp.arange(num_tokens, dtype=jnp.int32)
        global_valid_mask = (global_indices >= q_start) & (global_indices < q_end)

        source_indices = global_indices - q_start
        safe_source_indices = jnp.where(global_valid_mask, source_indices, 0)

        shifted_source = final_batch_indices[safe_source_indices]

        updated_output = jnp.where(
            global_valid_mask[:, None], shifted_source, carry_output
        )

        return updated_output, None

    final_indices, _ = jax.lax.scan(
        batch_step, final_indices_init, jnp.arange(B, dtype=jnp.int32)
    )
    return final_indices