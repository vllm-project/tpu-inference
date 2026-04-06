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
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# Import the underlying kernel directly
from tpu_inference.kernels.mla.v2.kernel import mla_ragged_paged_attention


def main():
    # --- 1. Set up the JAX Mesh ---
    # Based on your log: Mesh('data': 1, 'model': 8, ...)
    devices = jax.devices()
    mesh_shape = (1, len(devices)
                  )  # Adapts gracefully to however many devices are visible
    mesh = Mesh(np.array(devices).reshape(mesh_shape), ('data', 'model'))
    print(f"Running on mesh: {mesh}")

    # --- 2. Load the saved tensors ---
    print("Loading saved .npy files...")
    # Loading into jnp.array explicitly moves them to JAX arrays
    q_TNA = jnp.array(jnp.load("q_TNA.npy"))
    q_rope_TNH = jnp.array(jnp.load("q_rope_TNH.npy"))
    k_SA = jnp.array(jnp.load("k_SA.npy"))
    k_rope_SH = jnp.array(jnp.load("k_rope_SH.npy"))
    kv_cache = jnp.array(jnp.load("kv_cache.npy"))

    seq_lens = jnp.array(jnp.load("md_seq_lens.npy"))
    block_tables = jnp.array(jnp.load("md_block_tables.npy"))
    query_start_loc = jnp.array(jnp.load("md_query_start_loc.npy"))
    request_distribution = jnp.array(jnp.load("md_request_distribution.npy"))

    # NOTE: Set your scale values exactly as they were printed in your logs!
    sm_scale = 1.0 / (
        64**0.5
    )  # e.g., 0.125. Replace with the actual logged value if different.
    q_scale = None
    k_scale = None  # Replace with your logged k-scale if you use quantization scaling
    v_scale = None

    # --- 3. Define the exact Sharding Specs from your logs ---
    in_specs = (
        P(None, None, None),  # q_TNA
        P(None, None, None),  # q_rope_TNH
        P(None, None),  # k_SA
        P(None, None),  # k_rope_SH
        P(None, None, None, None),  # kv_cache
        P(None),  # md.seq_lens
        P(None),  # md.page_indices_flat
        P(None),  # md.query_start_loc
        P(None),  # md.distribution
    )

    out_specs = (
        P(None, None, None, None),  # new kv cache
        P(None, None, None),  # attention output TNA
    )

    # --- 4. Bind the arrays to the Mesh using NamedSharding ---
    with mesh:
        q_TNA = jax.device_put(q_TNA, NamedSharding(mesh, in_specs[0]))
        q_rope_TNH = jax.device_put(q_rope_TNH,
                                    NamedSharding(mesh, in_specs[1]))
        k_SA = jax.device_put(k_SA, NamedSharding(mesh, in_specs[2]))
        k_rope_SH = jax.device_put(k_rope_SH, NamedSharding(mesh, in_specs[3]))
        kv_cache = jax.device_put(kv_cache, NamedSharding(mesh, in_specs[4]))
        seq_lens = jax.device_put(seq_lens, NamedSharding(mesh, in_specs[5]))
        block_tables = jax.device_put(block_tables,
                                      NamedSharding(mesh, in_specs[6]))
        query_start_loc = jax.device_put(query_start_loc,
                                         NamedSharding(mesh, in_specs[7]))
        request_distribution = jax.device_put(request_distribution,
                                              NamedSharding(mesh, in_specs[8]))

    # --- 5. Define the kernel wrapper identically to your code ---
    def _mla_ragged_paged_attention(q, q_rope, k, k_rope, cache, s_lens,
                                    b_tables, q_start, dist):
        num_kv_pages_per_block = (3, 1, 1)
        num_queries_per_block = (1, 16, 16)

        out, new_cache = mla_ragged_paged_attention(
            q,
            q_rope,
            k,
            k_rope,
            cache,
            s_lens,
            b_tables,
            q_start,
            dist,
            sm_scale=sm_scale,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale)
        return new_cache, out

    sharded_attn_fn = jax.jit(
        jax.shard_map(_mla_ragged_paged_attention,
                      mesh=mesh,
                      in_specs=in_specs,
                      out_specs=out_specs,
                      check_vma=False))

    # --- 6. Execute the Shard-Mapped Kernel ---
    print("Executing compiled JAX kernel...")
    updated_kv_cache, output_TNA = sharded_attn_fn(q_TNA, q_rope_TNH, k_SA,
                                                   k_rope_SH, kv_cache,
                                                   seq_lens, block_tables,
                                                   query_start_loc,
                                                   request_distribution)

    # Block until JAX finishes asynchronous execution on the TPU
    updated_kv_cache.block_until_ready()
    output_TNA.block_until_ready()

    # --- 7. Check for NaNs to confirm repro ---
    print("\nExecution complete!")
    print(f"Output TNA shape: {output_TNA.shape}")
    print(f"Updated KV Cache shape: {updated_kv_cache.shape}")

    kv_has_nans = bool(jnp.isnan(updated_kv_cache).any())
    out_has_nans = bool(jnp.isnan(output_TNA).any())

    print("\nRESULTS:")
    print(f"KV Cache has NaNs: {kv_has_nans}")
    print(f"Output has NaNs:   {out_has_nans}")

    if kv_has_nans or out_has_nans:
        print("\nSUCCESS: Successfully reproduced the NaN issue offline.")
    else:
        print(
            "\nNOTE: No NaNs were generated this run. Double-check `sm_scale` and `k_scale`."
        )


if __name__ == "__main__":
    main()
