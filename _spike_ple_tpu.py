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
"""TPU sharding + BF16 de-risk for Gemma-4 PLE Stage 2 (kb_ple.md §3.1).

THROWAWAY. Adapted from ~/private-tool/tpu-inference/gemma4/spike_ple.py
(which validated math on CPU). This version exercises:
  - jax.sharding.Mesh + NamedSharding plumbing
  - nnx.with_partitioning weight init
  - bfloat16 dtype
  - the four math/shape tests on a real TPU device

Run via dev pipeline (see .buildkite/pipeline_dev.yml on this branch).
Local-on-TPU run: python3 /workspace/tpu_inference/_spike_ple_tpu.py
"""

from typing import Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh


def banner(s):
    print(f"\n{'=' * 60}\n{s}\n{'=' * 60}", flush=True)


def env_report():
    banner("env")
    print(f"jax {jax.__version__}", flush=True)
    print(f"flax {__import__('flax').__version__}", flush=True)
    print(f"default_backend = {jax.default_backend()}", flush=True)
    print(f"devices = {jax.devices()}", flush=True)
    print(f"local_device_count = {jax.local_device_count()}", flush=True)
    if jax.default_backend() != "tpu":
        print("FAIL: not running on TPU", flush=True)
        raise SystemExit(2)


class PerLayerInputCompute(nnx.Module):
    """kb_ple.md §3.1 model-level PLE compute, with sharding annotations.

    Shardings:
      embed_tokens_per_layer[V, L*P]:    replicated (vllm: VocabParallelEmbedding,
                                          but for our [V_ple, L*P] case we replicate
                                          since L*P is small and gather_output=True
                                          downstream).
      per_layer_model_projection[H, L*P]: shard H along "model" axis to mirror
                                          ColumnParallelLinear input dim. The output
                                          is gathered (gather_output=True), so the
                                          L*P dim stays replicated.
      per_layer_projection_norm_w[P]:    replicated.

    NOTE: The actual production module will use the JaxEinsum/JaxEmbed wrappers
    from tpu_inference/layers/jax/. This is a raw-NNX spike to keep the
    test self-contained.
    """

    def __init__(self,
                 V,
                 H,
                 P_,
                 L,
                 *,
                 mesh: Mesh,
                 rngs: nnx.Rngs,
                 dtype=jnp.bfloat16):
        self.V, self.H, self.P, self.L = V, H, P_, L
        self.eps = 1e-6
        self.dtype = dtype

        # Repo convention (see tpu_inference/models/jax/gemma4.py:70,135):
        #   - sharding spec attached via nnx.with_partitioning on the initializer
        #   - eager_sharding=False on the Param to defer sharding to JIT
        init = nnx.initializers.normal(stddev=0.02)
        embed_init = nnx.with_partitioning(init, (None, None))  # replicated
        proj_init = nnx.with_partitioning(init, ("model", None))  # H sharded
        norm_init = nnx.with_partitioning(nnx.initializers.ones_init(),
                                          (None, ))  # replicated

        self.embed_tokens_per_layer = nnx.Param(
            embed_init(rngs.params(), (V, L * P_), dtype),
            eager_sharding=False,
        )
        self.per_layer_model_projection = nnx.Param(
            proj_init(rngs.params(), (H, L * P_), dtype),
            eager_sharding=False,
        )
        self.per_layer_projection_norm_w = nnx.Param(
            norm_init(rngs.params(), (P_, ), dtype),
            eager_sharding=False,
        )

        # Constants per kb_ple §6 — Python floats (single-ownership, §11 lesson 3).
        self.embed_scale_per_layer = float(P_)**0.5
        self.per_layer_input_scale = 1.0 / (2.0**0.5)
        self.per_layer_projection_scale = float(H)**-0.5

    def _rmsnorm(self, x):
        x_f32 = x.astype(jnp.float32)  # accumulate in fp32 like prod RMSNorm
        rms = jnp.sqrt(
            jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + self.eps)
        normed = (x_f32 / rms).astype(x.dtype)
        return normed * self.per_layer_projection_norm_w[...]

    def __call__(
        self,
        input_ids: jax.Array,
        inputs_embeds: jax.Array,
        is_multimodal: Optional[jax.Array] = None,
    ) -> jax.Array:
        T = input_ids.shape[0]

        if is_multimodal is not None:
            ple_input_ids = jnp.where(is_multimodal, 0, input_ids)
        else:
            ple_input_ids = input_ids

        # Track A
        per_layer_embeds = self.embed_tokens_per_layer[...][ple_input_ids]
        per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer
        per_layer_embeds = per_layer_embeds.reshape(T, self.L, self.P)

        # Track B
        per_layer_projection = inputs_embeds @ self.per_layer_model_projection[
            ...]
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale
        per_layer_projection = per_layer_projection.reshape(T, self.L, self.P)
        per_layer_projection = self._rmsnorm(per_layer_projection)

        return (per_layer_projection +
                per_layer_embeds) * self.per_layer_input_scale


def hand_oracle_test(mesh):
    """Tiny-dim, hand-derived expected-output test in BF16."""
    banner("hand_oracle_test (BF16, sharded mesh)")
    V, H, P_, L, T = 4, 4, 2, 3, 2

    with mesh:
        module = PerLayerInputCompute(V,
                                      H,
                                      P_,
                                      L,
                                      mesh=mesh,
                                      rngs=nnx.Rngs(0),
                                      dtype=jnp.bfloat16)

        # Overwrite weights with known constants.
        et = jnp.zeros((V, L * P_), dtype=jnp.bfloat16)
        et = et.at[0, 0].set(jnp.bfloat16(1.0))
        et = et.at[1, 1].set(jnp.bfloat16(1.0))
        et = et.at[2, 2].set(jnp.bfloat16(1.0))
        et = et.at[3, 3].set(jnp.bfloat16(1.0))
        module.embed_tokens_per_layer.value = et

        pm = jnp.zeros((H, L * P_), dtype=jnp.bfloat16)
        for i in range(H):
            pm = pm.at[i, i].set(jnp.bfloat16(1.0))
        module.per_layer_model_projection.value = pm

        module.per_layer_projection_norm_w.value = jnp.ones((P_, ),
                                                            dtype=jnp.bfloat16)

        input_ids = jnp.array([0, 1], dtype=jnp.int32)
        inputs_embeds = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                                  dtype=jnp.bfloat16)

        out = module(input_ids, inputs_embeds)

    print(f"out.shape = {out.shape}", flush=True)
    print(f"out.dtype = {out.dtype}", flush=True)
    print(
        f"out (cast to fp32 for printing):\n{np.asarray(out.astype(jnp.float32))}",
        flush=True)
    expected = np.array(
        [[[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
         [[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]]],
        dtype=np.float32,
    )
    out_np = np.asarray(out.astype(jnp.float32))
    assert out.shape == (T, L, P_), f"shape: {out.shape}"
    assert out.dtype == jnp.bfloat16, f"dtype: {out.dtype}"
    # BF16 ulp at magnitude 2 is ~0.0156; allow 1e-2 atol.
    assert np.allclose(
        out_np, expected,
        atol=1e-2), f"value:\n{out_np}\nvs expected:\n{expected}"
    print("hand_oracle_test PASSED", flush=True)


def is_multimodal_test(mesh):
    banner("is_multimodal_test (BF16, sharded mesh)")
    V, H, P_, L, T = 4, 4, 2, 3, 2

    with mesh:
        module = PerLayerInputCompute(V,
                                      H,
                                      P_,
                                      L,
                                      mesh=mesh,
                                      rngs=nnx.Rngs(0),
                                      dtype=jnp.bfloat16)

        et = jnp.zeros((V, L * P_), dtype=jnp.bfloat16)
        et = et.at[0, 0].set(jnp.bfloat16(0.5))
        et = et.at[2, 5].set(jnp.bfloat16(99.0))
        module.embed_tokens_per_layer.value = et
        module.per_layer_model_projection.value = jnp.zeros((H, L * P_),
                                                            dtype=jnp.bfloat16)
        module.per_layer_projection_norm_w.value = jnp.ones((P_, ),
                                                            dtype=jnp.bfloat16)

        input_ids = jnp.array([0, 2], dtype=jnp.int32)
        inputs_embeds = jnp.zeros((T, H), dtype=jnp.bfloat16)
        is_multimodal = jnp.array([False, True], dtype=jnp.bool_)

        out = module(input_ids, inputs_embeds, is_multimodal=is_multimodal)

    expected = np.array(
        [[[0.5, 0.0], [0.0, 0.0], [0.0, 0.0]],
         [[0.5, 0.0], [0.0, 0.0], [0.0, 0.0]]],
        dtype=np.float32,
    )
    out_np = np.asarray(out.astype(jnp.float32))
    print(f"out (fp32):\n{out_np}", flush=True)
    assert np.allclose(out_np, expected,
                       atol=1e-2), f"out=\n{out_np}\nexp=\n{expected}"
    print("is_multimodal_test PASSED", flush=True)


def realistic_dims_test(mesh):
    """E2B-realistic dims V=262144, H=1536, P=256, L=35, T=4."""
    banner("realistic_dims_test (BF16, sharded mesh)")
    V, H, P_, L, T = 262144, 1536, 256, 35, 4

    with mesh:
        module = PerLayerInputCompute(V,
                                      H,
                                      P_,
                                      L,
                                      mesh=mesh,
                                      rngs=nnx.Rngs(42),
                                      dtype=jnp.bfloat16)
        input_ids = jax.random.randint(jax.random.key(0), (T, ), 0, V)
        inputs_embeds = jax.random.normal(jax.random.key(1), (T, H),
                                          dtype=jnp.float32).astype(
                                              jnp.bfloat16)
        out = module(input_ids, inputs_embeds)
        out.block_until_ready()
    out_np = np.asarray(out.astype(jnp.float32))
    print(f"out.shape = {out.shape}    expected ({T}, {L}, {P_})", flush=True)
    print(f"out.dtype = {out.dtype}", flush=True)
    print(f"out finite? {np.isfinite(out_np).all()}", flush=True)
    print(
        f"out abs-mean = {np.mean(np.abs(out_np)):.4f}, max = {np.max(np.abs(out_np)):.4f}",
        flush=True)
    assert out.shape == (T, L, P_)
    assert out.dtype == jnp.bfloat16
    assert np.isfinite(out_np).all(), "non-finite values"
    print("realistic_dims_test PASSED", flush=True)


def jit_test(mesh):
    banner("jit_test (BF16, sharded mesh, nnx.jit)")
    V, H, P_, L, T = 4, 4, 2, 3, 2

    with mesh:
        module = PerLayerInputCompute(V,
                                      H,
                                      P_,
                                      L,
                                      mesh=mesh,
                                      rngs=nnx.Rngs(0),
                                      dtype=jnp.bfloat16)

        @nnx.jit
        def fwd(m, ids, embeds):
            return m(ids, embeds)

        input_ids = jnp.array([0, 1], dtype=jnp.int32)
        inputs_embeds = jnp.zeros((T, H), dtype=jnp.bfloat16)
        out = fwd(module, input_ids, inputs_embeds)
        out.block_until_ready()
    print(f"out.shape = {out.shape}", flush=True)
    assert out.shape == (T, L, P_)
    print("jit_test PASSED", flush=True)


def sharding_inspection(mesh, V=4, H=4, P_=2, L=3, label=""):
    """Verify the sharded weight is actually placed via NamedSharding."""
    banner(f"sharding_inspection {label}".rstrip())
    with mesh:
        module = PerLayerInputCompute(V,
                                      H,
                                      P_,
                                      L,
                                      mesh=mesh,
                                      rngs=nnx.Rngs(0),
                                      dtype=jnp.bfloat16)
        w = module.per_layer_model_projection[...]
        print(f"per_layer_model_projection.sharding = {w.sharding}",
              flush=True)
        print(f"per_layer_model_projection.shape    = {w.shape}", flush=True)
        e = module.embed_tokens_per_layer[...]
        print(f"embed_tokens_per_layer.sharding     = {e.sharding}",
              flush=True)
        print(f"embed_tokens_per_layer.shape        = {e.shape}", flush=True)
    print("sharding_inspection done", flush=True)


def main():
    env_report()

    devices = jax.devices()
    mesh_full = Mesh(np.array(devices).reshape(len(devices)),
                     axis_names=("model", ))
    mesh_one = Mesh(np.array(devices[:1]).reshape(1), axis_names=("model", ))
    print(f"\nmesh_full = {mesh_full}    (size {len(devices)}, full sharding)",
          flush=True)
    print(f"mesh_one  = {mesh_one}    (size 1, for small-dim math tests)",
          flush=True)

    # Small-dim math/correctness tests on a singleton mesh:
    # H=4 in these tests doesn't evenly divide 8 (or any larger TP), so we
    # restrict the mesh to 1 device for them. They validate math, not sharding.
    sharding_inspection(mesh_one, label="(singleton mesh, small dims)")
    hand_oracle_test(mesh_one)
    is_multimodal_test(mesh_one)
    jit_test(mesh_one)

    # Realistic E2B dims on the full mesh — H=1536 / 8 = 192 (clean).
    # This is the test that actually exercises 8-way sharding.
    realistic_dims_test(mesh_full)
    # And inspect the real shard layout at realistic dims.
    sharding_inspection(mesh_full,
                        V=262144,
                        H=1536,
                        P_=256,
                        L=35,
                        label="(full mesh, E2B dims)")

    banner("All TPU spike tests PASSED")


if __name__ == "__main__":
    main()
