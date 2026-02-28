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
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
from tpu_inference.layers.common.utils import \
    reorder_concatenated_tensor_for_sharding

jax.config.parse_flags_with_absl()


def gen_moe_inputs(
    dtype: jnp.dtype,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    num_tokens: int,
    *,
    active_expert_ratio: float = 1.0,
    seed: int = 1234,
    has_bias: bool = False,
):
    key = jax.random.key(seed)

    a = jax.random.uniform(key, (num_tokens, hidden_size), dtype=dtype) / 10

    w1 = (jax.random.normal(key,
                            (num_experts, hidden_size, 2 * intermediate_size),
                            dtype=dtype) / 10)
    w2 = (jax.random.normal(key, (num_experts, intermediate_size, hidden_size),
                            dtype=dtype) / 10)

    if has_bias:
        b1 = (jax.random.normal(key, (num_experts, 1, 2 * intermediate_size),
                                dtype=dtype) / 10)
        b2 = jax.random.normal(key,
                               (num_experts, 1, hidden_size), dtype=dtype) / 10
    else:
        b1 = b2 = None

    gating_output = jax.random.uniform(
        key,
        (num_tokens, num_experts),
        minval=0,
        maxval=1,
        dtype=dtype,
    )

    num_active_experts = int(num_experts * active_expert_ratio)
    active_experts = jax.random.normal(key, (num_experts, ))
    val, _ = jax.lax.top_k(active_experts, k=num_active_experts)
    threshold = val[-1]

    mask = active_experts >= threshold
    gating_output = jnp.where(mask, gating_output, 0.0)

    return a, w1, w2, b1, b2, gating_output


def sub_channel_quantize(x: jax.Array, quant_dtype: jnp.dtype, wsz: int = 256):
    """Quantizes x with sub-channel quantization on the 2nd minor."""
    if jnp.issubdtype(quant_dtype, jnp.floating):
        dtype_info = jnp.finfo(quant_dtype)
    else:
        dtype_info = jnp.iinfo(quant_dtype)
    dtype_max = float(dtype_info.max)
    w_lst, scale_lst = [], []
    assert len(x.shape) >= 2
    assert x.shape[-2] % wsz == 0
    for i in range(0, x.shape[1], wsz):
        y = x[:, i:i + wsz, :]
        abs_max = jnp.abs(y).max(axis=1, keepdims=True)
        scale = (abs_max / dtype_max).astype(jnp.float32)
        w = (y / scale).astype(quant_dtype)
        w_lst.append(w)
        scale_lst.append(scale)
    return jnp.concat(w_lst, axis=1), jnp.concat(scale_lst, axis=1)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MoEKernelTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        self.mesh_devices = sorted(
            jax.devices(),
            key=lambda x: (
                x.coords[0],
                (-1 if x.coords[0] % 2 else 1) * x.coords[1],
            ),
        )

    def _test_moe(
        self,
        dtype,
        top_k,
        num_experts,
        hidden_size,
        intermediate_size,
        num_tokens,
        seed,
        renormalize_topk_logits: bool,
        active_expert_ratio: float = 1.0,
        act_fn: str = "silu",
        w_dtype: jnp.dtype | None = None,
        subc_quant_wsz: int | None = None,
        has_bias: bool = False,
        use_ep: bool = False,
        atol: float = 2e-1,
        rtol: float = 2e-1,
        run_baseline: bool = True,
        capture_xprof: bool = False,
        num_cores: int | None = None,
    ):
        if num_cores is None:
            num_cores = len(self.mesh_devices)
        mesh = Mesh(
            np.array(self.mesh_devices[:num_cores]).reshape(1, num_cores),
            axis_names=("data", "model"),
        )

        a, w1, w2, b1, b2, gating_output = gen_moe_inputs(
            dtype,
            num_experts,
            hidden_size,
            intermediate_size,
            num_tokens,
            seed=seed,
            has_bias=has_bias,
            active_expert_ratio=active_expert_ratio,
        )

        if run_baseline:
            # Run baseline without any sharding or quantization.
            single_chip_mesh = Mesh(
                np.array(jax.devices()[:1]).reshape(1, 1),
                axis_names=("data", "model"),
            )
            expected = jax.block_until_ready(
                fused_moe_func(
                    a,
                    w1,
                    w2,
                    w1_scale=None,
                    w2_scale=None,
                    w1_bias=b1,
                    w2_bias=b2,
                    activation=act_fn,
                    gating_output=gating_output,
                    topk=top_k,
                    renormalize=renormalize_topk_logits,
                    mesh=single_chip_mesh,
                    use_ep=False,
                    scoring_fn="softmax",
                ))

        # Pre-process weights.
        n_shards = mesh.shape["model"]
        if not use_ep:
            w1 = reorder_concatenated_tensor_for_sharding(
                w1, [intermediate_size, intermediate_size], n_shards, 2)
            if has_bias:
                b1 = reorder_concatenated_tensor_for_sharding(
                    b1, [intermediate_size, intermediate_size], n_shards, 2)

        # Quantize weights.
        w1_scale = w2_scale = None
        if w_dtype is not None:
            if subc_quant_wsz is None:
                w1, w1_scale = sub_channel_quantize(w1, w_dtype, w1.shape[1])
                w2, w2_scale = sub_channel_quantize(w2, w_dtype, w2.shape[1])
            else:
                w1, w1_scale = sub_channel_quantize(w1, w_dtype,
                                                    subc_quant_wsz)
                w2, w2_scale = sub_channel_quantize(w2, w_dtype,
                                                    subc_quant_wsz)

            w1_scale = jnp.expand_dims(w1_scale, axis=2)
            w2_scale = jnp.expand_dims(w2_scale, axis=2)

        # Shard weights.
        if use_ep:
            ep_sharding = NamedSharding(mesh, P("model"))

            w1 = jax.device_put(w1, ep_sharding)
            w2 = jax.device_put(w2, ep_sharding)
            if w_dtype is not None:
                w1_scale = jax.device_put(w1_scale, ep_sharding)
                w2_scale = jax.device_put(w2_scale, ep_sharding)
            if has_bias:
                b1 = jax.device_put(b1, ep_sharding)
                b2 = jax.device_put(b2, ep_sharding)
        else:
            w1 = jax.device_put(w1, NamedSharding(mesh, P(None, None,
                                                          "model")))
            w2 = jax.device_put(w2, NamedSharding(mesh, P(None, "model",
                                                          None)))
            if w_dtype is not None:
                w1_scale = jax.device_put(
                    w1_scale, NamedSharding(mesh, P(None, None, None,
                                                    "model")))
                w2_scale = jax.device_put(
                    w2_scale, NamedSharding(mesh, P(None, "model")))
            if has_bias:
                b1 = jax.device_put(
                    b1, NamedSharding(mesh, P(None, None, "model")))
                b2 = jax.device_put(b2, NamedSharding(mesh, P(None, None)))

        if capture_xprof:
            xprof_sess = xprof_session.XprofSession()
            xprof_sess.start_session()
        actual = jax.block_until_ready(
            fused_moe_func(
                a,
                w1,
                w2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_bias=b1,
                w2_bias=b2,
                activation=act_fn,
                gating_output=gating_output,
                topk=top_k,
                renormalize=renormalize_topk_logits,
                mesh=mesh,
                use_ep=use_ep,
                scoring_fn="softmax",
            ))
        if capture_xprof:
            url = xprof_sess.end_session_and_get_url()
            print(f"XProf URL: {url}")

        if run_baseline:
            self.assertAllClose(actual, expected, atol=atol, rtol=rtol)


#   @parameterized.product(
#       renormalize_topk_logits=[True, False],
#   )
#   def test_basic(self, renormalize_topk_logits):
#     dtype = jnp.bfloat16
#     top_k = 8
#     num_experts = 128
#     hidden_size = 1024
#     intermediate_size = 1024
#     num_tokens = 8 * 32
#     self._test_moe(
#         dtype=dtype,
#         top_k=top_k,
#         num_experts=num_experts,
#         hidden_size=hidden_size,
#         intermediate_size=intermediate_size,
#         num_tokens=num_tokens,
#         seed=1234,
#         renormalize_topk_logits=renormalize_topk_logits,
#     )

#   @parameterized.product(
#       act_fn=["silu", "swigluoai"],
#   )
#   def test_activation(self, act_fn):
#     dtype = jnp.bfloat16
#     top_k = 8
#     num_experts = 128
#     hidden_size = 1024
#     intermediate_size = 1024
#     num_tokens = 8 * 32
#     self._test_moe(
#         dtype=dtype,
#         top_k=top_k,
#         num_experts=num_experts,
#         hidden_size=hidden_size,
#         intermediate_size=intermediate_size,
#         num_tokens=num_tokens,
#         seed=1234,
#         renormalize_topk_logits=True,
#         act_fn=act_fn,
#     )

#   @parameterized.product(
#       w_dtype=[jnp.int8, jnp.float8_e5m2, jnp.float4_e2m1fn],
#   )
#   def test_sub_channel_quantization(self, w_dtype):
#     if w_dtype in (
#         jnp.float8_e5m2,
#         jnp.float4_e2m1fn,
#     ) and not jtu.is_device_tpu_at_least(version=7):
#       self.skipTest("Expect TPUv7+")
#     dtype = jnp.bfloat16
#     top_k = 8
#     num_experts = 128
#     hidden_size = 1024
#     intermediate_size = 1024
#     num_tokens = 8 * 32
#     self._test_moe(
#         dtype=dtype,
#         top_k=top_k,
#         num_experts=num_experts,
#         hidden_size=hidden_size,
#         intermediate_size=intermediate_size,
#         num_tokens=num_tokens,
#         seed=1234,
#         renormalize_topk_logits=False,
#         w_dtype=w_dtype,
#         subc_quant_wsz=256,
#         use_ep=True,
#     )

#   def test_bias(self):
#     dtype = jnp.bfloat16
#     top_k = 8
#     num_experts = 128
#     hidden_size = 1024
#     intermediate_size = 1024
#     num_tokens = 8 * 32
#     self._test_moe(
#         dtype=dtype,
#         top_k=top_k,
#         num_experts=num_experts,
#         hidden_size=hidden_size,
#         intermediate_size=intermediate_size,
#         num_tokens=num_tokens,
#         seed=1234,
#         renormalize_topk_logits=False,
#         has_bias=True,
#     )

#   @parameterized.product(use_ep=[True, False])
#   def test_benchmark_qwen_235(self, use_ep):
#     num_experts = 128
#     top_k = 8
#     hidden_size = 4096
#     intermediate_size = 1536
#     dtype = jnp.bfloat16
#     num_tokens = 8 * 64
#     seed = 54321
#     renormalize_topk_logits = True
#     self._test_moe(
#         dtype=dtype,
#         top_k=top_k,
#         num_experts=num_experts,
#         hidden_size=hidden_size,
#         intermediate_size=intermediate_size,
#         num_tokens=num_tokens,
#         seed=seed,
#         renormalize_topk_logits=renormalize_topk_logits,
#         act_fn="silu",
#         atol=5e-2,
#         rtol=5e-2,
#         use_ep=use_ep,
#     )

#   @parameterized.product(use_ep=[True, False])
#   def test_benchmark_qwen_30b_a3b(self, use_ep):
#     num_experts = 128
#     top_k = 8
#     hidden_size = 2048
#     intermediate_size = 768
#     dtype = jnp.bfloat16
#     num_tokens = 8 * 64
#     seed = 54321
#     renormalize_topk_logits = True
#     self._test_moe(
#         dtype=dtype,
#         top_k=top_k,
#         num_experts=num_experts,
#         hidden_size=hidden_size,
#         intermediate_size=intermediate_size,
#         num_tokens=num_tokens,
#         seed=seed,
#         renormalize_topk_logits=renormalize_topk_logits,
#         act_fn="silu",
#         atol=5e-2,
#         rtol=5e-2,
#         use_ep=use_ep,
#     )

#   @parameterized.product(
#       num_tokens=[64, 8192],
#       w_dtype=[jnp.float8_e4m3fn],
#       block_size=[None],
#       use_ep=[True],
#       active_expert_ratio=[0.6],
#   )
#   def test_benchmark_qwen_coder_480b(
#       self, num_tokens, w_dtype, block_size, use_ep, active_expert_ratio
#   ):
#     num_experts = 160
#     top_k = 8

#     subc_quant_wsz = None
#     if w_dtype is not None:
#       subc_quant_wsz = block_size
#     hidden_size = 6144
#     intermediate_size = 2560

#     dtype = jnp.bfloat16

#     seed = 54321
#     renormalize_topk_logits = True
#     self._test_moe(
#         dtype=dtype,
#         top_k=top_k,
#         num_experts=num_experts,
#         hidden_size=hidden_size,
#         intermediate_size=intermediate_size,
#         num_tokens=num_tokens,
#         seed=seed,
#         renormalize_topk_logits=renormalize_topk_logits,
#         subc_quant_wsz=subc_quant_wsz,
#         w_dtype=w_dtype,
#         use_ep=use_ep,
#         active_expert_ratio=active_expert_ratio,
#         act_fn="silu",
#         atol=1e1,
#         rtol=1e1,
#         num_cores=8,
#         run_baseline=False,
#         capture_xprof=True,
#     )

    @parameterized.product(
        num_tokens=[64],
        w_dtype=[jnp.float4_e2m1fn],
        has_bias=[True],
        block_size=[512],
        use_ep=[True],
    )
    def test_benchmark_gpt_oss_120(self, num_tokens, w_dtype, has_bias,
                                   block_size, use_ep):
        num_experts = 128
        top_k = 4

        subc_quant_wsz = None
        if w_dtype is not None:
            subc_quant_wsz = block_size
        hidden_size = 3072
        intermediate_size = 3072

        dtype = jnp.bfloat16

        seed = 54321
        renormalize_topk_logits = True
        self._test_moe(
            dtype=dtype,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_tokens=num_tokens,
            seed=seed,
            renormalize_topk_logits=renormalize_topk_logits,
            subc_quant_wsz=subc_quant_wsz,
            has_bias=has_bias,
            w_dtype=w_dtype,
            use_ep=use_ep,
            act_fn="swigluoai",
            atol=1e1,
            rtol=1e1,
            num_cores=2,
            capture_xprof=False,
            run_baseline=False,
        )

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
