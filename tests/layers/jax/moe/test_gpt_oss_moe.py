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
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.config import (DeviceConfig, ParallelConfig, VllmConfig,
                         set_current_vllm_config)

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.jax.moe.gpt_oss_moe import GptOssMoE, GptOssRouter
from tpu_inference.layers.jax.quantization.unquantized import \
    UnquantizedFusedMoEMethod


def _mesh():
    return Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1),
                axis_names=("data", "model"))


def _router(mesh, backend):
    return GptOssRouter(
        dtype=jnp.float32,
        hidden_size=4,
        num_experts=3,
        num_experts_per_tok=2,
        router_act="softmax",
        rngs=nnx.Rngs(0),
        activation_ffw_td=P("data", None),
        ed_sharding=P(None, "model"),
        e_sharding=P("model"),
        moe_backend=backend,
        mesh=mesh,
    )


def test_gpt_oss_router_returns_raw_biased_logits_for_fused_backend():
    mesh = _mesh()
    with jax.set_mesh(mesh):
        router = _router(mesh, MoEBackend.GMM_TP)
        router.kernel_DE.value = jnp.arange(12,
                                            dtype=jnp.float32).reshape(4, 3)
        router.bias_E.value = jnp.array([0.5, 1.0, -1.5], dtype=jnp.float32)
        x_TD = jnp.ones((2, 4), dtype=jnp.float32)

        logits_TE = router(x_TD)

    expected = x_TD @ router.kernel_DE.value + router.bias_E.value
    assert logits_TE.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(logits_TE), np.asarray(expected))


def test_gpt_oss_router_keeps_dense_topk_contract():
    mesh = _mesh()
    with jax.set_mesh(mesh):
        router = _router(mesh, MoEBackend.DENSE_MAT)
        x_TD = jnp.ones((2, 4), dtype=jnp.float32)

        weights_TX, indices_TX = router(x_TD)

    assert weights_TX.shape == (2, 2)
    assert indices_TX.shape == (2, 2)
    np.testing.assert_allclose(np.asarray(weights_TX.sum(axis=-1)),
                               np.ones((2, )))


def test_softmax_topk_renormalize_equals_gpt_oss_topk_softmax():
    logits_TE = jax.random.normal(jax.random.PRNGKey(42), (17, 11))
    top_k = 4

    softmax_scores_TE = jax.nn.softmax(logits_TE, axis=-1)
    shared_weights_TX, shared_indices_TX = jax.lax.top_k(
        softmax_scores_TE, top_k)
    shared_weights_TX = shared_weights_TX / jnp.sum(
        shared_weights_TX, axis=-1, keepdims=True)

    gpt_oss_logits_TX, gpt_oss_indices_TX = jax.lax.top_k(logits_TE, top_k)
    gpt_oss_weights_TX = jax.nn.softmax(gpt_oss_logits_TX, axis=-1)

    np.testing.assert_array_equal(np.asarray(shared_indices_TX),
                                  np.asarray(gpt_oss_indices_TX))
    np.testing.assert_allclose(np.asarray(shared_weights_TX),
                               np.asarray(gpt_oss_weights_TX),
                               rtol=1e-6,
                               atol=1e-6)


def test_gpt_oss_moe_is_routed_experts_subclass():
    """GptOssMoE is a JaxRoutedExperts subclass (like Gemma4MoE).

    This is a structural check: construction derives moe_backend from the
    current vLLM parallel config and installs the unquantized method by
    default. It does not run a forward pass -- that requires the full
    load -> process lifecycle (weights are not loaded here).
    """
    mesh = _mesh()
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"),
                             parallel_config=ParallelConfig(
                                 tensor_parallel_size=mesh.devices.size,
                                 enable_expert_parallel=False))

    with jax.set_mesh(mesh), set_current_vllm_config(vllm_config):
        moe = GptOssMoE(
            dtype=jnp.float32,
            mesh=mesh,
            rngs=nnx.Rngs(1),
            num_local_experts=3,
            hidden_size=4,
            intermediate_size_moe=6,
            num_experts_per_tok=2,
            random_init=True,
        )

    assert isinstance(moe.quant_method, UnquantizedFusedMoEMethod)
    assert moe.hidden_act == "swigluoai"
    assert moe.activation == "swigluoai"
    assert moe.top_k == 2
    # Single-device / no expert parallel -> a fused GMM backend, not DENSE_MAT.
    assert moe.moe_backend in MoEBackend.fused_moe_backends()
