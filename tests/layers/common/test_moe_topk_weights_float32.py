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
"""Tests for TPU_MOE_TOPK_WEIGHTS_FLOAT32 environment flag.

Verifies that FusedMoE top-k routing weights are kept in float32 when
TPU_MOE_TOPK_WEIGHTS_FLOAT32 is enabled, and cast to the layer dtype (e.g. bfloat16)
when disabled.
"""

import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

import tpu_inference.envs as envs
from tpu_inference.layers.common import fused_moe_gmm


def test_envs_flag_resolution():
    with mock.patch.dict(os.environ, {
            "TPU_MOE_TOPK_WEIGHTS_FLOAT32": "0",
            "MOE_TOPK_WEIGHTS_FLOAT32": "0"
    }):
        assert envs.TPU_MOE_TOPK_WEIGHTS_FLOAT32 is False

    with mock.patch.dict(os.environ, {"TPU_MOE_TOPK_WEIGHTS_FLOAT32": "1"}):
        assert envs.TPU_MOE_TOPK_WEIGHTS_FLOAT32 is True

    with mock.patch.dict(os.environ, {
            "TPU_MOE_TOPK_WEIGHTS_FLOAT32": "",
            "MOE_TOPK_WEIGHTS_FLOAT32": "1"
    }):
        assert envs.TPU_MOE_TOPK_WEIGHTS_FLOAT32 is True


def test_all_gather_topk_weights_dtype():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data", ))
    topk_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    topk_weights = jnp.array([[0.6, 0.4], [0.7, 0.3]], dtype=jnp.bfloat16)

    with mock.patch("jax.lax.with_sharding_constraint",
                    side_effect=lambda x, s: x):
        # Default / Flag disabled: returns weights cast to activation dtype (bfloat16)
        with mock.patch.dict(os.environ, {
                "TPU_MOE_TOPK_WEIGHTS_FLOAT32": "0",
                "MOE_TOPK_WEIGHTS_FLOAT32": "0"
        }):
            _, w_out = fused_moe_gmm.all_gather_topk_indices_and_weights(
                topk_indices, topk_weights, jnp.bfloat16, mesh)
            assert w_out.dtype == jnp.bfloat16

        # Flag enabled: keeps weights in float32
        with mock.patch.dict(os.environ,
                             {"TPU_MOE_TOPK_WEIGHTS_FLOAT32": "1"}):
            _, w_out = fused_moe_gmm.all_gather_topk_indices_and_weights(
                topk_indices, topk_weights, jnp.bfloat16, mesh)
            assert w_out.dtype == jnp.float32


def test_fused_moe_func_weights_dtype():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data", ))
    num_tokens = 16
    hidden_dim = 64
    intermediate_dim = 128
    num_experts = 8
    topk = 2

    hidden_states = jnp.ones((num_tokens, hidden_dim), dtype=jnp.bfloat16)
    gating_output = jnp.ones((num_tokens, num_experts), dtype=jnp.bfloat16)
    w1 = jnp.ones((num_experts, hidden_dim, 2 * intermediate_dim),
                  dtype=jnp.bfloat16)
    w2 = jnp.ones((num_experts, intermediate_dim, hidden_dim),
                  dtype=jnp.bfloat16)

    captured_weights = []

    def fake_tp_gmm(*args, **kwargs):
        weights = (kwargs.get("topk_weights")
                   if "topk_weights" in kwargs else args[9])
        captured_weights.append(weights)
        return jnp.zeros_like(args[0])

    with mock.patch.object(fused_moe_gmm,
                           "tensor_parallel_gmm",
                           side_effect=fake_tp_gmm):
        # Flag off: weights reaching combine stay bfloat16, output matches hidden_states dtype
        jax.clear_caches()
        with mock.patch.dict(os.environ, {
                "TPU_MOE_TOPK_WEIGHTS_FLOAT32": "0",
                "MOE_TOPK_WEIGHTS_FLOAT32": "0"
        }):
            out = fused_moe_gmm.fused_moe_func(
                hidden_states,
                w1,
                w2,
                None,
                None,
                None,
                None,
                gating_output,
                topk,
                True,
                mesh,
                False,
                "silu",
                "softmax",
            )
            assert captured_weights[-1].dtype == jnp.bfloat16
            assert out.dtype == hidden_states.dtype

        # Flag on: weights reaching combine stay float32, output matches hidden_states dtype
        jax.clear_caches()
        with mock.patch.dict(os.environ,
                             {"TPU_MOE_TOPK_WEIGHTS_FLOAT32": "1"}):
            out = fused_moe_gmm.fused_moe_func(
                hidden_states,
                w1,
                w2,
                None,
                None,
                None,
                None,
                gating_output,
                topk,
                True,
                mesh,
                False,
                "silu",
                "softmax",
            )
            assert captured_weights[-1].dtype == jnp.float32
            assert out.dtype == hidden_states.dtype
