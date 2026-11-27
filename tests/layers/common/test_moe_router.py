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
"""Tests for select_experts in moe_router."""

from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.common.moe_router import select_experts

# Expert 3 has the lowest score, but its correction bias makes it the top pick.
GATING = jnp.array([[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]], jnp.float32)
BIAS = jnp.array([0.0, 0.0, 0.0, 10.0], jnp.float32)


@pytest.mark.parametrize("approx_topk", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_correction_bias_picks_experts_but_not_weights(approx_topk,
                                                       renormalize):
    with mock.patch("tpu_inference.envs.MOE_APPROX_TOPK", approx_topk):
        weights, indices = select_experts(GATING,
                                          topk=2,
                                          renormalize=renormalize,
                                          scoring_fn="sigmoid",
                                          expert_score_correction_bias=BIAS)

    scores = np.asarray(1 / (1 + np.exp(-np.asarray(GATING))))
    want_indices = np.array([[3, 0], [3, 2]])
    want_weights = np.take_along_axis(scores, want_indices, axis=-1)
    if renormalize:
        want_weights = want_weights / want_weights.sum(axis=-1, keepdims=True)
    np.testing.assert_array_equal(np.asarray(indices), want_indices)
    np.testing.assert_allclose(np.asarray(weights), want_weights, rtol=1e-6)


@pytest.mark.parametrize("approx_topk", [False, True])
def test_no_correction_bias(approx_topk):
    with mock.patch("tpu_inference.envs.MOE_APPROX_TOPK", approx_topk):
        weights, indices = select_experts(GATING,
                                          topk=2,
                                          renormalize=False,
                                          scoring_fn="softmax")

    probs = np.asarray(
        jnp.exp(GATING) / jnp.exp(GATING).sum(-1, keepdims=True))
    want_indices = np.array([[0, 1], [3, 2]])
    np.testing.assert_array_equal(np.asarray(indices), want_indices)
    np.testing.assert_allclose(np.asarray(weights),
                               np.take_along_axis(probs, want_indices, -1),
                               rtol=1e-6)
