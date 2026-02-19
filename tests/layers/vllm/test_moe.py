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

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from tests.layers.common import utils as test_utils
# Adjust the import below to match the actual location of your function
# For example: from my_module.moe_selector import select_moe_backend_from_fused_moe_config
from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
from tpu_inference.layers.common.moe import MoEBackend
# We assume the function is in a file that can be imported.
# If testing locally without the full repo, you might need to adjust imports.
from tpu_inference.layers.vllm.moe import \
    select_moe_backend_from_fused_moe_config


# Mock the FusedMoEConfig so we don't need vllm installed to test the logic
@pytest.fixture
def mock_moe_config():
    """Creates a mock object mimicking FusedMoEConfig."""
    config = MagicMock()
    return config


@pytest.mark.parametrize(
    "env_use_ep_kernel, config_use_ep, expected_backend",
    [
        # Case 1: EP Kernel Env=True, Config EP=True -> FUSED_MOE
        (True, True, MoEBackend.FUSED_MOE),

        # Case 2: EP Kernel Env=False, Config EP=True -> GMM_EP
        (False, True, MoEBackend.GMM_EP),

        # Case 3: EP Kernel Env=False, Config EP=False -> GMM_TP
        (False, False, MoEBackend.GMM_TP),

        # Case 4: EP Kernel Env=True, Config EP=False -> Fallback to GMM_TP
        (True, False, MoEBackend.GMM_TP),
    ])
def test_select_moe_backend_logic(monkeypatch, mock_moe_config,
                                  env_use_ep_kernel, config_use_ep,
                                  expected_backend):
    """
    Tests the main logic paths for backend selection based on environment variables
    and the MoE configuration.
    """
    # Use patch as a context manager to limit the scope of the change to this block.
    # This avoids using monkeypatch and ensures the mock is cleaned up immediately.
    with patch("tpu_inference.envs.USE_MOE_EP_KERNEL", env_use_ep_kernel):
        mock_moe_config.use_ep = config_use_ep

        result = select_moe_backend_from_fused_moe_config(mock_moe_config)

    assert result == expected_backend


def test_fused_moe_func_ep_gmm_v2():
    """Test fused_moe_func with use_ep=True exercising the gmm_v2 code path.

    Parameters (bfloat16, sizes aligned to 128) ensure is_supported_by_gmm_v2
    returns True.
    """
    num_devices = len(jax.devices())
    mesh = test_utils.get_spmd_mesh(num_devices)

    dtype = jnp.bfloat16
    hidden_size = 256
    intermediate_size = 128
    num_experts = num_devices * 2
    num_tokens = 16
    topk = 8
    activation = "silu"
    scoring_fn = "softmax"
    renormalize = True

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)

    hidden_states = jax.random.normal(
        k1, (num_tokens, hidden_size), dtype=dtype) / 10.0

    # ref_moe_jax expects: w1 [num_experts, intermediate_size * 2, hidden_size]
    #                       w2 [num_experts, hidden_size, intermediate_size]
    w1_ref = jax.random.normal(
        k2, (num_experts, intermediate_size * 2, hidden_size),
        dtype=dtype) / 10.0
    w2_ref = jax.random.normal(
        k3, (num_experts, hidden_size, intermediate_size),
        dtype=dtype) / 10.0

    # fused_moe_func expects GMM layout (transposed dims 1,2):
    #   w1 [num_experts, hidden_size, intermediate_size * 2]
    #   w2 [num_experts, intermediate_size, hidden_size]
    w1 = jnp.transpose(w1_ref, (0, 2, 1))
    w2 = jnp.transpose(w2_ref, (0, 2, 1))

    gating_output = jax.random.normal(
        k4, (num_tokens, num_experts), dtype=dtype)

    result = fused_moe_func(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        mesh=mesh,
        use_ep=True,
        activation=activation,
        scoring_fn=scoring_fn,
    )

    expected = test_utils.ref_moe_jax(
        x=hidden_states,
        router_logits=gating_output,
        w1=w1_ref,
        w2=w2_ref,
        w1_bias=None,
        w2_bias=None,
        top_k=topk,
        renormalize=renormalize,
        activation=activation,
        scoring_func=scoring_fn,
    )

    assert result.shape == expected.shape
    assert jnp.allclose(result, expected, atol=2e-1, rtol=2e-1), \
        f"Max absolute diff: {jnp.max(jnp.abs(result - expected))}"
