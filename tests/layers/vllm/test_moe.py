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
from tpu_inference.layers.common.fused_moe_gmm import (
    fused_moe_func,
    _selective_gather_ep_0,
    _selective_gather_ep_1,
)
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


class TestSelectiveGatherEp:
    """Tests for _selective_gather_ep_0 and _selective_gather_ep_1.

    These run outside jit/shard_map so you can use plain Python debugging
    (breakpoint(), print, pdb) to inspect intermediate values.

    Setup:
        4 experts, 8 total token-expert slots (num_tokens * topk).
        group_sizes tells how many slots are routed to each expert.
        token_indices_sorted is the permutation that maps sorted position
        back to the original token index in hidden_states.
    """

    @pytest.fixture
    def gather_inputs(self):
        """Build a concrete routing scenario.

        Experts: 0, 1, 2, 3
        group_sizes: [2, 3, 1, 2]  -> cumsum = [2, 5, 6, 8]
          expert 0 owns sorted positions [0, 1]
          expert 1 owns sorted positions [2, 3, 4]
          expert 2 owns sorted positions [5]
          expert 3 owns sorted positions [6, 7]

        hidden_states: 8 tokens, hidden_size=4, each row = [i+1]*4
          so hidden_states[i] is easily identifiable.

        token_indices_sorted: the gather permutation produced by the
          router — maps each sorted slot to the original token index.
        """
        num_tokens = 8
        hidden_size = 4
        hidden_states = jnp.array(
            [[(i + 1)] * hidden_size for i in range(num_tokens)],
            dtype=jnp.float32,
        )
        # arbitrary permutation (each value in [0, num_tokens))
        token_indices_sorted = jnp.array([3, 7, 0, 5, 2, 6, 1, 4])
        group_sizes = jnp.array([2, 3, 1, 2])
        return hidden_states, token_indices_sorted, group_sizes

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _naive_selective_gather(hidden_states, token_indices_sorted,
                                group_sizes, group_offset,
                                num_experts_per_shard):
        """Straightforward NumPy-style reference implementation."""
        cumsum = jnp.cumsum(group_sizes)
        ep_start_expert = group_offset[0]
        start = 0 if ep_start_expert == 0 else int(cumsum[ep_start_expert - 1])
        end = int(cumsum[ep_start_expert + num_experts_per_shard - 1])

        num_total = token_indices_sorted.shape[0]
        hidden_size = hidden_states.shape[1]
        out = jnp.zeros((num_total, hidden_size), dtype=hidden_states.dtype)
        for pos in range(start, end):
            token_idx = int(token_indices_sorted[pos])
            out = out.at[pos].set(hidden_states[token_idx])
        return out

    # -- tests ---------------------------------------------------------

    @pytest.mark.parametrize("ep_expert_start, num_experts_per_shard", [
        (0, 1),   # shard owns expert 0
        (1, 1),   # shard owns expert 1
        (2, 2),   # shard owns experts 2-3
        (0, 2),   # shard owns experts 0-1
        (0, 4),   # shard owns all experts
    ])
    def test_selective_gather_ep_0(self, gather_inputs,
                                   ep_expert_start, num_experts_per_shard):
        # Experts: 0, 1, 2, 3
        # group_sizes: [2, 3, 1, 2]  -> cumsum = [2, 5, 6, 8]
        #   expert 0 owns sorted positions [0, 1]
        #   expert 1 owns sorted positions [2, 3, 4]
        #   expert 2 owns sorted positions [5]
        #   expert 3 owns sorted positions [6, 7]

        # hidden_states: 8 tokens, hidden_size=4, each row = [i+1]*4
        #   so hidden_states[i] is easily identifiable.
        # token_indices_sorted=[3, 7, 0, 5, 2, 6, 1, 4]
        # group_offset = jnp.arange(0, num_experts, num_experts_per_shard)
        hidden_states, token_indices_sorted, group_sizes = gather_inputs
        group_offset = jnp.array([ep_expert_start])
        cumsum_gs = jnp.cumsum(group_sizes)
        token_start = jnp.where(ep_expert_start > 0, cumsum_gs[ep_expert_start - 1], 0)
        token_end = cumsum_gs[ep_expert_start + num_experts_per_shard - 1]


        result = _selective_gather_ep_0(
            hidden_states, token_indices_sorted, group_sizes,
            group_offset, num_experts_per_shard,
        )
        expected = self._naive_selective_gather(
            hidden_states, token_indices_sorted, group_sizes,
            group_offset, num_experts_per_shard,
        )
        assert result.shape == expected.shape
        assert jnp.allclose(result[token_start:token_end], expected[token_start:token_end]), (
            f"ep_expert_start={ep_expert_start}, "
            f"num_experts_per_shard={num_experts_per_shard}\n"
            f"result:\n{result}\nexpected:\n{expected}"
        )

    @pytest.mark.parametrize("ep_expert_start, num_experts_per_shard", [
        (0, 1),   # shard owns expert 0
        (1, 1),   # shard owns expert 1
        (2, 2),   # shard owns experts 2-3
        (0, 2),   # shard owns experts 0-1
        (0, 4),   # shard owns all experts
    ])
    def test_selective_gather_ep_1(self, gather_inputs,
                                   ep_expert_start, num_experts_per_shard):
        # Experts: 0, 1, 2, 3
        # group_sizes: [2, 3, 1, 2]  -> cumsum = [2, 5, 6, 8]
        #   expert 0 owns sorted positions [0, 1]
        #   expert 1 owns sorted positions [2, 3, 4]
        #   expert 2 owns sorted positions [5]
        #   expert 3 owns sorted positions [6, 7]

        # hidden_states: 8 tokens, hidden_size=4, each row = [i+1]*4
        #   so hidden_states[i] is easily identifiable.
        # token_indices_sorted=[3, 7, 0, 5, 2, 6, 1, 4]
        # group_offset = jnp.arange(0, num_experts, num_experts_per_shard)
        hidden_states, token_indices_sorted, group_sizes = gather_inputs
        group_offset = jnp.array([ep_expert_start])
        cumsum_gs = jnp.cumsum(group_sizes)
        token_start = jnp.where(ep_expert_start > 0, cumsum_gs[ep_expert_start - 1], 0) # cumsum_gs[0]==2
        token_end = cumsum_gs[ep_expert_start + num_experts_per_shard - 1]  # 5

        result = _selective_gather_ep_1(
            hidden_states, token_indices_sorted, group_sizes,
            group_offset, num_experts_per_shard,
        )
        expected = self._naive_selective_gather(
            hidden_states, token_indices_sorted, group_sizes,
            group_offset, num_experts_per_shard,
        )
        assert result.shape == expected.shape
        assert jnp.allclose(result[token_start:token_end], expected[token_start:token_end]), (
            f"ep_expert_start={ep_expert_start}, "
            f"num_experts_per_shard={num_experts_per_shard}\n"
            f"result:\n{result}\nexpected:\n{expected}"
        )

    @pytest.mark.parametrize("ep_expert_start, num_experts_per_shard", [
        (0, 1),
        (1, 1),
        (2, 2),
        (0, 4),
    ])
    def test_ep_0_matches_ep_1(self, gather_inputs,
                               ep_expert_start, num_experts_per_shard):
        """Both implementations should produce identical results."""
        hidden_states, token_indices_sorted, group_sizes = gather_inputs
        group_offset = jnp.array([ep_expert_start])

        r0 = _selective_gather_ep_0(
            hidden_states, token_indices_sorted, group_sizes,
            group_offset, num_experts_per_shard,
        )
        r1 = _selective_gather_ep_1(
            hidden_states, token_indices_sorted, group_sizes,
            group_offset, num_experts_per_shard,
        )
        assert jnp.allclose(r0, r1), (
            f"ep_expert_start={ep_expert_start}, "
            f"num_experts_per_shard={num_experts_per_shard}\n"
            f"ep_0 result:\n{r0}\nep_1 result:\n{r1}"
        )
