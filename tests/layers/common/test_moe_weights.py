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

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from unittest.mock import patch, call

from tpu_inference import envs
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_fp8_moe_weights_direct, process_moe_weights,
    shard_moe_weights)
from tpu_inference.layers.common.sharding import ShardingAxisName


def test_shard_moe_weights_uses_expert_axis_for_gmm_ep():
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1)
    mesh = Mesh(devices,
                axis_names=("data", "attn_dp", "attn_dp_expert", "expert",
                            "model"))
    weights = FusedMoEWeights(
        w13_weight=jnp.ones((2, 4, 3), dtype=jnp.bfloat16),
        w13_weight_scale=jnp.ones((2, 2, 1, 4), dtype=jnp.float32),
        w13_bias=None,
        w2_weight=jnp.ones((2, 3, 4), dtype=jnp.bfloat16),
        w2_weight_scale=jnp.ones((2, 2, 1, 3), dtype=jnp.float32),
        w2_bias=None,
    )

    sharded = shard_moe_weights(weights, MoEBackend.GMM_EP, mesh)
    expected_spec = PartitionSpec(ShardingAxisName.EXPERT)

    assert isinstance(sharded.w13_weight.sharding, NamedSharding)
    assert sharded.w13_weight.sharding.spec == expected_spec
    assert sharded.w13_weight_scale.sharding.spec == expected_spec
    assert sharded.w2_weight.sharding.spec == expected_spec
    assert sharded.w2_weight_scale.sharding.spec == expected_spec


@patch.object(envs, 'NEW_MODEL_DESIGN', True)
def test_shard_moe_weights_uses_full_expert_axis_for_new_model_design():
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1)
    mesh = Mesh(devices,
                axis_names=("data", "attn_dp", "attn_dp_expert", "expert",
                            "model"))
    weights = FusedMoEWeights(
        w13_weight=jnp.ones((2, 4, 3), dtype=jnp.bfloat16),
        w13_weight_scale=jnp.ones((2, 2, 1, 4), dtype=jnp.float32),
        w13_bias=None,
        w2_weight=jnp.ones((2, 3, 4), dtype=jnp.bfloat16),
        w2_weight_scale=jnp.ones((2, 2, 1, 3), dtype=jnp.float32),
        w2_bias=None,
    )

    sharded = shard_moe_weights(weights, MoEBackend.GMM_EP, mesh)
    expected_spec = PartitionSpec(ShardingAxisName.EXPERT)

    assert isinstance(sharded.w13_weight.sharding, NamedSharding)
    assert sharded.w13_weight.sharding.spec == expected_spec
    assert sharded.w13_weight_scale.sharding.spec == expected_spec
    assert sharded.w2_weight.sharding.spec == expected_spec
    assert sharded.w2_weight_scale.sharding.spec == expected_spec


def test_shard_moe_weights_passes_global_shape_for_multihost_gmm_ep():
    """Test that in multi-host Ray mode with num_experts_global,
    general_device_put receives the correct global_shape parameter,
    which triggers make_array_from_process_local_data.
    """
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1)
    mesh = Mesh(devices,
                axis_names=("data", "attn_dp", "attn_dp_expert", "expert",
                            "model"))
    # Local shape: 32 experts (this host's shard)
    weights = FusedMoEWeights(
        w13_weight=jnp.ones((32, 4, 3), dtype=jnp.bfloat16),
        w13_weight_scale=jnp.ones((32, 2, 1, 4), dtype=jnp.float32),
        w13_bias=None,
        w2_weight=jnp.ones((32, 3, 4), dtype=jnp.bfloat16),
        w2_weight_scale=jnp.ones((32, 2, 1, 3), dtype=jnp.float32),
        w2_bias=None,
    )

    with patch.object(envs, 'TPU_MULTIHOST_BACKEND', 'ray'):
        with patch('jax.make_array_from_process_local_data') as mock_make:
            mock_make.side_effect = lambda s, t, g: t

            shard_moe_weights(weights, MoEBackend.GMM_EP, mesh,
                              num_experts_global=256)

        # make_array_from_process_local_data should be called for each
        # non-None weight field (w13_weight, w13_weight_scale, w2_weight,
        # w2_weight_scale) with the correct global_shape
        assert mock_make.call_count == 4
        # All calls should have global_shape starting with (256,)
        for c in mock_make.call_args_list:
            global_shape = c[0][2]  # third positional arg
            assert global_shape[0] == 256


def test_shard_moe_weights_no_global_shape_for_replicated():
    """When ``num_experts_global`` is not passed, the replicated path in
    ``general_device_put`` uses ``make_array_from_process_local_data`` with
    the tensor's own shape as the global shape (no host-side allgather),
    rather than plain ``jax.device_put`` (which would trigger
    ``process_allgather`` for data-consistency verification and OOM on
    large FP8 MoE weights).
    """
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1)
    mesh = Mesh(devices,
                axis_names=("data", "attn_dp", "attn_dp_expert", "expert",
                            "model"))
    weights = FusedMoEWeights(
        w13_weight=jnp.ones((2, 4, 3), dtype=jnp.bfloat16),
        w13_weight_scale=jnp.ones((2, 2, 1, 4), dtype=jnp.float32),
        w13_bias=None,
        w2_weight=jnp.ones((2, 3, 4), dtype=jnp.bfloat16),
        w2_weight_scale=jnp.ones((2, 2, 1, 3), dtype=jnp.float32),
        w2_bias=None,
    )

    with patch.object(envs, 'TPU_MULTIHOST_BACKEND', 'ray'):
        with patch('jax.make_array_from_process_local_data') as mock_make:
            mock_make.side_effect = lambda sharding, t, shape: t
            with patch('jax.device_put') as mock_device_put:
                mock_device_put.side_effect = lambda t, s: t

                shard_moe_weights(weights, MoEBackend.GMM_EP, mesh)

        # Replicated path on multi-host Ray uses
        # make_array_from_process_local_data with global_shape == local shape
        # (four MoE tensors: w13/w2 × weight/scale).
        assert mock_make.call_count == 4
        # Caller passes no global_shape, so device_put is not used for the
        # tensor itself; only the Layout overlay path would trigger it, and
        # this test supplies no Layout.
        assert mock_device_put.call_count == 0


def test_process_moe_weights_skips_layout_constraint_for_gmm_ep():
    weights = FusedMoEWeights(
        w13_weight=jnp.ones((2, 6, 4), dtype=jnp.float32),
        w13_weight_scale=None,
        w13_bias=None,
        w2_weight=jnp.ones((2, 4, 3), dtype=jnp.float32),
        w2_weight_scale=None,
        w2_bias=None,
    )

    with patch(
            'tpu_inference.layers.common.process_weights.moe_weights.with_layout_constraint',
            side_effect=AssertionError('layout constraint should be skipped for GMM_EP')) as mock_layout:
        processed = process_moe_weights(weights, MoEBackend.GMM_EP)

    mock_layout.assert_not_called()
    assert processed.w13_weight.shape[0] == 2
    assert processed.w2_weight.shape[0] == 2


def test_direct_fp8_path():
    """Exercises the VLLM_MOE_SKIP_REQUANTIZATION direct-FP8 shape transform.

    The direct path (``process_fp8_moe_weights_direct``) takes FP8 weights +
    2D block-quantized scales straight from the checkpoint, expands the
    scales to the 4D layout ``process_moe_weights`` produces, and hands
    them to the GMM kernel without going through FP8 → fp32 → FP8.  This
    test verifies:
      * FP8 weight tensors are passed through unchanged (no dequant).
      * 2D block scales are expanded to 4D ``(E, K_blocks, 1, N_full)``.
      * Output is routable through the selected ``MoEBackend``.
    """
    num_experts = 2
    hidden_size = 8           # multiple of block_size_n = 4
    intermediate_size = 12    # multiple of block_size_n = 4
    block_size_n = 4
    block_size_k = 2
    n_blocks_h = hidden_size // block_size_n     # 2
    k_blocks_h = hidden_size // block_size_k     # 4
    n_blocks_i = intermediate_size // block_size_n  # 3
    k_blocks_i = intermediate_size // block_size_k  # 6

    # Checkpoint layout: FP8 weights + 2D block-quantized scales.
    #   w13_weight:       [E, 2*intermediate_size, hidden_size] (gate+up concat)
    #   w13_weight_scale: [E, 2*n_blocks_i, k_blocks_h]   (2D block scale)
    #   w2_weight:        [E, hidden_size, intermediate_size]
    #   w2_weight_scale:  [E, n_blocks_h, k_blocks_i]
    weights = FusedMoEWeights(
        w13_weight=jnp.ones((num_experts, 2 * intermediate_size, hidden_size),
                            dtype=jnp.float8_e4m3fn),
        w13_weight_scale=jnp.ones(
            (num_experts, 2 * n_blocks_i, k_blocks_h), dtype=jnp.float32),
        w13_bias=None,
        w2_weight=jnp.ones((num_experts, hidden_size, intermediate_size),
                           dtype=jnp.float8_e4m3fn),
        w2_weight_scale=jnp.ones((num_experts, n_blocks_h, k_blocks_i),
                                 dtype=jnp.float32),
        w2_bias=None,
    )

    devices = np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1)
    mesh = Mesh(devices,
                axis_names=("data", "attn_dp", "attn_dp_expert", "expert",
                            "model"))

    processed = process_fp8_moe_weights_direct(
        weights,
        moe_backend=MoEBackend.GMM_TP,
        mesh=mesh,
        activation="silu",
        weight_block_size=(block_size_n, block_size_k),
    )

    # FP8 weights pass through (dtype preserved, no dequant happened).
    assert processed.w13_weight.dtype == jnp.float8_e4m3fn
    assert processed.w2_weight.dtype == jnp.float8_e4m3fn

    # Scales are expanded to 4D ``(E, K_blocks, 1, N_full)``.  Direct path
    # leaves the expanded-scale layout invariant with the legacy path so
    # the kernel's gmm_v2 dispatch doesn't need to branch on "origin".
    assert processed.w13_weight_scale.ndim == 4
    assert processed.w13_weight_scale.shape[0] == num_experts
    assert processed.w13_weight_scale.shape[2] == 1
    assert processed.w2_weight_scale.ndim == 4
    assert processed.w2_weight_scale.shape[0] == num_experts
    assert processed.w2_weight_scale.shape[2] == 1
