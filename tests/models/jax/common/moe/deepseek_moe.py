import unittest

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from tpu_commons.models.jax.common.base import ParamFactory
from tpu_commons.models.jax.common.sharding import ShardingConfig
from tpu_commons.models.jax.common.moe.deepseek_moe import (DeepSeekV3Router,
                                             DeepSeekV3RoutingConfig)


class TestDeepSeekV3Router(unittest.TestCase):

    def setUp(self):
        self.cpu_mesh = Mesh(jax.devices('cpu'), axis_names=('data', ))
        self.tpu_mesh = Mesh(jax.devices('tpu'), axis_names=('data', ))
        self.param_factory = ParamFactory(
            kernel_initializer=nnx.initializers.xavier_normal(),
            scale_initializer=nnx.initializers.ones)
        self.sharding_cfg = ShardingConfig()

    def test_get_topk_indices_single_group(self):
        """Test get_topk_indices with single expert group."""
        router_config = DeepSeekV3RoutingConfig(
            hidden_size=512,
            n_routed_experts=8,
            num_experts_per_token=2,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            norm_topk_prob=True,
            vllm_config=None,
            dtype=jnp.bfloat16,
        )

        router = DeepSeekV3Router(router_config,
                                  self.cpu_mesh,
                                  self.param_factory,
                                  self.sharding_cfg,
                                  quant=None)
        router.bias_E = jnp.zeros((4, ))

        scores = jnp.array([[0.1, 0.3, 0.2, 0.4]])  # shape: (1, 4)
        indices = router.get_topk_indices(scores)

        # Should return indices of top 2 experts
        expected_indices = jnp.array([[3, 1]])  # experts with scores 0.4, 0.3
        self.assertTrue(jnp.array_equal(indices, expected_indices))

    def test_get_topk_indices_2_groups(self):
        """Test get_topk_indices with 2 expert groups."""
        router_config = DeepSeekV3RoutingConfig(
            hidden_size=512,
            n_routed_experts=4,
            num_experts_per_token=2,
            n_group=2,
            topk_group=1,
            routed_scaling_factor=1.0,
            norm_topk_prob=True,
            vllm_config=None,
            dtype=jnp.bfloat16,
        )
        router = DeepSeekV3Router(router_config,
                                  self.cpu_mesh,
                                  self.param_factory,
                                  self.sharding_cfg,
                                  quant=None)
        router.bias_E = jnp.zeros((4, ))

        # 4 experts, 2 groups, 2 experts per group
        scores = jnp.array([[[0.1, 0.3, 0.2, 0.4]]])  # shape: (1, 1, 4)
        indices = router.get_topk_indices(scores)

        # Should return indices of top 2 experts
        expected_indices = jnp.array([[[3, 2]]])
        self.assertTrue(jnp.array_equal(indices, expected_indices))

    def test_router_e2e(self):
        router_config = DeepSeekV3RoutingConfig(
            hidden_size=512,
            n_routed_experts=8,
            num_experts_per_token=2,
            n_group=2,
            topk_group=1,
            routed_scaling_factor=1.0,
            norm_topk_prob=True,
            vllm_config=None,
            dtype=jnp.bfloat16,
        )
        router = DeepSeekV3Router(router_config,
                                  self.tpu_mesh,
                                  self.param_factory,
                                  self.sharding_cfg,
                                  quant=None)
        router.generate_kernel(nnx.Rngs(42))
        x = jnp.ones((1, 2, 512))
        weights, indices = router(x, "prefill")
        self.assertEqual(weights.shape, (1, 2, 2))
        self.assertEqual(indices.shape, (1, 2, 2))

        weights, indices = router(x, "generate")
        self.assertEqual(weights.shape, (1, 2, 2))
        self.assertEqual(indices.shape, (1, 2, 2))


if __name__ == '__main__':
    unittest.main()
