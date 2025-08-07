import unittest

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from tpu_commons.models.jax.common.base import ParamFactory
from tpu_commons.models.jax.common.moe.deepseek_moe import DeepSeekV3Router
from tpu_commons.models.jax.common.sharding import ShardingConfig


class TestDeepSeekV3Router(unittest.TestCase):

    def setUp(self):
        self.cpu_mesh = Mesh(jax.devices('cpu'), axis_names=('data', ))
        self.tpu_mesh = Mesh(jax.devices('tpu'), axis_names=('data', ))
        self.param_factory = ParamFactory(
            kernel_initializer=nnx.initializers.xavier_normal(),
            scale_initializer=nnx.initializers.ones,
            random_init=True)
        self.sharding_cfg = ShardingConfig()

    def test_get_topk_indices_single_group(self):
        """Test get_topk_indices with single expert group."""
        router = DeepSeekV3Router(mesh=self.cpu_mesh,
                                  param_factory=self.param_factory,
                                  sharding_cfg=self.sharding_cfg,
                                  hidden_size=512,
                                  num_experts=8,
                                  num_experts_per_tok=2,
                                  n_groups=1,
                                  topk_groups=1,
                                  norm_topk_prob=True,
                                  routed_scaling_factor=1.0,
                                  dtype=jnp.bfloat16,
                                  quant=None)
        router.bias_E = jnp.zeros((4, ))

        scores = jnp.array([[0.1, 0.3, 0.2, 0.4]])  # shape: (1, 4)
        indices = router.get_topk_indices(scores)

        # Should return indices of top 2 experts
        expected_indices = jnp.array([[3, 1]])  # experts with scores 0.4, 0.3
        self.assertTrue(jnp.array_equal(indices, expected_indices))

    def test_get_topk_indices_2_groups(self):
        """Test get_topk_indices with 2 expert groups."""
        router = DeepSeekV3Router(mesh=self.cpu_mesh,
                                  param_factory=self.param_factory,
                                  sharding_cfg=self.sharding_cfg,
                                  hidden_size=512,
                                  num_experts=4,
                                  num_experts_per_tok=2,
                                  n_groups=2,
                                  topk_groups=1,
                                  norm_topk_prob=True,
                                  routed_scaling_factor=1.0,
                                  dtype=jnp.bfloat16,
                                  quant=None)
        router.bias_E = jnp.zeros((4, ))

        # 4 experts, 2 groups, 2 experts per group
        scores = jnp.array([[[0.1, 0.3, 0.2, 0.4]]])  # shape: (1, 1, 4)
        indices = router.get_topk_indices(scores)

        # Should return indices of top 2 experts
        expected_indices = jnp.array([[[3, 2]]])
        self.assertTrue(jnp.array_equal(indices, expected_indices))

    def test_router_e2e(self):
        router = DeepSeekV3Router(mesh=self.cpu_mesh,
                                  param_factory=self.param_factory,
                                  sharding_cfg=self.sharding_cfg,
                                  hidden_size=512,
                                  num_experts=8,
                                  num_experts_per_tok=2,
                                  n_groups=2,
                                  topk_groups=1,
                                  norm_topk_prob=True,
                                  routed_scaling_factor=1.0,
                                  dtype=jnp.bfloat16,
                                  quant=None)
        router.generate_kernel(nnx.Rngs(42))
        x = jnp.ones((2, 512))
        weights, indices = router(x, "prefill")
        self.assertEqual(weights.shape, (2, 2))
        self.assertEqual(indices.shape, (2, 2))

        weights, indices = router(x, "generate")
        self.assertEqual(weights.shape, (2, 2))
        self.assertEqual(indices.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
