import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtu
from jax.sharding import Mesh

from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe, ref_moe

jax.config.parse_flags_with_absl()


def gen_moe_inputs(
    dtype,
    top_k,
    num_experts,
    hidden_size,
    intermediate_size,
    num_tokens,
    *,
    seed=1234,
):
    key = jax.random.key(seed)
    k0, k1, k2, k4, k5 = jax.random.split(key, 5)
    a = jax.random.normal(k0, (num_tokens, hidden_size),
                          dtype=jnp.float32).astype(dtype) / 10
    w1 = (jax.random.normal(
        k1,
        (num_experts, 2, hidden_size, intermediate_size),
        dtype=jnp.float32,
    ) / 10).astype(dtype)
    w2 = (jax.random.normal(k2, (num_experts, intermediate_size, hidden_size),
                            dtype=jnp.float32) / 10).astype(dtype)
    gating_output = (
        jax.random.normal(k4, (num_tokens, num_experts), dtype=jnp.float32) +
        jnp.arange(num_tokens * num_experts, dtype=jnp.float32).reshape(
            num_tokens, num_experts) / 100)
    # To generate unique top-k!
    top_k_indices = jax.random.randint(k5, (num_tokens, top_k),
                                       minval=0,
                                       maxval=num_experts - 1,
                                       dtype=jnp.int32)
    one_hot = (jnp.sum(
        jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.float32),
        axis=1,
    ) * 10)
    gating_output = (gating_output + one_hot).astype(dtype)
    return a, w1, w2, gating_output


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
        self.mesh = Mesh(np.array(self.mesh_devices).reshape(1, -1),
                         axis_names=("data", "model"))

    def test_basic(self):
        dtype = jnp.bfloat16
        top_k = 2
        num_experts = 16
        hidden_size = 256
        intermediate_size = 256
        num_tokens = 8 * 2

        a, w1, w2, gating_output = gen_moe_inputs(
            dtype,
            top_k,
            num_experts,
            hidden_size,
            intermediate_size,
            num_tokens,
        )

        actual = jax.block_until_ready(
            fused_ep_moe(
                mesh=self.mesh,
                tokens=a,
                w1=w1,
                w2=w2,
                gating_output=gating_output,
                top_k=top_k,
                bt=32,
                bf=512,
                bd1=512,
                bd2=512,
                btc=32,
                bfc=256,
                bd1c=256,
                bd2c=256,
            ))
        expected = ref_moe(a, w1, w2, gating_output, top_k)
        self.assertAllClose(expected, actual, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
