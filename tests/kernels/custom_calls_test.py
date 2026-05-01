import timeit
from functools import partial
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax import shard_map
from tpu_inference.kernels.custom_calls.kernel import xpose_full

def benchmark_op(name, op_func, input_data, number=10):
    """Utility to benchmark a JAX operation and print results."""
    # Warmup
    res = op_func(input_data)
    if isinstance(res, (list, tuple)):
        res[0].block_until_ready()
    else:
        res.block_until_ready()
    
    def sync_op():
        out = op_func(input_data)
        if isinstance(out, (list, tuple)):
            out[0].block_until_ready()
        else:
            out.block_until_ready()

    t = timeit.timeit(sync_op, number=number)
    avg_time = t / number
    print(f"\n{name} (shape {input_data.shape}): Mean execution time: {avg_time:.6f}s")
    return res

def xpose_full_wrapper(x, transpose_axes):
    """Helper to extract the first output from xpose_full."""
    return xpose_full(x, transpose_axes=transpose_axes)[0]

class CustomCallsTest(parameterized.TestCase):

    @parameterized.parameters(
        dict(shape=(1024, 1024), transpose_axes=(1, 0)),
        dict(shape=(32, 64, 128), transpose_axes=(2, 0, 1)),
        dict(shape=(8, 16, 32, 64), transpose_axes=(3, 2, 1, 0)),
    )
    def test_xpose_full(self, shape, transpose_axes):
        input_data = jnp.ones(shape, dtype=jnp.float32)
        
        name = f"xpose_full_{len(shape)}d"
        result = benchmark_op(name, lambda x: xpose_full(x, transpose_axes=transpose_axes), input_data)
        
        expected = jnp.transpose(input_data, transpose_axes)
        
        # Validation
        self.assertEqual(result[0].shape, expected.shape)
        self.assertTrue(jnp.allclose(result[0], expected))

    def test_xpose_sharded_mla(self):
        # Mimic q_nope scenario from flash_attn_mla.py
        # q_nope shape (N, B, L) where N=Heads, B=Batch, L=LoraRank
        num_devices = len(jax.devices())
        N, B, L = 16, 128 * num_devices, 512
        shape = (N, B, L)
        input_data = jnp.ones(shape, dtype=jnp.float32)
        
        mesh = Mesh(jax.devices(), ('model',))
        transpose_axes = (1, 0, 2)
        
        # Define the sharded operation using partial instead of lambda
        # to be more JAX-idiomatic and avoid potential recompilation issues.
        sharded_xpose_fn = shard_map(
            partial(xpose_full_wrapper, transpose_axes=transpose_axes),
            mesh=mesh,
            in_specs=P(None, 'model', None),
            out_specs=P('model', None, None),
            check_vma=False
        )
        
        @jax.jit
        def run_sharded_xpose(x):
            return sharded_xpose_fn(x)

        result = benchmark_op("xpose_sharded_mla", run_sharded_xpose, input_data)
        
        expected = jnp.transpose(input_data, transpose_axes)
        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(jnp.allclose(result, expected))

    def test_native_comparison(self):
        shape = (1024, 1024)
        input_data = jnp.ones(shape, dtype=jnp.float32)
        benchmark_op("native_jax_transpose", lambda x: jnp.transpose(x, (1, 0)), input_data)

if __name__ == "__main__":
    absltest.main()
