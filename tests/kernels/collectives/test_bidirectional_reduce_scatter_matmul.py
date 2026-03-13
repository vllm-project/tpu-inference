# SPDX-License-Identifier: Apache-2.0
"""Test script for the bidirectional reduce-scatter matmul kernel.

This script tests the bidirectional reduce-scatter matmul kernel with M-split
algorithm for correctness by comparing its output with a reference implementation.

The kernel implements a bidirectional ring-based reduce-scatter fused with matmul where:
- Each device has x[M, K_shard] and y[N, K_shard] (K is sharded)
- Output is x @ y.T with M dimension scattered across devices
- Uses M-split algorithm: M split into N blocks, each block into TOP/BOT halves
- LEFT direction handles all TOP halves, RIGHT handles all BOT halves
- Uses delayed computation for compute-communication overlap

Usage:
    pytest tests/experimental/test_bidirection_reduce_scatter_matmul.py

Requirements:
    - Must be run on a TPU VM with JAX installed
    - Requires multiple TPU devices for multi-device tests
    - JAX with TPU support: pip install jax[tpu]
"""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from experimental.reduce_scatter_matmul.bidirection_reduce_scatter_matmul import \
    bidirectional_reduce_scatter_matmul
from jax import lax, random
from jax._src import test_util as jtu
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def reduce_scatter_matmul_reference(
    x_shard: jax.Array,
    y_shard: jax.Array,
    axis_name: str = "x",
) -> jax.Array:
    """Reference implementation using lax.psum_scatter.

    Args:
        x_shard: Input tensor [M, K_shard] where K is sharded
        y_shard: Weight tensor [N, K_shard] where K is sharded
        axis_name: Name of the device axis

    Returns:
        Output tensor [M_shard, N] where M is scattered
    """
    # Local matmul: [M, K_shard] @ [K_shard, N] = [M, N]
    local_result = jnp.dot(x_shard, y_shard.T)

    # Reduce-scatter along M dimension
    # psum_scatter with tiled=True: scatter the result evenly across devices
    result = lax.psum_scatter(local_result,
                              axis_name,
                              scatter_dimension=0,
                              tiled=True)

    return result


def bidirectional_reduce_scatter_matmul_sharded(
    x: jax.Array,
    y: jax.Array,
    mesh: Mesh,
    axis_name: str = "x",
    bm: int = 128,
    bn: int = 128,
    bk: int = 128,
) -> jax.Array:
    """Wrapper that applies sharding to bidirectional_reduce_scatter_matmul.

    Args:
        x: Input tensor [M, K] with K sharded
        y: Weight tensor [N, K] with K sharded
        mesh: JAX mesh for device placement
        axis_name: Name of the device axis
        bm: Block size for M dimension
        bn: Block size for N dimension
        bk: Block size for K dimension

    Returns:
        Output tensor [M, N] with M sharded
    """

    @jax.jit
    def kernel_fn(x_shard, y_shard):
        return bidirectional_reduce_scatter_matmul(x_shard,
                                                   y_shard,
                                                   axis_name=axis_name,
                                                   bm=bm,
                                                   bn=bn,
                                                   bk=bk)

    sharded_fn = shard_map(
        kernel_fn,
        mesh=mesh,
        in_specs=(P(None, "x"), P(None, "x")),  # K sharded
        out_specs=P("x", None),  # M sharded
        check_rep=False,
    )

    return sharded_fn(x, y)


def _print_array_comparison(result, expected, name=""):
    """Print diagnostic info for array comparison."""
    result_np = np.asarray(result).astype(np.float32)
    expected_np = np.asarray(expected).astype(np.float32)

    diff = np.abs(result_np - expected_np)
    rel_diff = diff / (np.abs(expected_np) + 1e-8)

    print(f"\n  {name} Comparison:")
    print(
        f"    Result shape: {result_np.shape}, dtype: {np.asarray(result).dtype}"
    )
    print(
        f"    Expected shape: {expected_np.shape}, dtype: {np.asarray(expected).dtype}"
    )
    print(
        f"    Result range: [{float(result_np.min()):.4f}, {float(result_np.max()):.4f}]"
    )
    print(
        f"    Expected range: [{float(expected_np.min()):.4f}, {float(expected_np.max()):.4f}]"
    )
    print(f"    Max abs diff: {float(diff.max()):.6f}")
    print(f"    Max rel diff: {float(rel_diff.max()):.6f}")


class BidirectionalReduceScatterMatmulTest(parameterized.TestCase):
    """Parameterized tests for bidirectional reduce-scatter matmul kernel."""

    @parameterized.product(
        dtype=[jnp.bfloat16, jnp.float32],
        shape_multiplier=[1, 2, 4],  # Multiplier for base dimensions
        device_count=[2, 4, 8],
        use_random_values=[False, True],
    )
    def test_reduce_scatter_matmul(self, dtype, shape_multiplier, device_count,
                                   use_random_values):
        """Parameterized test covering dtypes, shapes, and device counts."""
        num_devices = jax.device_count()

        if device_count > num_devices or num_devices % device_count != 0:
            self.skipTest(
                f"Requires {device_count} devices, but only {num_devices} available or not divisible"
            )

        # Clear caches to avoid conflicts between different mesh configurations
        jax.clear_caches()

        # Base dimensions scaled by multiplier
        # For bidirectional kernel: M_block = 256 * multiplier, M_half_block = 128 * multiplier
        # Must be divisible by bm=128
        m = 256 * device_count * shape_multiplier
        k = 128 * device_count * shape_multiplier
        n = 128 * shape_multiplier

        # Create inputs
        if use_random_values:
            key = random.key(42)
            key1, key2 = random.split(key)
            x = random.uniform(key1, shape=(m, k),
                               dtype=jnp.float32).astype(dtype)
            y = random.uniform(key2, shape=(n, k),
                               dtype=jnp.float32).astype(dtype)
        else:
            x = jnp.ones((m, k), dtype=dtype)
            y = jnp.ones((n, k), dtype=dtype)

        # Setup mesh with subset of devices
        all_devices = jax.devices()
        selected_devices = np.array(all_devices[:device_count])
        mesh = Mesh(selected_devices, axis_names=("x", ))

        x_sharding = NamedSharding(mesh, P(None, "x"))
        y_sharding = NamedSharding(mesh, P(None, "x"))

        x_sharded = jax.device_put(x, x_sharding)
        y_sharded = jax.device_put(y, y_sharding)

        print(
            f"\n  Test: dtype={dtype}, m={m}, k={k}, n={n}, devices={device_count}, random={use_random_values}"
        )

        # Run kernel
        result = bidirectional_reduce_scatter_matmul_sharded(x_sharded,
                                                             y_sharded,
                                                             mesh,
                                                             axis_name="x")

        # Reference implementation
        def reference_fn(x_shard, y_shard):
            return reduce_scatter_matmul_reference(x_shard,
                                                   y_shard,
                                                   axis_name="x")

        reference_fn_sharded = shard_map(
            reference_fn,
            mesh=mesh,
            in_specs=(P(None, "x"), P(None, "x")),
            out_specs=P("x", None),
            check_rep=False,
        )

        expected = reference_fn_sharded(x_sharded, y_sharded)

        # Verify shape
        self.assertEqual(result.shape, expected.shape)

        # Verify dtype
        self.assertEqual(result.dtype, dtype)

        # Verify values with appropriate tolerance
        rtol = 0.05 if dtype == jnp.bfloat16 else 1e-4
        atol = 0.1 if dtype == jnp.bfloat16 else 1e-5
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)

    def test_output_shape(self):
        """Test that output shape is correct."""
        num_devices = jax.device_count()

        if num_devices < 2:
            self.skipTest("Requires at least 2 devices")

        m = 256 * num_devices
        k = 256 * num_devices
        n = 128
        m_per_device = m // num_devices

        x = jnp.ones((m, k), dtype=jnp.bfloat16)
        y = jnp.ones((n, k), dtype=jnp.bfloat16)

        devices = mesh_utils.create_device_mesh((num_devices, ))
        mesh = Mesh(devices, axis_names=("x", ))

        x_sharding = NamedSharding(mesh, P(None, "x"))
        y_sharding = NamedSharding(mesh, P(None, "x"))

        x_sharded = jax.device_put(x, x_sharding)
        y_sharded = jax.device_put(y, y_sharding)

        result = bidirectional_reduce_scatter_matmul_sharded(x_sharded,
                                                             y_sharded,
                                                             mesh,
                                                             axis_name="x")

        # Global shape should be [M, N] with M sharded
        self.assertEqual(result.shape, (m, n))

        # Check per-device local shape
        local_shape = result.addressable_shards[0].data.shape
        expected_local_shape = (m_per_device, n)
        self.assertEqual(local_shape, expected_local_shape)

    def test_slice_mapping(self):
        """Verify that each device outputs the correct M_block."""
        num_devices = jax.device_count()

        if num_devices < 2:
            self.skipTest("Requires at least 2 devices")

        m = 256 * num_devices
        k = 256 * num_devices
        n = 128
        m_per_device = m // num_devices

        # Create x where each m_block has a unique value pattern
        x = jnp.zeros((m, k), dtype=jnp.bfloat16)
        for i in range(num_devices):
            start = i * m_per_device
            end = (i + 1) * m_per_device
            x = x.at[start:end, :].set(float(i + 1))

        y = jnp.ones((n, k), dtype=jnp.bfloat16)

        devices = mesh_utils.create_device_mesh((num_devices, ))
        mesh = Mesh(devices, axis_names=("x", ))

        x_sharding = NamedSharding(mesh, P(None, "x"))
        y_sharding = NamedSharding(mesh, P(None, "x"))

        x_sharded = jax.device_put(x, x_sharding)
        y_sharded = jax.device_put(y, y_sharding)

        result = bidirectional_reduce_scatter_matmul_sharded(x_sharded,
                                                             y_sharded,
                                                             mesh,
                                                             axis_name="x")

        for device_idx, shard in enumerate(result.addressable_shards):
            # Check both TOP and BOT halves
            top_value = float(shard.data[0, 0])
            bot_value = float(shard.data[m_per_device // 2, 0])
            expected_value = (device_idx + 1) * k

            self.assertAlmostEqual(
                top_value,
                expected_value,
                delta=expected_value * 0.05,
                msg=f"TOP slice mapping wrong on device {device_idx}")
            self.assertAlmostEqual(
                bot_value,
                expected_value,
                delta=expected_value * 0.05,
                msg=f"BOT slice mapping wrong on device {device_idx}")

    def test_top_bot_consistency(self):
        """Test that TOP and BOT halves are consistent with reference."""
        num_devices = jax.device_count()

        if num_devices < 2:
            self.skipTest("Requires at least 2 devices")

        m = 256 * num_devices
        k = 256 * num_devices
        n = 128
        m_per_device = m // num_devices
        m_half = m_per_device // 2

        key = random.key(999)
        key1, key2 = random.split(key)

        x = random.uniform(key1, shape=(m, k),
                           dtype=jnp.float32).astype(jnp.bfloat16)
        y = random.uniform(key2, shape=(n, k),
                           dtype=jnp.float32).astype(jnp.bfloat16)

        devices = mesh_utils.create_device_mesh((num_devices, ))
        mesh = Mesh(devices, axis_names=("x", ))

        x_sharding = NamedSharding(mesh, P(None, "x"))
        y_sharding = NamedSharding(mesh, P(None, "x"))

        x_sharded = jax.device_put(x, x_sharding)
        y_sharded = jax.device_put(y, y_sharding)

        result = bidirectional_reduce_scatter_matmul_sharded(x_sharded,
                                                             y_sharded,
                                                             mesh,
                                                             axis_name="x")

        # Reference implementation
        def reference_fn(x_shard, y_shard):
            return reduce_scatter_matmul_reference(x_shard,
                                                   y_shard,
                                                   axis_name="x")

        reference_fn_sharded = shard_map(
            reference_fn,
            mesh=mesh,
            in_specs=(P(None, "x"), P(None, "x")),
            out_specs=P("x", None),
            check_rep=False,
        )

        expected = reference_fn_sharded(x_sharded, y_sharded)

        # Check each device's TOP and BOT halves match reference
        for device_idx, (result_shard, expected_shard) in enumerate(
                zip(result.addressable_shards, expected.addressable_shards)):
            result_data = np.asarray(result_shard.data).astype(np.float32)
            expected_data = np.asarray(expected_shard.data).astype(np.float32)

            top_result = result_data[:m_half, :]
            bot_result = result_data[m_half:, :]
            top_expected = expected_data[:m_half, :]
            bot_expected = expected_data[m_half:, :]

            np.testing.assert_allclose(
                top_result,
                top_expected,
                rtol=0.05,
                atol=0.1,
                err_msg=f"TOP half mismatch on device {device_idx}")
            np.testing.assert_allclose(
                bot_result,
                bot_expected,
                rtol=0.05,
                atol=0.1,
                err_msg=f"BOT half mismatch on device {device_idx}")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
