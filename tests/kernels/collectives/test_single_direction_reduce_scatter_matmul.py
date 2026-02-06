# SPDX-License-Identifier: Apache-2.0
"""Test script for the single-direction reduce-scatter matmul kernel.

This script tests the single-direction reduce-scatter matmul kernel with
delayed computation for correctness by comparing its output with the
reference implementation using JAX's lax.psum_scatter operation.

The kernel implements a ring-based reduce-scatter fused with matmul where:
- Each device has x[M, K_shard] and y[N, K_shard] (K is sharded)
- Output is x @ y.T with M dimension scattered across devices
- Uses delayed computation: matmul for each shard is computed only when
  the accumulator for that shard arrives, allowing compute-communication overlap

Usage:
    pytest tests/experimental/test_single_direction_reduce_scatter_matmul.py

Requirements:
    - Must be run on a TPU VM with JAX installed
    - Requires multiple TPU devices for multi-device tests
    - JAX with TPU support: pip install jax[tpu]
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from experimental.reduce_scatter_matmul.single_direction_reduce_scatter_matmul import (
    reduce_scatter_matmul_reference, single_dir_reduce_scatter_matmul_sharded)
from jax import random
from jax._src import test_util as jtu
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def _get_available_device_counts():
    """Get list of valid device counts for testing."""
    num_devices = jax.device_count()
    # Return device counts that divide evenly into total devices
    return [
        dc for dc in [2, 4, 8] if dc <= num_devices and num_devices % dc == 0
    ]


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
    print(f"    Max abs diff: {float(diff.max()):.4f}")
    print(f"    Max rel diff: {float(rel_diff.max()):.4f}")


class SingleDirectionReduceScatterMatmulTest(parameterized.TestCase):
    """Parameterized tests for single-direction reduce-scatter matmul kernel."""

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
        m = 128 * device_count * shape_multiplier
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
        result = single_dir_reduce_scatter_matmul_sharded(x_sharded,
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

        m = 128 * num_devices
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

        result = single_dir_reduce_scatter_matmul_sharded(x_sharded,
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
        """Verify that each device outputs the correct M_shard."""
        num_devices = jax.device_count()

        if num_devices < 2:
            self.skipTest("Requires at least 2 devices")

        m = 128 * num_devices
        k = 256 * num_devices
        n = 128
        m_per_device = m // num_devices

        # Create x where each m_slice has a unique value pattern
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

        result = single_dir_reduce_scatter_matmul_sharded(x_sharded,
                                                          y_sharded,
                                                          mesh,
                                                          axis_name="x")

        for device_idx, shard in enumerate(result.addressable_shards):
            actual_value = float(shard.data[0, 0])
            expected_value = (device_idx + 1) * k
            self.assertAlmostEqual(
                actual_value,
                expected_value,
                delta=expected_value * 0.05,
                msg=f"Slice mapping wrong on device {device_idx}")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
