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
    pytest tests/experimental/ragged_collectives/test_single_direction_reduce_scatter_matmul.py

Requirements:
    - Must be run on a TPU VM with JAX installed
    - Requires multiple TPU devices for multi-device tests
    - JAX with TPU support: pip install jax[tpu]

Output:
    - Pass/fail status for each test
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from . import (reduce_scatter_matmul_reference,
               single_dir_reduce_scatter_matmul_sharded)


def _print_array_comparison(result, expected, name=""):
    """Print diagnostic info for array comparison."""
    # Convert to float32 for consistent formatting (handles bfloat16)
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
    print(
        f"    Result[0,0]: {float(result_np[0,0]):.4f}, Expected[0,0]: {float(expected_np[0,0]):.4f}"
    )

    # Check if result is a fraction of expected (indicates missing accumulations)
    if float(expected_np.mean()) > 1e-6:
        ratio = float(result_np.mean()) / float(expected_np.mean())
        print(f"    Mean ratio (result/expected): {ratio:.4f}")
        if 0.7 < ratio < 0.9:
            print(
                f"    ⚠ Result is ~{ratio*100:.0f}% of expected - may be missing accumulations"
            )


def test_single_dir_reduce_scatter_matmul_smoke():
    """Simple smoke test with small dimensions for quick verification."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_single_dir_reduce_scatter_matmul_smoke: requires at least 2 devices"
        )
        return

    # Small dimensions for fast testing
    m = 128 * num_devices
    k = 128 * num_devices
    n = 128

    x = jnp.ones((m, k), dtype=jnp.bfloat16)
    y = jnp.ones((n, k), dtype=jnp.bfloat16)

    devices = mesh_utils.create_device_mesh((num_devices, ))
    mesh = Mesh(devices, axis_names=("x", ))

    x_sharding = NamedSharding(mesh, P(None, "x"))
    y_sharding = NamedSharding(mesh, P(None, "x"))

    x_sharded = jax.device_put(x, x_sharding)
    y_sharded = jax.device_put(y, y_sharding)

    print(f"  Smoke test: m={m}, k={k}, n={n}, num_devices={num_devices}")
    print(
        f"  Each device has: x[{m}, {k//num_devices}], y[{n}, {k//num_devices}]"
    )
    print(f"  Expected output per device: [{m//num_devices}, {n}]")

    result = single_dir_reduce_scatter_matmul_sharded(x_sharded,
                                                      y_sharded,
                                                      mesh,
                                                      axis_name="x")

    # Reference
    def reference_fn(x_shard, y_shard):
        return reduce_scatter_matmul_reference(x_shard, y_shard, axis_name="x")

    reference_fn_sharded = shard_map(
        reference_fn,
        mesh=mesh,
        in_specs=(P(None, "x"), P(None, "x")),
        out_specs=P("x", None),
        check_rep=False,
    )

    expected = reference_fn_sharded(x_sharded, y_sharded)

    _print_array_comparison(result, expected, "Smoke Test")

    # With all ones, each element should be k (sum of k ones)
    expected_value = k
    actual_value = float(result.addressable_shards[0].data[0, 0])
    print(
        f"  Expected value (k): {expected_value}, Actual: {actual_value:.0f}")

    np.testing.assert_allclose(result, expected, rtol=0.02)
    print("✓ test_single_dir_reduce_scatter_matmul_smoke passed!")


def test_single_dir_reduce_scatter_matmul_basic():
    """Test that single_dir_reduce_scatter_matmul correctly computes matmul + reduce-scatter."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_single_dir_reduce_scatter_matmul_basic: requires at least 2 devices"
        )
        return

    # Test dimensions
    # x: [M, K] with K sharded -> each device has [M, K_shard]
    # y: [N, K] with K sharded -> each device has [N, K_shard]
    # Output: [M, N] with M sharded -> each device has [M_shard, N]
    m = 128 * num_devices
    k = 256 * num_devices  # K will be sharded
    n = 128

    # Create inputs (full shapes, will be sharded)
    x = jnp.ones((m, k), dtype=jnp.bfloat16)
    y = jnp.ones((n, k), dtype=jnp.bfloat16)

    # Setup mesh
    devices = mesh_utils.create_device_mesh((num_devices, ))
    mesh = Mesh(devices, axis_names=("x", ))

    # Shard K dimension for inputs
    x_sharding = NamedSharding(mesh, P(None, "x"))  # K sharded
    y_sharding = NamedSharding(mesh, P(None, "x"))  # K sharded

    x_sharded = jax.device_put(x, x_sharding)
    y_sharded = jax.device_put(y, y_sharding)

    # Perform single-direction reduce-scatter matmul
    result = single_dir_reduce_scatter_matmul_sharded(x_sharded,
                                                      y_sharded,
                                                      mesh,
                                                      axis_name="x")

    # Reference implementation using shard_map
    def reference_fn(x_shard, y_shard):
        return reduce_scatter_matmul_reference(x_shard, y_shard, axis_name="x")

    reference_fn_sharded = shard_map(
        reference_fn,
        mesh=mesh,
        in_specs=(P(None, "x"), P(None, "x")),
        out_specs=P("x", None),
        check_rep=False,
    )

    expected = reference_fn_sharded(x_sharded, y_sharded)

    # Check shapes
    assert (result.shape == expected.shape
            ), f"Shape mismatch: got {result.shape}, expected {expected.shape}"

    _print_array_comparison(result, expected, "Basic Test")

    # Each element should be K (since input is all ones and x @ y.T sums K ones)
    np.testing.assert_allclose(result, expected, rtol=0.02)

    print("✓ test_single_dir_reduce_scatter_matmul_basic passed!")


def test_single_dir_reduce_scatter_matmul_with_random_values():
    """Test with random input values to verify numerical correctness."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_single_dir_reduce_scatter_matmul_with_random_values: requires at least 2 devices"
        )
        return

    m = 128 * num_devices
    k = 256 * num_devices
    n = 128

    key = random.key(42)
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

    result = single_dir_reduce_scatter_matmul_sharded(x_sharded,
                                                      y_sharded,
                                                      mesh,
                                                      axis_name="x")

    def reference_fn(x_shard, y_shard):
        return reduce_scatter_matmul_reference(x_shard, y_shard, axis_name="x")

    reference_fn_sharded = shard_map(
        reference_fn,
        mesh=mesh,
        in_specs=(P(None, "x"), P(None, "x")),
        out_specs=P("x", None),
        check_rep=False,
    )

    expected = reference_fn_sharded(x_sharded, y_sharded)

    np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.1)

    print("✓ test_single_dir_reduce_scatter_matmul_with_random_values passed!")


def test_single_dir_reduce_scatter_matmul_different_dtypes():
    """Test with different data types."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_single_dir_reduce_scatter_matmul_different_dtypes: requires at least 2 devices"
        )
        return

    devices = mesh_utils.create_device_mesh((num_devices, ))
    mesh = Mesh(devices, axis_names=("x", ))

    for dtype in [jnp.float32, jnp.bfloat16]:
        m = 128 * num_devices
        k = 256 * num_devices
        n = 128

        key = random.key(0)
        key1, key2 = random.split(key)

        x = random.uniform(key1, shape=(m, k)).astype(dtype)
        y = random.uniform(key2, shape=(n, k)).astype(dtype)

        x_sharding = NamedSharding(mesh, P(None, "x"))
        y_sharding = NamedSharding(mesh, P(None, "x"))

        x_sharded = jax.device_put(x, x_sharding)
        y_sharded = jax.device_put(y, y_sharding)

        result = single_dir_reduce_scatter_matmul_sharded(x_sharded,
                                                          y_sharded,
                                                          mesh,
                                                          axis_name="x")

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

        assert result.dtype == dtype

        assert result.dtype == dtype, f"Expected dtype {dtype}, got {result.dtype}"

        rtol = 0.05 if dtype == jnp.bfloat16 else 1e-4
        atol = 0.1 if dtype == jnp.bfloat16 else 1e-5
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)

        print(f"  ✓ dtype {dtype} works")

    print("✓ test_single_dir_reduce_scatter_matmul_different_dtypes passed!")


def test_single_dir_reduce_scatter_matmul_larger_dimensions():
    """Test with larger matrix dimensions."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_single_dir_reduce_scatter_matmul_larger_dimensions: requires at least 2 devices"
        )
        return

    m = 256 * num_devices
    k = 512 * num_devices
    n = 256

    key = random.key(123)
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

    result = single_dir_reduce_scatter_matmul_sharded(x_sharded,
                                                      y_sharded,
                                                      mesh,
                                                      axis_name="x")

    def reference_fn(x_shard, y_shard):
        return reduce_scatter_matmul_reference(x_shard, y_shard, axis_name="x")

    reference_fn_sharded = shard_map(
        reference_fn,
        mesh=mesh,
        in_specs=(P(None, "x"), P(None, "x")),
        out_specs=P("x", None),
        check_rep=False,
    )

    expected = reference_fn_sharded(x_sharded, y_sharded)

    np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.1)

    print(
        f"✓ test_single_dir_reduce_scatter_matmul_larger_dimensions (m={m}, k={k}, n={n}) passed!"
    )


def test_output_shape():
    """Test that output shape is correct."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print("⚠ Skipping test_output_shape: requires at least 2 devices")
        return

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
    assert result.shape == (
        m,
        n,
    ), f"Global shape mismatch: got {result.shape}, expected ({m}, {n})"

    # Check per-device local shape
    local_shape = result.addressable_shards[0].data.shape
    expected_local_shape = (m_per_device, n)

    assert (
        local_shape == expected_local_shape
    ), f"Local shape mismatch: got {local_shape}, expected {expected_local_shape}"

    print(
        f"✓ test_output_shape passed! (local shape per device: {local_shape})")


def test_different_device_counts():
    """Test with different device counts."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_different_device_counts: requires at least 2 devices"
        )
        return

    # Test with different device counts that divide num_devices
    # Skip the full device count since it's already tested by other tests
    device_counts = [
        dc for dc in [2, 4, 8] if dc < num_devices and num_devices % dc == 0
    ]

    for dc in device_counts:
        # Clear caches to avoid conflicts between different mesh configurations
        jax.clear_caches()

        m = 128 * dc
        k = 256 * dc
        n = 128

        x = jnp.ones((m, k), dtype=jnp.bfloat16)
        y = jnp.ones((n, k), dtype=jnp.bfloat16)

        all_devices = jax.devices()
        selected_devices = np.array(all_devices[:dc])
        mesh = Mesh(selected_devices, axis_names=("x", ))

        x_sharding = NamedSharding(mesh, P(None, "x"))
        y_sharding = NamedSharding(mesh, P(None, "x"))

        x_sharded = jax.device_put(x, x_sharding)
        y_sharded = jax.device_put(y, y_sharding)

        result = single_dir_reduce_scatter_matmul_sharded(x_sharded,
                                                          y_sharded,
                                                          mesh,
                                                          axis_name="x")

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

        np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.1)
        print(f"  ✓ device_count={dc} works")

    print("✓ test_different_device_counts passed!")


def test_slice_mapping():
    """Verify that each device outputs the correct M_shard.

    Device i should own M_shard i (rows i*m_per_device to (i+1)*m_per_device).
    """
    num_devices = jax.device_count()

    if num_devices < 2:
        print("⚠ Skipping test_slice_mapping: requires at least 2 devices")
        return

    m = 128 * num_devices
    k = 256 * num_devices
    n = 128
    m_per_device = m // num_devices

    # Create x where each m_slice has a unique value pattern
    # Slice i has value (i+1) in all elements
    x = jnp.zeros((m, k), dtype=jnp.bfloat16)
    for i in range(num_devices):
        start = i * m_per_device
        end = (i + 1) * m_per_device
        x = x.at[start:end, :].set(float(i + 1))

    # y is all ones so matmul just sums along K_shard
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

    print(f"  Verifying slice mapping (num_devices={num_devices}):")
    print("  Expected: device i outputs slice i with value (i+1) * k")

    all_correct = True
    for device_idx, shard in enumerate(result.addressable_shards):
        actual_value = float(shard.data[0, 0])
        # After reduce-scatter: value should be (device_idx + 1) * k
        # because x[slice i] = i+1 and matmul sums k elements
        expected_value = (device_idx + 1) * k

        print(f"  Device {device_idx}: got value {actual_value:.0f}, "
              f"expected {expected_value}")

        if abs(actual_value -
               expected_value) > expected_value * 0.05:  # 5% tolerance
            all_correct = False
            print(f"    ⚠ MISMATCH on device {device_idx}")

    if all_correct:
        print("✓ test_slice_mapping passed!")
    else:
        raise AssertionError(
            "test_slice_mapping FAILED - slice mapping is wrong")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Running Single-Direction Reduce-Scatter Matmul Tests")
    print(f"Number of devices: {jax.device_count()}")
    print(f"Device type: {jax.devices()[0].device_kind}")
    print("=" * 70 + "\n")

    # Multi-device tests
    print("--- Multi-Device Tests ---")
    test_single_dir_reduce_scatter_matmul_smoke()
    test_single_dir_reduce_scatter_matmul_basic()
    test_single_dir_reduce_scatter_matmul_with_random_values()
    test_single_dir_reduce_scatter_matmul_different_dtypes()
    test_single_dir_reduce_scatter_matmul_larger_dimensions()
    test_output_shape()
    test_different_device_counts()
    test_slice_mapping()

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
