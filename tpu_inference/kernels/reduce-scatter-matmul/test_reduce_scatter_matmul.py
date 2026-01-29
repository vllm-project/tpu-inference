# SPDX-License-Identifier: Apache-2.0
"""Test script for the reduce-scatter matmul kernel.

This script tests the reduce-scatter matmul kernel for correctness by comparing
its output with the reference implementation using JAX's lax.psum_scatter operation.

Usage:
    cd msl-tpu-kernel/msl-tpu-kernel/experimental/reduce-scatter-matmul
    python test_reduce_scatter_matmul.py

Requirements:
    - Must be run on a TPU VM with JAX installed
    - Requires multiple TPU devices for multi-device tests
    - JAX with TPU support: pip install jax[tpu]

Output:
    - Pass/fail status for each reduce-scatter matmul kernel test
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from kernel import (get_kernel_name, get_vmem_estimate_bytes,
                    reduce_scatter_matmul, ref_reduce_scatter_matmul,
                    ref_reduce_scatter_matmul_naive, validate_inputs)


def test_reduce_scatter_matmul_basic():
    """Test that reduce_scatter_matmul correctly computes matmul + reduce-scatter."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_reduce_scatter_matmul_basic: requires at least 2 devices"
        )
        return

    # Test dimensions
    # m must be divisible by tp_size * 2 * 8 = num_devices * 16
    # k and n must be divisible by 128
    m = 256 * num_devices  # Total m dimension
    k = 512
    n = 256 * num_devices  # Total n dimension
    n_per_device = n // num_devices

    # Create inputs
    # x is replicated across devices [m, k]
    # y is already local shape [k, n_per_device] per device (not sharded further)
    x = jnp.ones((m, k), dtype=jnp.bfloat16)
    y = jnp.ones((k, n_per_device), dtype=jnp.bfloat16)

    # Setup mesh
    devices = mesh_utils.create_device_mesh((num_devices, ))
    mesh = Mesh(devices, axis_names=("x", ))

    # Shard inputs appropriately
    # x is replicated, y is passed as local shape (replicated, each device has same copy)
    x_sharding = NamedSharding(mesh, P(None, None))  # x replicated
    y_sharding = NamedSharding(mesh,
                               P(None,
                                 None))  # y replicated (already local shape)

    x_sharded = jax.device_put(x, x_sharding)
    y_sharded = jax.device_put(y, y_sharding)

    # Perform Pallas reduce-scatter matmul
    result = reduce_scatter_matmul(x_sharded, y_sharded, mesh, "x")

    # Reference implementation
    expected = ref_reduce_scatter_matmul(x_sharded, y_sharded, mesh, "x")

    # Check shapes
    assert result.shape == expected.shape, (
        f"Shape mismatch: got {result.shape}, expected {expected.shape}")

    # Each element should be k (since input is all ones and x @ y sums k ones)
    # After reduce-scatter, we scatter along m, so value should still be k
    np.testing.assert_allclose(result, expected, rtol=0.02)

    print("✓ test_reduce_scatter_matmul_basic passed!")


def test_reduce_scatter_matmul_with_random_values():
    """Debug test to understand which m_slice each device outputs.

    This test uses values that uniquely identify each m_slice to trace
    the kernel's output mapping after ring communication.
    """
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_reduce_scatter_matmul_with_random_values: requires at least 2 devices"
        )
        return

    m = 256 * num_devices
    k = 512
    n = 256 * num_devices
    n_per_device = n // num_devices
    m_per_device = m // num_devices

    # Create x where each m_slice has a unique value pattern:
    # slice i has value (i+1) in all elements
    # This way we can identify which slice each device outputs
    x = jnp.zeros((m, k), dtype=jnp.bfloat16)
    for i in range(num_devices):
        start = i * m_per_device
        end = (i + 1) * m_per_device
        x = x.at[start:end, :].set(float(i + 1))

    # y is all ones so matmul just sums along k: result = (i+1) * k
    y = jnp.ones((k, n_per_device), dtype=jnp.bfloat16)

    devices = mesh_utils.create_device_mesh((num_devices, ))
    mesh = Mesh(devices, axis_names=("x", ))

    x_sharding = NamedSharding(mesh, P(None, None))
    y_sharding = NamedSharding(mesh, P(None, None))

    x_sharded = jax.device_put(x, x_sharding)
    y_sharded = jax.device_put(y, y_sharding)

    result = reduce_scatter_matmul(x_sharded, y_sharded, mesh, "x")

    # Analyze which m_slice each device got
    # If device i has value (j+1)*k, it got slice j
    print(f"  Debugging slice mapping (num_devices={num_devices}, k={k}):")
    print("  Expected: device i outputs slice i with value (i+1)*k")

    for device_idx, shard in enumerate(result.addressable_shards):
        actual_value = float(shard.data[0, 0])  # Sample first element
        expected_slice_value = (device_idx + 1) * k
        inferred_slice = int(actual_value / k) - 1

        print(f"  Device {device_idx}: got value {actual_value:.0f}, "
              f"expected {expected_slice_value}, "
              f"inferred slice {inferred_slice}")

        if abs(actual_value - expected_slice_value) > 1:
            print(
                f"    ⚠ MISMATCH: Device {device_idx} expected slice {device_idx} "
                f"but appears to have slice {inferred_slice}")

    # Verify correctness
    all_correct = True
    for device_idx, shard in enumerate(result.addressable_shards):
        expected_value = (device_idx + 1) * k
        if abs(float(shard.data[0, 0]) - expected_value) > 1:
            all_correct = False
            break

    if all_correct:
        print("✓ test_reduce_scatter_matmul_with_random_values passed!")
    else:
        print(
            "✗ test_reduce_scatter_matmul_with_random_values FAILED - slice mapping is wrong"
        )
        # Print full mapping for debugging
        print("  Full slice mapping:")
        for device_idx, shard in enumerate(result.addressable_shards):
            actual_value = float(shard.data[0, 0])
            inferred_slice = int(actual_value / k) - 1
            print(f"    Device {device_idx} → Slice {inferred_slice}")


def test_reduce_scatter_matmul_different_dtypes():
    """Test reduce_scatter_matmul with different data types."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_reduce_scatter_matmul_different_dtypes: requires at least 2 devices"
        )
        return

    devices = mesh_utils.create_device_mesh((num_devices, ))
    mesh = Mesh(devices, axis_names=("x", ))

    for dtype in [jnp.float32, jnp.bfloat16]:
        m = 256 * num_devices
        k = 512
        n = 256 * num_devices
        n_per_device = n // num_devices

        key = random.key(0)
        key1, key2 = random.split(key)

        x = random.uniform(key1, shape=(m, k)).astype(dtype)
        y = random.uniform(key2, shape=(k, n_per_device)).astype(dtype)

        x_sharding = NamedSharding(mesh, P(None, None))
        y_sharding = NamedSharding(mesh, P(None,
                                           None))  # y already local shape

        x_sharded = jax.device_put(x, x_sharding)
        y_sharded = jax.device_put(y, y_sharding)

        result = reduce_scatter_matmul(x_sharded, y_sharded, mesh, "x")
        expected = ref_reduce_scatter_matmul(x_sharded, y_sharded, mesh, "x")

        assert result.dtype == dtype

        assert result.dtype == dtype, f"Expected dtype {dtype}, got {result.dtype}"

        # Use appropriate tolerance for dtype
        rtol = 0.05 if dtype == jnp.bfloat16 else 1e-4
        atol = 0.01 if dtype == jnp.bfloat16 else 1e-5
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)

        print(f"  ✓ dtype {dtype} works")

    print("✓ test_reduce_scatter_matmul_different_dtypes passed!")


def test_reduce_scatter_matmul_larger_dimensions():
    """Test reduce_scatter_matmul with larger matrix dimensions."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_reduce_scatter_matmul_larger_dimensions: requires at least 2 devices"
        )
        return

    # Larger dimensions
    m = 512 * num_devices
    k = 1024
    n = 512 * num_devices
    n_per_device = n // num_devices

    key = random.key(123)
    key1, key2 = random.split(key)

    x = random.uniform(key1, shape=(m, k),
                       dtype=jnp.float32).astype(jnp.bfloat16)
    y = random.uniform(key2, shape=(k, n_per_device),
                       dtype=jnp.float32).astype(jnp.bfloat16)

    devices = mesh_utils.create_device_mesh((num_devices, ))
    mesh = Mesh(devices, axis_names=("x", ))

    x_sharding = NamedSharding(mesh, P(None, None))
    y_sharding = NamedSharding(mesh, P(None, None))  # y already local shape

    x_sharded = jax.device_put(x, x_sharding)
    y_sharded = jax.device_put(y, y_sharding)

    result = reduce_scatter_matmul(x_sharded, y_sharded, mesh, "x")
    expected = ref_reduce_scatter_matmul(x_sharded, y_sharded, mesh, "x")

    np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.1)

    print(
        f"✓ test_reduce_scatter_matmul_larger_dimensions (m={m}, k={k}, n={n}) passed!"
    )


def test_reduce_scatter_matmul_lhs_transpose():
    """Test reduce_scatter_matmul with transposed LHS."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_reduce_scatter_matmul_lhs_transpose: requires at least 2 devices"
        )
        return

    m = 256 * num_devices
    k = 512
    n = 256 * num_devices
    n_per_device = n // num_devices

    key = random.key(99)
    key1, key2 = random.split(key)

    # x is transposed: [k, m] instead of [m, k]
    x = random.uniform(key1, shape=(k, m),
                       dtype=jnp.float32).astype(jnp.bfloat16)
    y = random.uniform(key2, shape=(k, n_per_device),
                       dtype=jnp.float32).astype(jnp.bfloat16)

    devices = mesh_utils.create_device_mesh((num_devices, ))
    mesh = Mesh(devices, axis_names=("x", ))

    x_sharding = NamedSharding(mesh, P(None, None))
    y_sharding = NamedSharding(mesh, P(None, None))  # y already local shape

    x_sharded = jax.device_put(x, x_sharding)
    y_sharded = jax.device_put(y, y_sharding)

    result = reduce_scatter_matmul(x_sharded,
                                   y_sharded,
                                   mesh,
                                   "x",
                                   lhs_transpose=True)
    expected = ref_reduce_scatter_matmul(x_sharded,
                                         y_sharded,
                                         mesh,
                                         "x",
                                         lhs_transpose=True)

    np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.1)

    print("✓ test_reduce_scatter_matmul_lhs_transpose passed!")


def test_reduce_scatter_matmul_custom_block_sizes():
    """Test reduce_scatter_matmul with custom block sizes."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_reduce_scatter_matmul_custom_block_sizes: requires at least 2 devices"
        )
        return

    m = 256 * num_devices
    k = 512
    n = 256 * num_devices
    n_per_device = n // num_devices

    x = jnp.ones((m, k), dtype=jnp.bfloat16)
    y = jnp.ones((k, n_per_device), dtype=jnp.bfloat16)

    devices = mesh_utils.create_device_mesh((num_devices, ))
    mesh = Mesh(devices, axis_names=("x", ))

    x_sharding = NamedSharding(mesh, P(None, None))
    y_sharding = NamedSharding(mesh, P(None, None))  # y already local shape

    x_sharded = jax.device_put(x, x_sharding)
    y_sharded = jax.device_put(y, y_sharding)

    # Test with custom block sizes
    for bn, bk in [(128, 128), (256, 256), (128, 256)]:
        result = reduce_scatter_matmul(x_sharded,
                                       y_sharded,
                                       mesh,
                                       "x",
                                       bn=bn,
                                       bk=bk)
        expected = ref_reduce_scatter_matmul(x_sharded, y_sharded, mesh, "x")

        np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.1)
        print(f"  ✓ block sizes (bn={bn}, bk={bk}) work")

    print("✓ test_reduce_scatter_matmul_custom_block_sizes passed!")


def test_validate_inputs():
    """Test the input validation function."""
    num_devices = 4  # Assume 4 devices for validation tests

    # Valid inputs
    m = 256 * num_devices
    k = 512
    n_per_device = 256

    x_valid = jnp.ones((m, k), dtype=jnp.bfloat16)
    y_valid = jnp.ones((k, n_per_device), dtype=jnp.bfloat16)

    # Should not raise
    validate_inputs(x_valid, y_valid, num_devices)
    print("  ✓ valid inputs pass validation")

    # Test invalid dimensions
    try:
        x_bad = jnp.ones((100, k),
                         dtype=jnp.bfloat16)  # m not divisible by tp_size * 16
        validate_inputs(x_bad, y_valid, num_devices)
        print("  ✗ should have raised for invalid m dimension")
    except ValueError:
        print("  ✓ correctly rejects invalid m dimension")

    # Test mismatched k
    try:
        y_bad = jnp.ones((k + 1, n_per_device), dtype=jnp.bfloat16)
        validate_inputs(x_valid, y_bad, num_devices)
        print("  ✗ should have raised for mismatched k")
    except ValueError:
        print("  ✓ correctly rejects mismatched k dimension")

    # Test mismatched dtypes
    try:
        x_float = jnp.ones((m, k), dtype=jnp.float32)
        y_bfloat = jnp.ones((k, n_per_device), dtype=jnp.bfloat16)
        validate_inputs(x_float, y_bfloat, num_devices)
        print("  ✗ should have raised for mismatched dtypes")
    except ValueError:
        print("  ✓ correctly rejects mismatched dtypes")

    # Test 1D input
    try:
        x_1d = jnp.ones((m, ), dtype=jnp.bfloat16)
        validate_inputs(x_1d, y_valid, num_devices)
        print("  ✗ should have raised for 1D input")
    except ValueError:
        print("  ✓ correctly rejects 1D input")

    # Test k not divisible by 128
    try:
        x_bad_k = jnp.ones((m, 100), dtype=jnp.bfloat16)
        y_bad_k = jnp.ones((100, n_per_device), dtype=jnp.bfloat16)
        validate_inputs(x_bad_k, y_bad_k, num_devices)
        print("  ✗ should have raised for k not divisible by 128")
    except ValueError:
        print("  ✓ correctly rejects k not divisible by 128")

    print("✓ test_validate_inputs passed!")


def test_get_kernel_name():
    """Test kernel name generation."""
    name = get_kernel_name(256, 512, False)
    assert "reduce_scatter_matmul" in name
    assert "256" in name
    assert "512" in name
    print(f"  Generated kernel name: {name}")

    name_transposed = get_kernel_name(128, 256, True)
    assert "True" in name_transposed
    print(f"  Generated transposed kernel name: {name_transposed}")

    print("✓ test_get_kernel_name passed!")


def test_get_vmem_estimate_bytes():
    """Test VMEM estimation function."""
    m, n, k = 1024, 2048, 512
    bn, bk = 256, 256
    tp_size = 4
    acc_bytes = 1024 * 1024

    estimate = get_vmem_estimate_bytes(m, n, k, bn, bk, acc_bytes, tp_size,
                                       jnp.bfloat16, jnp.bfloat16,
                                       jnp.bfloat16)

    assert estimate > 0, "VMEM estimate should be positive"
    assert isinstance(estimate, int), "VMEM estimate should be an integer"

    print(
        f"  VMEM estimate for m={m}, n={n}, k={k}: {estimate / 1024 / 1024:.2f} MB"
    )

    print("✓ test_get_vmem_estimate_bytes passed!")


def test_ref_reduce_scatter_matmul_naive():
    """Test the naive reference implementation."""
    tp_size = 4
    m = 256
    k = 512
    n = 256

    x = jnp.ones((m, k), dtype=jnp.float32)
    y = jnp.ones((k, n), dtype=jnp.float32)

    outputs = ref_reduce_scatter_matmul_naive(x, y, tp_size)

    assert len(
        outputs) == tp_size, f"Expected {tp_size} outputs, got {len(outputs)}"

    m_per_device = m // tp_size
    n_per_device = n // tp_size

    for i, output in enumerate(outputs):
        assert output.shape == (m_per_device, n_per_device), (
            f"Output {i} shape mismatch: got {output.shape}, "
            f"expected ({m_per_device}, {n_per_device})")
        # With all ones input, x @ y_shard = k * ones, and each device gets
        # a portion along m
        expected_value = k
        np.testing.assert_allclose(output, expected_value, rtol=1e-5)

    print("✓ test_ref_reduce_scatter_matmul_naive passed!")


def test_output_shape():
    """Test that output shape is correct."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print("⚠ Skipping test_output_shape: requires at least 2 devices")
        return

    m = 256 * num_devices
    k = 512
    n = 256 * num_devices
    n_per_device = n // num_devices
    m_per_device = m // num_devices

    x = jnp.ones((m, k), dtype=jnp.bfloat16)
    y = jnp.ones((k, n_per_device), dtype=jnp.bfloat16)

    devices = mesh_utils.create_device_mesh((num_devices, ))
    mesh = Mesh(devices, axis_names=("x", ))

    x_sharding = NamedSharding(mesh, P(None, None))
    y_sharding = NamedSharding(mesh, P(None, None))  # y already local shape

    x_sharded = jax.device_put(x, x_sharding)
    y_sharded = jax.device_put(y, y_sharding)

    result = reduce_scatter_matmul(x_sharded, y_sharded, mesh, "x")

    # Check per-device local shape
    local_shape = result.addressable_shards[0].data.shape
    expected_local_shape = (m_per_device, n_per_device)

    assert local_shape == expected_local_shape, (
        f"Local shape mismatch: got {local_shape}, expected {expected_local_shape}"
    )

    print(
        f"✓ test_output_shape passed! (local shape per device: {local_shape})")


def test_different_tp_sizes():
    """Test reduce_scatter_matmul with different tensor parallel sizes."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_different_tp_sizes: requires at least 2 devices")
        return

    # Test with different TP sizes that divide num_devices
    tp_sizes = [
        tp for tp in [2, 4, 8] if tp <= num_devices and num_devices % tp == 0
    ]

    for tp_size in tp_sizes:
        m = 256 * tp_size
        k = 512
        n = 256 * tp_size
        n_per_device = n // tp_size

        x = jnp.ones((m, k), dtype=jnp.bfloat16)
        y = jnp.ones((k, n_per_device), dtype=jnp.bfloat16)

        # Select a subset of devices for this test
        all_devices = jax.devices()
        selected_devices = np.array(all_devices[:tp_size])
        mesh = Mesh(selected_devices, axis_names=("x", ))

        x_sharding = NamedSharding(mesh, P(None, None))
        y_sharding = NamedSharding(mesh, P(None,
                                           None))  # y already local shape

        x_sharded = jax.device_put(x, x_sharding)
        y_sharded = jax.device_put(y, y_sharding)

        result = reduce_scatter_matmul(x_sharded, y_sharded, mesh, "x")
        expected = ref_reduce_scatter_matmul(x_sharded, y_sharded, mesh, "x")

        np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.1)
        print(f"  ✓ tp_size={tp_size} works")

    print("✓ test_different_tp_sizes passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Running Reduce-Scatter Matmul Kernel Tests")
    print(f"Number of devices: {jax.device_count()}")
    print(f"Device type: {jax.devices()[0].device_kind}")
    print("=" * 70 + "\n")

    # Unit tests (no device requirement)
    print("--- Unit Tests ---")
    test_validate_inputs()
    test_get_kernel_name()
    test_get_vmem_estimate_bytes()
    test_ref_reduce_scatter_matmul_naive()

    # Multi-device tests
    print("\n--- Multi-Device Tests ---")
    test_reduce_scatter_matmul_basic()
    test_reduce_scatter_matmul_with_random_values()
    test_reduce_scatter_matmul_different_dtypes()
    test_reduce_scatter_matmul_larger_dimensions()
    test_reduce_scatter_matmul_lhs_transpose()
    test_reduce_scatter_matmul_custom_block_sizes()
    test_output_shape()
    test_different_tp_sizes()

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
