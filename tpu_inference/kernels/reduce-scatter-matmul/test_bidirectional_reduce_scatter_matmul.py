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

Output:
    - Pass/fail status for each test
"""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from experimental.reduce_scatter_matmul.bidirection_reduce_scatter_matmul import \
    bidirectional_reduce_scatter_matmul  # noqa: E402
from jax import lax, random  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.experimental.shard_map import shard_map  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402


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
    print(
        f"    Result[0,0]: {float(result_np[0,0]):.4f}, Expected[0,0]: {float(expected_np[0,0]):.4f}"
    )

    if float(expected_np.mean()) > 1e-6:
        ratio = float(result_np.mean()) / float(expected_np.mean())
        print(f"    Mean ratio (result/expected): {ratio:.4f}")
        if 0.7 < ratio < 0.9:
            print(
                f"    ⚠ Result is ~{ratio*100:.0f}% of expected - may be missing accumulations"
            )


def test_bidirection_reduce_scatter_matmul_smoke():
    """Simple smoke test with small dimensions for quick verification."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_bidirection_reduce_scatter_matmul_smoke: requires at least 2 devices"
        )
        return

    # Small dimensions for fast testing
    # M must be divisible by num_devices and result in m_half_block divisible by bm
    m = 256 * num_devices  # M_block = 256, M_half_block = 128, divisible by bm=128
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

    result = bidirectional_reduce_scatter_matmul_sharded(x_sharded,
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
    print("✓ test_bidirection_reduce_scatter_matmul_smoke passed!")


def test_bidirection_reduce_scatter_matmul_basic():
    """Test that bidirectional_reduce_scatter_matmul correctly computes matmul + reduce-scatter."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_bidirection_reduce_scatter_matmul_basic: requires at least 2 devices"
        )
        return

    # Test dimensions
    m = 256 * num_devices
    k = 256 * num_devices  # K will be sharded
    n = 128

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

    assert (result.shape == expected.shape
            ), f"Shape mismatch: got {result.shape}, expected {expected.shape}"

    _print_array_comparison(result, expected, "Basic Test")

    np.testing.assert_allclose(result, expected, rtol=0.02)

    print("✓ test_bidirection_reduce_scatter_matmul_basic passed!")


def test_bidirection_reduce_scatter_matmul_with_random_values():
    """Test with random input values to verify numerical correctness."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_bidirection_reduce_scatter_matmul_with_random_values: requires at least 2 devices"
        )
        return

    m = 256 * num_devices
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

    result = bidirectional_reduce_scatter_matmul_sharded(x_sharded,
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

    _print_array_comparison(result, expected, "Random Values Test")

    np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.1)

    print(
        "✓ test_bidirection_reduce_scatter_matmul_with_random_values passed!")


def test_bidirection_reduce_scatter_matmul_different_dtypes():
    """Test with different data types."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_bidirection_reduce_scatter_matmul_different_dtypes: requires at least 2 devices"
        )
        return

    devices = mesh_utils.create_device_mesh((num_devices, ))
    mesh = Mesh(devices, axis_names=("x", ))

    for dtype in [jnp.float32, jnp.bfloat16]:
        m = 256 * num_devices
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

        result = bidirectional_reduce_scatter_matmul_sharded(x_sharded,
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

        assert result.dtype == dtype, f"Expected dtype {dtype}, got {result.dtype}"

        rtol = 0.05 if dtype == jnp.bfloat16 else 1e-4
        atol = 0.1 if dtype == jnp.bfloat16 else 1e-5
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)

        print(f"  ✓ dtype {dtype} works")

    print("✓ test_bidirection_reduce_scatter_matmul_different_dtypes passed!")


def test_bidirection_reduce_scatter_matmul_larger_dimensions():
    """Test with larger matrix dimensions."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_bidirection_reduce_scatter_matmul_larger_dimensions: requires at least 2 devices"
        )
        return

    m = 512 * num_devices
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

    result = bidirectional_reduce_scatter_matmul_sharded(x_sharded,
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
        f"✓ test_bidirection_reduce_scatter_matmul_larger_dimensions (m={m}, k={k}, n={n}) passed!"
    )


def test_output_shape():
    """Test that output shape is correct."""
    num_devices = jax.device_count()

    if num_devices < 2:
        print("⚠ Skipping test_output_shape: requires at least 2 devices")
        return

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
    device_counts = [
        dc for dc in [2, 4, 8] if dc < num_devices and num_devices % dc == 0
    ]

    for dc in device_counts:
        jax.clear_caches()

        m = 256 * dc
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

        result = bidirectional_reduce_scatter_matmul_sharded(x_sharded,
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
    """Verify that each device outputs the correct M_block.

    Device i should own M_block i (rows i*m_per_device to (i+1)*m_per_device).
    This tests both TOP and BOT halves are correctly assembled.
    """
    num_devices = jax.device_count()

    if num_devices < 2:
        print("⚠ Skipping test_slice_mapping: requires at least 2 devices")
        return

    m = 256 * num_devices
    k = 256 * num_devices
    n = 128
    m_per_device = m // num_devices

    # Create x where each m_block has a unique value pattern
    # Block i has value (i+1) in all elements
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

    result = bidirectional_reduce_scatter_matmul_sharded(x_sharded,
                                                         y_sharded,
                                                         mesh,
                                                         axis_name="x")

    print(f"  Verifying slice mapping (num_devices={num_devices}):")
    print("  Expected: device i outputs block i with value (i+1) * k")

    all_correct = True
    for device_idx, shard in enumerate(result.addressable_shards):
        # Check both TOP and BOT halves
        top_value = float(shard.data[0, 0])
        bot_value = float(shard.data[m_per_device // 2, 0])

        expected_value = (device_idx + 1) * k

        print(
            f"  Device {device_idx}: TOP={top_value:.0f}, BOT={bot_value:.0f}, "
            f"expected={expected_value}")

        if abs(top_value - expected_value) > expected_value * 0.05:
            all_correct = False
            print(f"    ⚠ TOP MISMATCH on device {device_idx}")
        if abs(bot_value - expected_value) > expected_value * 0.05:
            all_correct = False
            print(f"    ⚠ BOT MISMATCH on device {device_idx}")

    if all_correct:
        print("✓ test_slice_mapping passed!")
    else:
        raise AssertionError(
            "test_slice_mapping FAILED - slice mapping is wrong")


def test_top_bot_consistency():
    """Test that TOP and BOT halves are consistent within each device's output.

    Both halves should contain the same block's data after reduction.
    """
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_top_bot_consistency: requires at least 2 devices")
        return

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
        return reduce_scatter_matmul_reference(x_shard, y_shard, axis_name="x")

    reference_fn_sharded = shard_map(
        reference_fn,
        mesh=mesh,
        in_specs=(P(None, "x"), P(None, "x")),
        out_specs=P("x", None),
        check_rep=False,
    )

    expected = reference_fn_sharded(x_sharded, y_sharded)

    # Check each device's TOP and BOT halves match reference
    print(f"  Checking TOP/BOT consistency across {num_devices} devices:")

    for device_idx, (result_shard, expected_shard) in enumerate(
            zip(result.addressable_shards, expected.addressable_shards)):
        result_data = np.asarray(result_shard.data).astype(np.float32)
        expected_data = np.asarray(expected_shard.data).astype(np.float32)

        top_result = result_data[:m_half, :]
        bot_result = result_data[m_half:, :]
        top_expected = expected_data[:m_half, :]
        bot_expected = expected_data[m_half:, :]

        top_diff = np.max(np.abs(top_result - top_expected))
        bot_diff = np.max(np.abs(bot_result - bot_expected))

        print(
            f"  Device {device_idx}: TOP max_diff={top_diff:.6f}, BOT max_diff={bot_diff:.6f}"
        )

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

    print("✓ test_top_bot_consistency passed!")


def test_bidirectional_vs_single_direction():
    """Compare bidirectional implementation with single-direction reference.

    Both should produce the same result, but bidirectional should use 2x bandwidth.
    """
    num_devices = jax.device_count()

    if num_devices < 2:
        print(
            "⚠ Skipping test_bidirectional_vs_single_direction: requires at least 2 devices"
        )
        return

    m = 256 * num_devices
    k = 256 * num_devices
    n = 128

    key = random.key(777)
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

    # Bidirectional result
    result_bidir = bidirectional_reduce_scatter_matmul_sharded(x_sharded,
                                                               y_sharded,
                                                               mesh,
                                                               axis_name="x")

    # Reference (single direction conceptually, using psum_scatter)
    def reference_fn(x_shard, y_shard):
        return reduce_scatter_matmul_reference(x_shard, y_shard, axis_name="x")

    reference_fn_sharded = shard_map(
        reference_fn,
        mesh=mesh,
        in_specs=(P(None, "x"), P(None, "x")),
        out_specs=P("x", None),
        check_rep=False,
    )

    result_ref = reference_fn_sharded(x_sharded, y_sharded)

    _print_array_comparison(result_bidir, result_ref,
                            "Bidirectional vs Reference")

    np.testing.assert_allclose(result_bidir, result_ref, rtol=0.05, atol=0.1)

    print("✓ test_bidirectional_vs_single_direction passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Running Bidirectional Reduce-Scatter Matmul Tests")
    print(f"Number of devices: {jax.device_count()}")
    print(f"Device type: {jax.devices()[0].device_kind}")
    print("=" * 70 + "\n")

    # Multi-device tests
    print("--- Multi-Device Tests ---")
    test_bidirection_reduce_scatter_matmul_smoke()
    test_bidirection_reduce_scatter_matmul_basic()
    test_bidirection_reduce_scatter_matmul_with_random_values()
    test_bidirection_reduce_scatter_matmul_different_dtypes()
    test_bidirection_reduce_scatter_matmul_larger_dimensions()
    test_output_shape()
    test_different_device_counts()
    test_slice_mapping()
    test_top_bot_consistency()
    test_bidirectional_vs_single_direction()

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
