from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_commons.models.jax.sampling_metadata import (
    DEFAULT_SAMPLING_PARAMS, TPUSupportedSamplingMetadata)

## Mocks and Fixtures


@dataclass
class MockInputBatch:
    """A mock of the InputBatch class, using NumPy arrays for CPU tensors."""

    all_greedy: bool
    num_reqs: int = 0
    temperature_cpu: np.ndarray = None
    top_k_cpu: np.ndarray = None
    top_p_cpu: np.ndarray = None


@pytest.fixture(scope="module")
def mesh() -> Mesh:
    """Creates a 1D JAX mesh for testing on available devices."""
    if not jax.devices():
        pytest.skip("No JAX devices available for testing.")
    return Mesh(np.array(jax.devices()), axis_names=("data", ))


## Test Cases


def test_from_input_batch_all_greedy(mesh: Mesh):
    """
    Tests TPUSupportedSamplingMetadata.from_input_batch when **all_greedy is True**.

    It should return an object with `do_sampling=False` and `None` for the tensors.
    """
    mock_batch = MockInputBatch(all_greedy=True)
    padded_num_reqs = 4

    metadata = TPUSupportedSamplingMetadata.from_input_batch(
        mesh=mesh, input_batch=mock_batch, padded_num_reqs=padded_num_reqs)

    assert not metadata.do_sampling, "do_sampling should be False for greedy requests"
    assert metadata.temperature is None
    assert metadata.top_k is None
    assert metadata.top_p is None


def test_from_input_batch_with_sampling_and_padding(mesh: Mesh):
    """
    Tests TPUSupportedSamplingMetadata.from_input_batch with sampling enabled,
    requiring the tensors to be **padded** to the correct shape.
    """
    num_reqs = 2
    padded_num_reqs = 4

    # Input tensors must be large enough to hold the padded values.
    temp_tensor = np.array([0.7, 0.8, 0.0, 0.0], dtype=np.float32)
    top_k_tensor = np.array([10, 20, 0, 0], dtype=np.int32)
    top_p_tensor = np.array([0.9, 0.95, 0.0, 0.0], dtype=np.float32)

    mock_batch = MockInputBatch(
        all_greedy=False,
        num_reqs=num_reqs,
        temperature_cpu=temp_tensor,
        top_k_cpu=top_k_tensor,
        top_p_cpu=top_p_tensor,
    )

    metadata = TPUSupportedSamplingMetadata.from_input_batch(
        mesh=mesh, input_batch=mock_batch, padded_num_reqs=padded_num_reqs)

    # 1. Check metadata flags and types
    assert metadata.do_sampling, "do_sampling should be True"
    assert isinstance(metadata.temperature, jnp.ndarray)
    assert isinstance(metadata.top_k, jnp.ndarray)
    assert isinstance(metadata.top_p, jnp.ndarray)

    # 2. Check shapes
    assert metadata.temperature.shape == (padded_num_reqs, )
    assert metadata.top_k.shape == (padded_num_reqs, )
    assert metadata.top_p.shape == (padded_num_reqs, )

    # 3. Check sharding (should be fully replicated)
    expected_sharding = NamedSharding(mesh, PartitionSpec(None))
    assert metadata.temperature.sharding == expected_sharding
    assert metadata.top_k.sharding == expected_sharding
    assert metadata.top_p.sharding == expected_sharding

    # 4. Check that values were correctly padded
    expected_temp = np.array(
        [
            0.7, 0.8, DEFAULT_SAMPLING_PARAMS["temperature"],
            DEFAULT_SAMPLING_PARAMS["temperature"]
        ],
        dtype=np.float32,
    )
    expected_top_k = np.array(
        [
            10, 20, DEFAULT_SAMPLING_PARAMS["top_k"],
            DEFAULT_SAMPLING_PARAMS["top_k"]
        ],
        dtype=np.int32,
    )
    expected_top_p = np.array(
        [
            0.9, 0.95, DEFAULT_SAMPLING_PARAMS["top_p"],
            DEFAULT_SAMPLING_PARAMS["top_p"]
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(np.asarray(metadata.temperature), expected_temp)
    np.testing.assert_array_equal(np.asarray(metadata.top_k), expected_top_k)
    np.testing.assert_allclose(np.asarray(metadata.top_p), expected_top_p)


def test_from_input_batch_no_padding_needed(mesh: Mesh):
    """
    Tests the case where `num_reqs` equals `padded_num_reqs`, so **no padding** should occur.
    """
    num_reqs = 4
    padded_num_reqs = 4

    temp_tensor = np.array([0.7, 0.8, 0.6, 0.5], dtype=np.float32)
    top_k_tensor = np.array([10, 20, 30, 40], dtype=np.int32)
    top_p_tensor = np.array([0.9, 0.95, 0.85, 0.8], dtype=np.float32)

    mock_batch = MockInputBatch(
        all_greedy=False,
        num_reqs=num_reqs,
        temperature_cpu=temp_tensor,
        top_k_cpu=top_k_tensor,
        top_p_cpu=top_p_tensor,
    )

    metadata = TPUSupportedSamplingMetadata.from_input_batch(
        mesh=mesh, input_batch=mock_batch, padded_num_reqs=padded_num_reqs)

    assert metadata.do_sampling
    # Check that values are identical to the input, since no padding was needed
    np.testing.assert_allclose(np.asarray(metadata.temperature), temp_tensor)
    np.testing.assert_array_equal(np.asarray(metadata.top_k), top_k_tensor)
    np.testing.assert_allclose(np.asarray(metadata.top_p), top_p_tensor)


def test_jax_tree_util_registration():
    """
    Tests that the dataclass is correctly registered as a **JAX PyTree**,
    meaning `jax.tree_util` functions can operate on it as expected. 🌳
    """
    metadata = TPUSupportedSamplingMetadata(
        temperature=jnp.array([0.7]),
        top_k=jnp.array([10]),
        top_p=jnp.array([0.9]),
        do_sampling=True,
    )

    # Flatten the PyTree
    leaves, treedef = jax.tree_util.tree_flatten(metadata)

    # The leaves should be the "data_fields" specified in the decorator
    assert len(leaves) == 3
    np.testing.assert_array_equal(leaves[0], jnp.array([0.7]))
    np.testing.assert_array_equal(leaves[1], jnp.array([10]))
    np.testing.assert_array_equal(leaves[2], jnp.array([0.9]))

    # Reconstruct the PyTree from leaves
    new_metadata = jax.tree_util.tree_unflatten(treedef, leaves)

    # The reconstructed object should match the original
    assert new_metadata.do_sampling == metadata.do_sampling
    np.testing.assert_array_equal(new_metadata.temperature,
                                  metadata.temperature)
    np.testing.assert_array_equal(new_metadata.top_k, metadata.top_k)
    np.testing.assert_array_equal(new_metadata.top_p, metadata.top_p)
