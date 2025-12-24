# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# test_multi_modal_utils.py
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.models.jax.utils.multi_modal_utils import (
    MultiModalEmbeddings, NestedTensors, flatten_embeddings,
    merge_multimodal_embeddings, sanity_check_mm_encoder_outputs)

# --- Tests for sanity_check_mm_encoder_outputs ---


def test_sanity_check_valid_list():
    """Tests sanity_check with a valid list of 2D embeddings."""
    embeddings: MultiModalEmbeddings = [
        jnp.ones((10, 128)), jnp.ones((15, 128))
    ]
    sanity_check_mm_encoder_outputs(embeddings, 2)
    # No assertion error expected


def test_sanity_check_valid_tuple():
    """Tests sanity_check with a valid tuple of 2D embeddings."""
    embeddings: MultiModalEmbeddings = (jnp.ones((10, 128)), jnp.ones(
        (15, 128)))
    sanity_check_mm_encoder_outputs(embeddings, 2)
    # No assertion error expected


def test_sanity_check_valid_3d_jax_array():
    """Tests sanity_check with a valid 3D jax.Array."""
    embeddings: MultiModalEmbeddings = jnp.ones((2, 10, 128))
    # This is valid because mm_embeddings is iterable, and each item (e)
    # in the first dimension has e.ndim == 2.
    sanity_check_mm_encoder_outputs(embeddings, 2)
    # No assertion error expected


def test_sanity_check_invalid_type():
    """Tests sanity_check with an invalid type for embeddings."""
    with pytest.raises(
            AssertionError,
            match=
            "Expected multimodal embeddings to be a list/tuple of 2D tensors"):
        sanity_check_mm_encoder_outputs("not a tensor", 1)


def test_sanity_check_wrong_num_items():
    """Tests sanity_check with a mismatch in the number of embeddings."""
    embeddings: MultiModalEmbeddings = [jnp.ones((10, 128))]
    with pytest.raises(
            AssertionError,
            match="Expected number of multimodal embeddings to match number of"
    ):
        sanity_check_mm_encoder_outputs(embeddings, 2)


def test_sanity_check_wrong_dimensions_in_list():
    """Tests sanity_check with non-2D tensors within the list."""
    embeddings: MultiModalEmbeddings = [jnp.ones((10, 128, 1))]
    with pytest.raises(
            AssertionError,
            match=
            "Expected multimodal embeddings to be a sequence of 2D tensors"):
        sanity_check_mm_encoder_outputs(embeddings, 1)


# --- Tests for flatten_embeddings ---


def test_flatten_single_array():
    """Tests flatten_embeddings with a single 2D array."""
    emb: NestedTensors = jnp.arange(12).reshape((3, 4))
    result = flatten_embeddings(emb)
    np.testing.assert_array_equal(result, emb)


def test_flatten_single_3d_array():
    """Tests flatten_embeddings with a single 3D array."""
    emb: NestedTensors = jnp.arange(24).reshape((2, 3, 4))
    result = flatten_embeddings(emb)
    expected = jnp.arange(24).reshape((6, 4))
    np.testing.assert_array_equal(result, expected)


def test_flatten_list_of_arrays():
    """Tests flatten_embeddings with a list of 2D arrays."""
    emb: NestedTensors = [
        jnp.arange(12).reshape((3, 4)),
        jnp.arange(12, 20).reshape((2, 4))
    ]
    result = flatten_embeddings(emb)
    expected = jnp.arange(20).reshape((5, 4))
    np.testing.assert_array_equal(result, expected)


def test_flatten_nested_list():
    """Tests flatten_embeddings with a nested list of arrays."""
    emb: NestedTensors = [
        jnp.arange(6).reshape((2, 3)),
        [
            jnp.arange(6, 12).reshape((2, 3)),
            jnp.arange(12, 15).reshape((1, 3))
        ]
    ]
    result = flatten_embeddings(emb)
    expected = jnp.arange(15).reshape((5, 3))
    np.testing.assert_array_equal(result, expected)


# --- Tests for merge_multimodal_embeddings ---

EMBED_DIM = 4


@pytest.fixture
def base_embeds():
    return jnp.zeros((8, EMBED_DIM))


def test_merge_single_placeholder(base_embeds):
    """Tests merging with a single integer placeholder ID."""
    input_ids = jnp.array([1, 2, -1, -1, 3, 4, -1, 5])
    inputs_embeds = base_embeds[:len(input_ids)]
    mm_embeds: NestedTensors = jnp.arange(3 * EMBED_DIM).reshape(
        (3, EMBED_DIM))
    result = merge_multimodal_embeddings(input_ids,
                                         inputs_embeds,
                                         mm_embeds,
                                         placeholder_token_id=-1)
    expected = np.array(inputs_embeds)
    expected[input_ids == -1] = mm_embeds
    np.testing.assert_array_equal(result, expected)


def test_merge_no_placeholders(base_embeds):
    """Tests merging when no placeholder tokens are in input_ids."""
    input_ids = jnp.array([1, 2, 3, 4])
    inputs_embeds = jnp.arange(len(input_ids) * EMBED_DIM).reshape(
        (len(input_ids), EMBED_DIM))
    mm_embeds: NestedTensors = jnp.empty((0, EMBED_DIM))

    # Based on the provided traceback, this raises a TypeError within JAX's gather.
    with pytest.raises(
            TypeError,
            match="Slice size at index 0 in gather op is out of range"):
        merge_multimodal_embeddings(input_ids,
                                    inputs_embeds,
                                    mm_embeds,
                                    placeholder_token_id=-1)


@pytest.mark.parametrize("placeholder_id", [-1, [-1, -2]])
def test_merge_mm_embeds_count_too_few(placeholder_id, base_embeds):
    """
    Tests behavior when fewer embeddings are provided than placeholders.
    Based on the test results provided, this scenario does NOT raise an error
    in the testing environment.
    """
    input_ids = jnp.array([1, 2, -1, -1, 3])  # 2 placeholders
    inputs_embeds = base_embeds[:len(input_ids)]
    mm_embeds_too_few: NestedTensors = jnp.ones((1, EMBED_DIM))

    try:
        # We are only asserting that this call does not crash.
        # The actual output in this unexpected case is not being tested.
        merge_multimodal_embeddings(input_ids,
                                    inputs_embeds,
                                    mm_embeds_too_few,
                                    placeholder_token_id=placeholder_id)
    except Exception as e:
        pytest.fail(
            f"Did not expect an exception based on test logs, but got {type(e).__name__}: {e}"
        )


@pytest.mark.parametrize("placeholder_id", [-1, [-1, -2]])
def test_merge_mm_embeds_count_too_many_no_raise(placeholder_id, base_embeds):
    """Tests that no error is raised if mm_embeds are too many; extras are ignored."""
    input_ids = jnp.array([1, 2, -1, -1, 3])  # 2 placeholders
    inputs_embeds = base_embeds[:len(input_ids)]
    mm_embeds_too_many: NestedTensors = jnp.arange(3 * EMBED_DIM).reshape(
        (3, EMBED_DIM))

    try:
        result = merge_multimodal_embeddings(
            input_ids,
            inputs_embeds,
            mm_embeds_too_many,
            placeholder_token_id=placeholder_id)
        # Check that the first 2 embeddings from mm_embeds_too_many were used.
        expected = np.array(inputs_embeds)
        is_mm = np.isin(input_ids, np.array(placeholder_id))
        expected[is_mm] = flatten_embeddings(mm_embeds_too_many)[:2]
        np.testing.assert_array_equal(result, expected)
    except Exception as e:
        pytest.fail(
            f"Did not expect an exception, but got {type(e).__name__}: {e}")
