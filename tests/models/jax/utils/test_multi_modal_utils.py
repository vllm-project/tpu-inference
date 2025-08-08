# test_multi_modal_utils.py
import jax.numpy as jnp
import numpy as np
import pytest
from multi_modal_utils import (MultiModalEmbeddings, NestedTensors,
                               _count_flattened_embeddings,
                               _embedding_count_expression,
                               _flatten_embeddings,
                               merge_multimodal_embeddings,
                               sanity_check_mm_encoder_outputs)

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


# --- Tests for _flatten_embeddings ---


def test_flatten_single_array():
    """Tests _flatten_embeddings with a single 2D array."""
    emb: NestedTensors = jnp.arange(12).reshape((3, 4))
    result = _flatten_embeddings(emb)
    expected = jnp.arange(12).reshape((3, 4))
    np.testing.assert_array_equal(result, expected)


def test_flatten_single_3d_array():
    """Tests _flatten_embeddings with a single 3D array."""
    emb: NestedTensors = jnp.arange(24).reshape((2, 3, 4))
    result = _flatten_embeddings(emb)
    expected = jnp.arange(24).reshape((6, 4))
    np.testing.assert_array_equal(result, expected)


def test_flatten_list_of_arrays():
    """Tests _flatten_embeddings with a list of 2D arrays."""
    emb: NestedTensors = [
        jnp.arange(12).reshape((3, 4)),
        jnp.arange(12, 20).reshape((2, 4))
    ]
    result = _flatten_embeddings(emb)
    expected = jnp.arange(20).reshape((5, 4))
    np.testing.assert_array_equal(result, expected)


def test_flatten_nested_list():
    """Tests _flatten_embeddings with a nested list of arrays."""
    emb: NestedTensors = [
        jnp.arange(6).reshape((2, 3)),
        [
            jnp.arange(6, 12).reshape((2, 3)),
            jnp.arange(12, 15).reshape((1, 3))
        ]
    ]
    result = _flatten_embeddings(emb)
    expected = jnp.arange(15).reshape((5, 3))
    np.testing.assert_array_equal(result, expected)


def test_flatten_varying_leading_dims():
    """Tests _flatten_embeddings with arrays having different leading dimensions before flattening."""
    emb: NestedTensors = [jnp.ones((1, 2, 5)), jnp.ones((3, 5))]
    result = _flatten_embeddings(emb)
    expected = jnp.concatenate([jnp.ones((2, 5)), jnp.ones((3, 5))], axis=0)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (5, 5)


# --- Tests for _embedding_count_expression ---


def test_count_expression_single():
    emb: NestedTensors = jnp.zeros((3, 4, 5))
    assert _embedding_count_expression(emb) == "3 x 4"


def test_count_expression_list():
    emb: NestedTensors = [jnp.zeros((2, 5)), jnp.zeros((3, 5))]
    assert _embedding_count_expression(emb) == "2 + 3"


def test_count_expression_nested():
    emb: NestedTensors = [
        jnp.zeros((1, 2, 5)), [jnp.zeros((3, 5)),
                               jnp.zeros((1, 5))]
    ]
    assert _embedding_count_expression(emb) == "1 x 2 + 3 + 1"


# --- Tests for _count_flattened_embeddings ---
def test_count_flattened_empty():
    assert _count_flattened_embeddings([]) == 0
    assert _count_flattened_embeddings(jnp.empty((0, 4))) == 0


def test_count_flattened_single():
    assert _count_flattened_embeddings(jnp.ones((3, 4))) == 3


def test_count_flattened_3d():
    assert _count_flattened_embeddings(jnp.ones((2, 3, 4))) == 6


def test_count_flattened_list():
    assert _count_flattened_embeddings([jnp.ones((2, 4)),
                                        jnp.ones((3, 4))]) == 5


def test_count_flattened_nested():
    assert _count_flattened_embeddings([jnp.ones((1, 4)), [jnp.ones(
        (2, 4))]]) == 3


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


def test_merge_list_placeholders(base_embeds):
    """Tests merging with a list of placeholder IDs."""
    input_ids = jnp.array([1, 2, -1, -2, -2, 3, -1, 4])
    inputs_embeds = base_embeds[:len(input_ids)]
    mm_embeds: NestedTensors = jnp.arange(4 * EMBED_DIM).reshape(
        (4, EMBED_DIM))

    result = merge_multimodal_embeddings(input_ids,
                                         inputs_embeds,
                                         mm_embeds,
                                         placeholder_token_id=[-1, -2])

    expected = np.array(inputs_embeds)
    is_mm = np.isin(input_ids, [-1, -2])
    expected[is_mm] = mm_embeds
    np.testing.assert_array_equal(result, expected)


def test_merge_nested_mm_embeds(base_embeds):
    """Tests merging with nested multimodal embeddings."""
    input_ids = jnp.array([-1, -1, -1, -1, 1])
    inputs_embeds = base_embeds[:len(input_ids)]
    mm_embeds: NestedTensors = [
        jnp.ones((1, EMBED_DIM)), [jnp.ones((2, EMBED_DIM)) * 2],
        jnp.ones((1, EMBED_DIM)) * 3
    ]

    result = merge_multimodal_embeddings(input_ids,
                                         inputs_embeds,
                                         mm_embeds,
                                         placeholder_token_id=-1)

    expected = np.array([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [0, 0, 0, 0],
    ])
    np.testing.assert_array_equal(result, expected)


def test_merge_no_placeholders():
    """Tests merging when no placeholder tokens are in input_ids."""
    input_ids = jnp.array([1, 2, 3, 4])
    inputs_embeds = jnp.arange(len(input_ids) * EMBED_DIM).reshape(
        (len(input_ids), EMBED_DIM))
    mm_embeds: NestedTensors = jnp.empty((0, EMBED_DIM))

    result = merge_multimodal_embeddings(input_ids,
                                         inputs_embeds,
                                         mm_embeds,
                                         placeholder_token_id=-1)
    np.testing.assert_array_equal(result, inputs_embeds)


def test_merge_no_placeholders_with_embeddings_raises():
    """Tests merging raises error if embeddings are provided without placeholders."""
    input_ids = jnp.array([1, 2, 3, 4])
    inputs_embeds = jnp.arange(len(input_ids) * EMBED_DIM).reshape(
        (len(input_ids), EMBED_DIM))
    mm_embeds: NestedTensors = jnp.ones((1, EMBED_DIM))
    with pytest.raises(
            ValueError,
            match=
            "Input has no placeholder tokens, but 1 multimodal embeddings were provided"
    ):
        merge_multimodal_embeddings(input_ids,
                                    inputs_embeds,
                                    mm_embeds,
                                    placeholder_token_id=-1)


@pytest.mark.parametrize("placeholder_id", [-1, [-1, -2]])
def test_merge_mm_embeds_count_too_few_raises(placeholder_id, base_embeds):
    """Tests that a ValueError is raised if mm_embeds are too few."""
    input_ids = jnp.array([1, 2, -1, -1, 3])  # 2 placeholders
    inputs_embeds = base_embeds[:len(input_ids)]
    mm_embeds_too_few: NestedTensors = jnp.ones((1, EMBED_DIM))

    with pytest.raises(
            ValueError,
            match=r"Number of multimodal embeddings \(1\) does not match "
            r"the number of placeholder tokens \(2\)"):
        merge_multimodal_embeddings(input_ids,
                                    inputs_embeds,
                                    mm_embeds_too_few,
                                    placeholder_token_id=placeholder_id)


@pytest.mark.parametrize("placeholder_id", [-1, [-1, -2]])
def test_merge_mm_embeds_count_too_many_raises(placeholder_id, base_embeds):
    """Tests that a ValueError is raised if mm_embeds are too many."""
    input_ids = jnp.array([1, 2, -1, -1, 3])  # 2 placeholders
    inputs_embeds = base_embeds[:len(input_ids)]
    mm_embeds_too_many: NestedTensors = jnp.ones((3, EMBED_DIM))

    with pytest.raises(
            ValueError,
            match=r"Number of multimodal embeddings \(3\) does not match "
            r"the number of placeholder tokens \(2\)"):
        merge_multimodal_embeddings(input_ids,
                                    inputs_embeds,
                                    mm_embeds_too_many,
                                    placeholder_token_id=placeholder_id)
