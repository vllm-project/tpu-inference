from typing import Union

import jax
import jax.numpy as jnp
from typing_extensions import TypeAlias
from vllm.logger import init_logger

logger = init_logger(__name__)

NestedTensors: TypeAlias = Union[list["NestedTensors"], list["jax.Array"],
                                 "jax.Array", tuple["jax.Array", ...]]
"""
Uses a list instead of a tensor if the dimensions of each element do not match.
"""

MultiModalEmbeddings = Union[list[jax.Array], jax.Array, tuple[jax.Array, ...]]
"""
The output embeddings must be one of the following formats:

- A list or tuple of 2D tensors, where each tensor corresponds to
    each input multimodal data item (e.g, image).
- A single 3D tensor, with the batch dimension grouping the 2D tensors.
"""


def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    """
    Perform sanity checks for the result of
    [`vllm.model_executor.models.SupportsMultiModal.get_multimodal_embeddings`][].
    """
    assert isinstance(mm_embeddings, (list, tuple, jax.Array)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")


def _flatten_embeddings(embeddings: NestedTensors) -> jax.Array:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, jax.Array):
        return embeddings.reshape(-1, embeddings.shape[-1])

    return jnp.concatenate([_flatten_embeddings(t) for t in embeddings],
                           axis=0)


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, jax.Array):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(
        _embedding_count_expression(inner) for inner in embeddings)


def _merge_multimodal_embeddings(
    inputs_embeds: jax.Array,
    is_multimodal: jax.Array,
    multimodal_embeddings: NestedTensors,
) -> jax.Array:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.
        This returns a new array with the updated values.
    Note:
        This returns a new array with the updated values.
    """
    flattened = _flatten_embeddings(multimodal_embeddings)
    # The check for matching number of tokens is removed as it is not
    # JIT-compatible. If the shapes mismatch, JAX will raise an error
    # during execution anyway. The user-friendly error message is
    # sacrificed for JIT compatibility.

    # JIT-compatible implementation using jnp.where to avoid
    # NonConcreteBooleanIndexError.
    # Create a dummy row to handle indices for non-multimodal tokens.
    # The content of the dummy row does not matter as it will be masked out.
    dummy_row = jnp.zeros_like(flattened[0:1])

    # Prepend the dummy row to the flattened embeddings.
    flattened_padded = jnp.concatenate([dummy_row, flattened], axis=0)

    # Create gather indices. For each token in the input sequence, this gives
    # the index into `flattened_padded`.
    # For non-multimodal tokens, the index will be 0 (pointing to the dummy
    # row). For the k-th multimodal token, the index will be k.
    gather_indices = jnp.cumsum(is_multimodal)

    # Gather the embeddings to be placed.
    update_values = flattened_padded[gather_indices]

    # Use jnp.where to select between original and new embeddings.
    condition = jnp.expand_dims(is_multimodal, axis=-1)
    return jnp.where(condition, update_values, inputs_embeds)


def merge_multimodal_embeddings(
    input_ids: jax.Array,
    inputs_embeds: jax.Array,
    multimodal_embeddings: NestedTensors,
    placeholder_token_id: Union[int, list[int]],
) -> jax.Array:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    ``placeholder_token_id`` can be a list of token ids (e.g, token ids
    of img_start, img_break, and img_end tokens) when needed: This means
    the order of these tokens in the ``input_ids`` MUST MATCH the order of
    their embeddings in ``multimodal_embeddings`` since we need to
    slice-merge instead of individually scattering.

    For example, if input_ids is "TTTTTSIIIBIIIBIIIETTT", where
    - T is text token
    - S is image start token
    - I is image embedding token
    - B is image break token
    - E is image end token.

    Then the image embeddings (that correspond to I's) from vision encoder
    must be padded with embeddings of S, B, and E in the same order of
    input_ids for a correct embedding merge.

        This returns a new array with the updated values.
    """
    if isinstance(placeholder_token_id, list):
        placeholder_token_id = jnp.array(placeholder_token_id)

        return _merge_multimodal_embeddings(
            inputs_embeds,
            jnp.isin(input_ids, placeholder_token_id),
            multimodal_embeddings,
        )

    return _merge_multimodal_embeddings(
        inputs_embeds,
        (input_ids == placeholder_token_id),
        multimodal_embeddings,
    )
