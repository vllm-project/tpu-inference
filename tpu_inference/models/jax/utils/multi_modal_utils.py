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

from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
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
    [`vllm.model_executor.models.SupportsMultiModal.embed_multimodal`][].
    """
    assert isinstance(mm_embeddings, (list, tuple, jax.Array)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method.")

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method.")

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method.")


def flatten_embeddings(embeddings: NestedTensors) -> jax.Array:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, jax.Array):
        return embeddings.reshape(-1, embeddings.shape[-1])

    return jnp.concatenate([flatten_embeddings(t) for t in embeddings], axis=0)


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, jax.Array):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(
        _embedding_count_expression(inner) for inner in embeddings)


def normalize_mm_grid_thw(
    grid_thw: object,
) -> tuple[tuple[int, int, int], ...]:
    """Normalize grid_thw into a tuple-of-tuples.

    Accepts (3,), (N, 3), or (B, N, 3) style list/tuple/numpy/torch/jax inputs.
    """
    if grid_thw is None:
        return ()

    if isinstance(grid_thw, (list, tuple)):
        if len(grid_thw) == 0:
            return ()
        if len(grid_thw) == 3 and all(
                isinstance(v, (int, np.integer)) for v in grid_thw):
            return (tuple(int(v) for v in grid_thw), )
        if all(isinstance(row, (list, tuple)) for row in grid_thw):
            if grid_thw and grid_thw[0] and isinstance(grid_thw[0][0],
                                                      (list, tuple)):
                flat_rows = [row for batch in grid_thw for row in batch]
                return tuple(tuple(int(v) for v in row) for row in flat_rows)
            return tuple(tuple(int(v) for v in row) for row in grid_thw)

    if hasattr(grid_thw, "detach"):
        grid_thw = grid_thw.detach().cpu()
    if hasattr(grid_thw, "numpy"):
        try:
            grid_thw = grid_thw.numpy()
        except Exception:
            pass

    arr = np.asarray(grid_thw)
    if arr.size == 0:
        return ()
    if arr.ndim == 1 and arr.shape[0] == 3:
        return (tuple(int(v) for v in arr.tolist()), )
    if arr.ndim == 2 and arr.shape[1] == 3:
        return tuple(tuple(int(v) for v in row) for row in arr.tolist())
    if arr.ndim == 3 and arr.shape[2] == 3:
        flat = arr.reshape(-1, 3)
        return tuple(tuple(int(v) for v in row) for row in flat.tolist())

    raise ValueError(
        "Incorrect type/shape of grid_thw. Expected (3,), (N, 3), or (B, N, 3)."
    )


def reshape_mm_tensor(mm_input: object, name: str) -> jax.Array:
    """Normalize multimodal tensor input to a 2D JAX array."""
    if isinstance(mm_input, list):
        arrays_to_concat = [jnp.asarray(item) for item in mm_input]
        return jnp.concatenate(arrays_to_concat, axis=0)

    if hasattr(mm_input, "detach"):
        mm_input = mm_input.detach().cpu()
    if hasattr(mm_input, "numpy"):
        try:
            mm_input = mm_input.numpy()
        except Exception:
            pass

    if hasattr(mm_input, 'ndim'):
        array_input = jnp.asarray(mm_input)
        if array_input.ndim == 2:
            return array_input
        if array_input.ndim == 3:
            return array_input.reshape(-1, array_input.shape[-1])

    raise ValueError(f"Incorrect type of {name}. "
                     f"Got type: {type(mm_input)}")


def split_mm_embeddings_by_grid(
    embeddings: jax.Array,
    grid_thw: tuple[tuple[int, int, int], ...],
    spatial_merge_size: int,
    deepstack_embeddings: Optional[list[jax.Array]] = None,
) -> tuple[tuple[jax.Array, ...], Optional[list[list[jax.Array]]]]:
    """Split concatenated multimodal embeddings back into per-item chunks."""
    sizes = np.array([
        t * (h // spatial_merge_size) * (w // spatial_merge_size)
        for t, h, w in grid_thw
    ],
                     dtype=np.int64)

    if sizes.size == 0:
        return (), None
    if sizes.size == 1:
        item_splits = (embeddings, )
        if not deepstack_embeddings:
            return item_splits, None
        return item_splits, [[layer_embeds for layer_embeds in deepstack_embeddings]]

    split_indices = np.cumsum(sizes)[:-1]
    item_splits = tuple(jnp.split(embeddings, split_indices))

    if not deepstack_embeddings:
        return item_splits, None

    layer_splits = [
        tuple(jnp.split(layer_embeds, split_indices))
        for layer_embeds in deepstack_embeddings
    ]
    deepstack_by_item = []
    for item_idx in range(len(item_splits)):
        deepstack_by_item.append(
            [layer_split[item_idx] for layer_split in layer_splits])
    return item_splits, deepstack_by_item


def _merge_multimodal_embeddings(
    inputs_embeds: jax.Array,
    is_multimodal: jax.Array,
    multimodal_embeddings: jax.Array,
) -> jax.Array:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.
        This returns a new array with the updated values.
    Note:
        This returns a new array with the updated values.
    """
    # The check for matching number of tokens is removed as it is not
    # JIT-compatible. If the shapes mismatch, JAX will raise an error
    # during execution anyway. The user-friendly error message is
    # sacrificed for JIT compatibility.

    # JIT-compatible implementation using jnp.where to avoid
    # NonConcreteBooleanIndexError.
    # Create a dummy row to handle indices for non-multimodal tokens.
    # The content of the dummy row does not matter as it will be masked out.
    dummy_row = jnp.zeros_like(multimodal_embeddings[0:1])

    # Prepend the dummy row to the flattened embeddings.
    flattened_padded = jnp.concatenate([dummy_row, multimodal_embeddings],
                                       axis=0)

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
    multimodal_embeddings: jax.Array,
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
