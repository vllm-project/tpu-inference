"""Smoke tests for the modality-agnostic helpers that handle video grids."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.models.jax.utils.multi_modal_utils import (
    merge_multimodal_embeddings,
    normalize_mm_grid_thw,
    split_mm_embeddings_by_grid,
)


HIDDEN = 8
SPATIAL_MERGE = 2
VIDEO_TOKEN_ID = 151656
IMAGE_TOKEN_ID = 151655
TEXT_TOKEN_ID = 1


def _video_token_count(t: int, h: int, w: int, merge: int = SPATIAL_MERGE) -> int:
    return t * (h // merge) * (w // merge)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ([4, 8, 8], ((4, 8, 8),)),
        ((4, 8, 8), ((4, 8, 8),)),
        ([[4, 8, 8]], ((4, 8, 8),)),
        ([[1, 8, 8], [4, 8, 8]], ((1, 8, 8), (4, 8, 8))),
        (np.array([[4, 8, 8]]), ((4, 8, 8),)),
        (np.array([[[4, 8, 8]]]), ((4, 8, 8),)),
        (None, ()),
        ([], ()),
    ],
)
def test_normalize_grid_thw_accepts_video_shapes(raw, expected):
    assert normalize_mm_grid_thw(raw) == expected


def test_split_embeddings_round_trip_single_video():
    grid = ((4, 8, 8),)
    n_tokens = _video_token_count(*grid[0])
    embeds = jnp.arange(n_tokens * HIDDEN, dtype=jnp.float32).reshape(n_tokens, HIDDEN)

    splits, deepstack = split_mm_embeddings_by_grid(embeds, grid, SPATIAL_MERGE)

    assert deepstack is None
    assert len(splits) == 1
    assert splits[0].shape == (n_tokens, HIDDEN)
    assert jnp.array_equal(splits[0], embeds)


def test_split_embeddings_mixed_image_and_video():
    image_grid = (1, 8, 8)
    video_grid = (4, 8, 8)
    grid = (image_grid, video_grid)

    n_image = _video_token_count(*image_grid)
    n_video = _video_token_count(*video_grid)
    total = n_image + n_video
    assert (n_image, n_video) == (16, 64)

    embeds = jnp.arange(total * HIDDEN, dtype=jnp.float32).reshape(total, HIDDEN)
    splits, _ = split_mm_embeddings_by_grid(embeds, grid, SPATIAL_MERGE)

    assert len(splits) == 2
    assert splits[0].shape == (n_image, HIDDEN)
    assert splits[1].shape == (n_video, HIDDEN)
    assert jnp.array_equal(splits[0], embeds[:n_image])
    assert jnp.array_equal(splits[1], embeds[n_image:])


def test_merge_video_embeddings_into_input_ids():
    t, h, w = 2, 4, 4
    n_video_tokens = _video_token_count(t, h, w)
    assert n_video_tokens == 8

    text_prefix = [TEXT_TOKEN_ID] * 3
    text_suffix = [TEXT_TOKEN_ID] * 2
    input_ids = jnp.array(
        text_prefix + [VIDEO_TOKEN_ID] * n_video_tokens + text_suffix,
        dtype=jnp.int32,
    )
    seq_len = input_ids.shape[0]

    inputs_embeds = jnp.zeros((seq_len, HIDDEN), dtype=jnp.float32)
    video_embeds = jnp.tile(
        jnp.arange(1, n_video_tokens + 1, dtype=jnp.float32)[:, None],
        (1, HIDDEN),
    )

    merged = merge_multimodal_embeddings(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        multimodal_embeddings=video_embeds,
        placeholder_token_id=VIDEO_TOKEN_ID,
    )

    assert jnp.all(merged[:3] == 0)
    assert jnp.all(merged[3 + n_video_tokens:] == 0)
    np.testing.assert_array_equal(
        np.asarray(merged[3:3 + n_video_tokens]), np.asarray(video_embeds)
    )


def test_merge_handles_image_and_video_token_ids_together():
    n_image_tokens = 4
    n_video_tokens = 8
    total = n_image_tokens + n_video_tokens

    # Layout: [text, IMG x4, text, VIDEO x8, text]
    input_ids = jnp.array(
        [TEXT_TOKEN_ID] + [IMAGE_TOKEN_ID] * n_image_tokens + [TEXT_TOKEN_ID]
        + [VIDEO_TOKEN_ID] * n_video_tokens + [TEXT_TOKEN_ID],
        dtype=jnp.int32,
    )
    seq_len = input_ids.shape[0]
    inputs_embeds = jnp.zeros((seq_len, HIDDEN), dtype=jnp.float32)

    image_embeds = jnp.full((n_image_tokens, HIDDEN), 1.0, dtype=jnp.float32)
    video_embeds = jnp.full((n_video_tokens, HIDDEN), 2.0, dtype=jnp.float32)
    mm_embeds = jnp.concatenate([image_embeds, video_embeds], axis=0)

    merged = merge_multimodal_embeddings(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        multimodal_embeddings=mm_embeds,
        placeholder_token_id=[IMAGE_TOKEN_ID, VIDEO_TOKEN_ID],
    )

    assert merged[0, 0] == 0
    assert jnp.all(merged[1:1 + n_image_tokens] == 1.0)
    assert merged[1 + n_image_tokens, 0] == 0
    video_start = 1 + n_image_tokens + 1
    assert jnp.all(merged[video_start:video_start + n_video_tokens] == 2.0)
    assert merged[-1, 0] == 0
