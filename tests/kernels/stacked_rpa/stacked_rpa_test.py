# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""CPU contract tests for the mode-specific stacked-RPA implementation."""

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.stacked_rpa import (configs,
                                                            flash_attention)
from tpu_inference.kernels.experimental.stacked_rpa.decode import \
    block_sizes as decode_block_sizes
from tpu_inference.kernels.experimental.stacked_rpa.decode import \
    config as decode_config
from tpu_inference.kernels.experimental.stacked_rpa.prefill import \
    block_sizes as prefill_block_sizes
from tpu_inference.kernels.experimental.stacked_rpa.prefill import \
    config as prefill_config

_VMEM_LIMIT_BYTES = 128 * 1024 * 1024


def _make_model_config(
    *,
    sliding_window: int | None = None,
) -> configs.ModelConfigs:
    return configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=128,
        mask_value=-1e30,
        sliding_window=sliding_window,
    )


def _make_serving_config(
    *,
    total_q_tokens: int = 128,
    pages_per_seq: int = 8,
) -> configs.ServingConfigs:
    return configs.ServingConfigs(
        num_seqs=4,
        page_size=128,
        total_q_tokens=total_q_tokens,
        num_page_indices=4 * pages_per_seq,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
    )


def _make_decode_block_sizes(
    *,
    bq: int = 1,
    bkv: int = 256,
    batch: int = 4,
    n_buffer: int = 2,
) -> decode_block_sizes.BlockSizes:
    return decode_block_sizes.BlockSizes(
        bq_sz=bq,
        bkv_sz=bkv,
        batch_size=batch,
        n_buffer=n_buffer,
    )


def _make_prefill_block_sizes(
    *,
    bq: int = 1,
    bq_c: int = 1,
    bkv: int = 256,
    batch: int = 4,
    n_buffer: int = 2,
) -> prefill_block_sizes.BlockSizes:
    return prefill_block_sizes.BlockSizes(
        bq_sz=bq,
        bq_c_sz=bq_c,
        bkv_sz=bkv,
        batch_size=batch,
        n_buffer=n_buffer,
    )


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_make_decode_block_sizes, id="decode"),
        pytest.param(_make_prefill_block_sizes, id="prefill"),
    ],
)
@pytest.mark.parametrize("n_buffer", [1, 3])
def test_block_sizes_require_double_buffering(factory, n_buffer):
    with pytest.raises(ValueError, match="requires double buffering"):
        factory(n_buffer=n_buffer)


def test_decode_config_builder_creates_valid_decode_config():
    block = _make_decode_block_sizes(bq=4)
    cfg = decode_config.make_config(
        _make_model_config(),
        _make_serving_config(),
        block,
        _VMEM_LIMIT_BYTES,
        decode_q_len=4,
    )

    cfg.validate_decode()
    assert cfg.mode is configs.RpaCase.DECODE
    assert cfg.decode_q_len == 4
    assert cfg.dense_pack


@pytest.mark.parametrize(
    ("mode", "dense_pack", "expected_dense_pack"),
    [
        pytest.param(configs.RpaCase.MIXED, False, False, id="mixed"),
        pytest.param(configs.RpaCase.MIXED, True, True, id="mixed-dense-pack"),
        pytest.param(configs.RpaCase.PREFILL, True, False, id="prefill"),
    ],
)
def test_prefill_config_builder_enables_dense_pack_only_for_mixed(
    mode,
    dense_pack,
    expected_dense_pack,
):
    cfg = prefill_config.make_config(
        mode,
        _make_model_config(),
        _make_serving_config(),
        _make_prefill_block_sizes(),
        _VMEM_LIMIT_BYTES,
        dense_pack=dense_pack,
    )

    assert cfg.mode is mode
    assert cfg.dense_pack is expected_dense_pack
    assert cfg.is_stacked is expected_dense_pack


def test_prefill_override_is_preserved_and_page_aligned():
    chosen = prefill_block_sizes.choose_block_sizes(
        _make_model_config(),
        _make_serving_config(total_q_tokens=32),
        _VMEM_LIMIT_BYTES,
        override=_make_prefill_block_sizes(
            bq=512,
            bq_c=128,
            bkv=129,
            batch=1,
        ),
    )

    assert chosen == _make_prefill_block_sizes(
        bq=512,
        bq_c=128,
        bkv=256,
        batch=1,
    )


def test_sliding_window_kv_span_is_page_aligned():
    chosen = decode_block_sizes.required_window_bkv_size(
        _make_model_config(sliding_window=4096),
        _make_serving_config(pages_per_seq=64),
        decode_q_len=4,
    )

    assert chosen == 4352


def test_decode_block_floors_bq_without_replacing_bkv():
    block = _make_decode_block_sizes(bq=1, bkv=8192, batch=2)

    assert block.floor_bq_to_decode_q_len(4) == _make_decode_block_sizes(
        bq=4,
        bkv=8192,
        batch=2,
    )


@pytest.mark.parametrize(
    ("size_delta", "skip_mask", "expect_unmasked"),
    [
        pytest.param(0, None, False, id="unconditional-mask"),
        pytest.param(0, 0, False, id="small-apply"),
        pytest.param(0, 1, True, id="small-skip"),
        pytest.param(1, 0, False, id="large-apply"),
        pytest.param(1, 1, True, id="large-skip"),
    ],
)
def test_attention_mask(
    size_delta,
    skip_mask,
    expect_unmasked,
):
    qk = jnp.arange(
        flash_attention._BRANCHLESS_MASK_MAX_ELEMENTS + size_delta,
        dtype=jnp.float32,
    ).reshape(1, 1, -1)
    mask_value = jnp.asarray(-1e30, dtype=qk.dtype)
    actual = flash_attention._apply_attention_mask(
        qk,
        lambda: jnp.zeros_like(qk, dtype=jnp.bool_),
        None if skip_mask is None else jnp.asarray(skip_mask),
        mask_value,
    )
    expected = qk if expect_unmasked else jnp.full_like(qk, mask_value)

    np.testing.assert_array_equal(actual, expected)
