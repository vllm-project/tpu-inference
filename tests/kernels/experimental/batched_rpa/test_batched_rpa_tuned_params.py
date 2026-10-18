# Copyright 2026 Google LLC
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

from unittest import mock

import jax.numpy as jnp
import pytest

from tpu_inference.kernels.experimental.batched_rpa import configs
from tpu_inference.kernels.experimental.batched_rpa.tuned_params import (
    TunableParams,
    TuningKey,
    calculate_block_sizes,
    get_tuned_params,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MODEL_CONFIG = configs.ModelConfigs(
    num_q_heads=16,
    num_kv_heads=2,
    head_dim=128,
    mask_value=-1e9,
)

_SERVE_CONFIG = configs.ServingConfigs(
    num_seqs=4,
    page_size=16,
    total_q_tokens=64,
    num_page_indices=64,
    dtype_q=jnp.bfloat16,
    dtype_kv=jnp.bfloat16,
    dtype_out=jnp.bfloat16,
)

_TUNABLE = TunableParams(
    bq_sz=32,
    bq_c_sz=16,
    bkv_sz=64,
    batch_size=4,
)

_BLOCK_SIZES = _TUNABLE.to_block_sizes()

# ---------------------------------------------------------------------------
# get_tuned_params – empty-mapping fallback
# ---------------------------------------------------------------------------


def test_get_tuned_params_fallback_to_calculate_block_sizes():
    """When the key is absent from tuned_params_mapping, get_tuned_params must
    delegate to calculate_block_sizes and return whatever it produces."""

    decode_bs = configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=1, batch_size=1)
    prefill_bs = configs.BlockSizes(bq_sz=2, bq_c_sz=2, bkv_sz=2, batch_size=2)

    import tpu_inference.kernels.experimental.batched_rpa.tuned_params as tp_module

    with mock.patch.dict(tp_module.tuned_params_mapping, {}, clear=True), \
         mock.patch(
             "tpu_inference.kernels.experimental.batched_rpa.tuned_params"
             ".calculate_block_sizes",
             return_value=(decode_bs, prefill_bs),
         ) as mock_calc:

        result_decode = get_tuned_params(_MODEL_CONFIG,
                                         _SERVE_CONFIG,
                                         vmem_limit_bytes=1 << 28,
                                         case='decode')
        result_prefill = get_tuned_params(_MODEL_CONFIG,
                                          _SERVE_CONFIG,
                                          vmem_limit_bytes=1 << 28,
                                          case='prefill')

    assert result_decode == decode_bs
    assert result_prefill == prefill_bs
    assert mock_calc.call_count == 2


# ---------------------------------------------------------------------------
# get_tuned_params – populated-mapping hit
# ---------------------------------------------------------------------------


def test_get_tuned_params_populated_mapping_hit():
    """When the key is present in tuned_params_mapping the pre-tuned
    BlockSizes should be returned without calling calculate_block_sizes."""

    import tpu_inference.kernels.experimental.batched_rpa.tuned_params as tp_module

    key = TuningKey.from_config(_MODEL_CONFIG, _SERVE_CONFIG, case='decode')
    mapping = {key: _TUNABLE}

    with mock.patch.dict(tp_module.tuned_params_mapping, mapping, clear=True), \
         mock.patch(
             "tpu_inference.kernels.experimental.batched_rpa.tuned_params"
             ".calculate_block_sizes",
         ) as mock_calc:

        result = get_tuned_params(_MODEL_CONFIG,
                                  _SERVE_CONFIG,
                                  vmem_limit_bytes=1 << 28,
                                  case='decode')

    assert result == _BLOCK_SIZES
    mock_calc.assert_not_called()


# ---------------------------------------------------------------------------
# TunableParams.__ge__ and __le__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lo,hi,expect_le,expect_ge",
    [
        # identical params → both True
        (
            TunableParams(bq_sz=16, bq_c_sz=8, bkv_sz=32, batch_size=4),
            TunableParams(bq_sz=16, bq_c_sz=8, bkv_sz=32, batch_size=4),
            True,
            True,
        ),
        # lo is strictly smaller in every dimension
        (
            TunableParams(bq_sz=8, bq_c_sz=4, bkv_sz=16, batch_size=2),
            TunableParams(bq_sz=16, bq_c_sz=8, bkv_sz=32, batch_size=4),
            True,
            False,
        ),
        # lo is strictly larger in every dimension
        (
            TunableParams(bq_sz=32, bq_c_sz=16, bkv_sz=64, batch_size=8),
            TunableParams(bq_sz=16, bq_c_sz=8, bkv_sz=32, batch_size=4),
            False,
            True,
        ),
        # mixed: lo has some dims larger – neither le nor ge should be True
        (
            TunableParams(bq_sz=8, bq_c_sz=16, bkv_sz=32, batch_size=4),
            TunableParams(bq_sz=16, bq_c_sz=8, bkv_sz=32, batch_size=4),
            False,
            False,
        ),
    ])
def test_tunable_params_ge_le(lo, hi, expect_le, expect_ge):
    assert (lo <= hi) is expect_le, f"Expected lo<=hi to be {expect_le}"
    assert (lo >= hi) is expect_ge, f"Expected lo>=hi to be {expect_ge}"


def test_tunable_params_ge_le_single_dim_difference():
    """When only one dimension differs the larger side is >= and the smaller
    side is <=, because both operators require ALL dims to satisfy the
    relation."""
    base = TunableParams(bq_sz=16, bq_c_sz=8, bkv_sz=32, batch_size=4)
    bigger_bq = TunableParams(bq_sz=32, bq_c_sz=8, bkv_sz=32, batch_size=4)

    # base is ≤ bigger_bq in every dimension (all others are equal)
    assert (base <= bigger_bq) is True
    assert (base >= bigger_bq) is False  # base.bq_sz < bigger_bq.bq_sz
    assert (bigger_bq >= base) is True  # all dims of bigger_bq ≥ base
    assert (bigger_bq <= base) is False  # bigger_bq.bq_sz > base.bq_sz


def test_calculate_block_sizes_page_size_and_sliding_window():
    """Verify bkv_sz is always a multiple of page_size and clamped to sliding_window."""
    mock_info = mock.MagicMock()
    mock_info.num_lanes = 128
    mock_info.num_sublanes = 8
    mock_info.mxu_column_size = 128
    mock_info.generation = 6
    mock_info.chip_version = "v6e"
    mock_info.fp8_ops_per_second = 2
    mock_info.bf16_ops_per_second = 1

    with mock.patch("jax.experimental.pallas.tpu.get_tpu_info",
                    return_value=mock_info):
        # 1. Sliding window layer across TP=1..8 (page_size=256, sliding_window=1024)
        for tp in (1, 2, 4, 8):
            m_sw = configs.ModelConfigs(
                num_q_heads=32 // tp,
                num_kv_heads=16 // tp,
                head_dim=256,
                mask_value=-1e9,
                sliding_window=1024,
            )
            s_sw = configs.ServingConfigs(
                num_seqs=64,
                page_size=256,
                total_q_tokens=2048,
                num_page_indices=64 * 32,
                dtype_q=jnp.bfloat16,
                dtype_kv=jnp.float8_e4m3fn,
                dtype_out=jnp.bfloat16,
            )
            d_sw, p_sw = calculate_block_sizes(m_sw,
                                               s_sw,
                                               vmem_limit_bytes=128 * 1024 *
                                               1024)
            assert d_sw.bkv_sz % s_sw.page_size == 0
            assert p_sw.bkv_sz % s_sw.page_size == 0
            assert d_sw.bkv_sz == 256
            assert p_sw.bkv_sz <= 1024

        # 2. Global layer across TP=1..4 & dtypes (head_dim=512, page_size=512)
        for tp in (1, 2, 4):
            for kv_dtype in (jnp.float8_e4m3fn, jnp.bfloat16):
                for kv_layout in (configs.KVLayout.HEAD_ALONG_SUBLANE,
                                  configs.KVLayout.SEQ_ALONG_LANE):
                    m_gl = configs.ModelConfigs(
                        num_q_heads=16 // tp,
                        num_kv_heads=max(1, 4 // tp),
                        head_dim=512,
                        mask_value=-1e9,
                        sliding_window=None,
                    )
                    s_gl = configs.ServingConfigs(
                        num_seqs=64,
                        page_size=512,
                        total_q_tokens=2048,
                        num_page_indices=64 * 16,
                        dtype_q=jnp.bfloat16,
                        dtype_kv=kv_dtype,
                        dtype_out=jnp.bfloat16,
                        kv_layout=kv_layout,
                    )
                    d_gl, p_gl = calculate_block_sizes(m_gl,
                                                       s_gl,
                                                       vmem_limit_bytes=128 *
                                                       1024 * 1024)
                    assert d_gl.bkv_sz % s_gl.page_size == 0
                    assert p_gl.bkv_sz % s_gl.page_size == 0

        # 3. Tight VMEM (64 MiB v7x / 128 MiB v6e with page_size=512/1024) and
        # non-power-of-two page_size=48 must stay aligned without VMEM OOM.
        for vmem_mb, page_sz in ((64, 512), (64, 1024), (128, 1024), (128,
                                                                      48)):
            m_tight = configs.ModelConfigs(
                num_q_heads=16,
                num_kv_heads=4,
                head_dim=512,
                mask_value=-1e9,
                sliding_window=None,
            )
            s_tight = configs.ServingConfigs(
                num_seqs=64,
                page_size=page_sz,
                total_q_tokens=2048,
                num_page_indices=64 * 16,
                dtype_q=jnp.bfloat16,
                dtype_kv=jnp.bfloat16,
                dtype_out=jnp.bfloat16,
            )
            d_t, p_t = calculate_block_sizes(m_tight,
                                             s_tight,
                                             vmem_limit_bytes=vmem_mb * 1024 *
                                             1024)
            assert d_t.bkv_sz % page_sz == 0
            assert d_t.bkv_sz % 128 == 0
            assert p_t.bkv_sz % page_sz == 0
            assert p_t.bkv_sz % 128 == 0
            assert p_t.bq_sz % 128 == 0

    # 4. SFS (v8i) prefill max_bq_sz=512 cap is preserved even when page_size=1024.
    mock_sfs = mock.MagicMock()
    mock_sfs.num_lanes = 128
    mock_sfs.num_sublanes = 8
    mock_sfs.mxu_column_size = 128
    mock_sfs.generation = 8
    mock_sfs.fp8_ops_per_second = 2
    mock_sfs.bf16_ops_per_second = 1
    with mock.patch("jax.experimental.pallas.tpu.get_tpu_info",
                    return_value=mock_sfs), \
         mock.patch.object(configs, "is_sfs", return_value=True):
        m_sfs = configs.ModelConfigs(
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=256,
            mask_value=-1e9,
        )
        s_sfs = configs.ServingConfigs(
            num_seqs=64,
            page_size=1024,
            total_q_tokens=4096,
            num_page_indices=64 * 16,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.float8_e4m3fn,
            dtype_out=jnp.bfloat16,
        )
        d_sfs, p_sfs = calculate_block_sizes(m_sfs,
                                             s_sfs,
                                             vmem_limit_bytes=256 * 1024 *
                                             1024)
        assert d_sfs.bkv_sz % 1024 == 0
        assert p_sfs.bkv_sz % 1024 == 0
        assert p_sfs.bq_sz <= configs.SFS_PREFILL_MAX_BQ_SZ
