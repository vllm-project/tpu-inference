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
    TunableParams, TuningKey, get_tuned_params)

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
    n_buffer=2,
)

_BLOCK_SIZES = _TUNABLE.to_block_sizes()

# ---------------------------------------------------------------------------
# get_tuned_params – empty-mapping fallback
# ---------------------------------------------------------------------------


def test_get_tuned_params_fallback_to_calculate_block_sizes():
    """When the key is absent from tuned_params_mapping, get_tuned_params must
    delegate to calculate_block_sizes and return whatever it produces."""

    decode_bs = configs.BlockSizes(bq_sz=1,
                                   bq_c_sz=1,
                                   bkv_sz=1,
                                   batch_size=1,
                                   n_buffer=1)
    prefill_bs = configs.BlockSizes(bq_sz=2,
                                    bq_c_sz=2,
                                    bkv_sz=2,
                                    batch_size=2,
                                    n_buffer=2)

    import tpu_inference.kernels.experimental.batched_rpa.tuned_params as tp_module

    with mock.patch.dict(tp_module.tuned_params_mapping, {}, clear=True), \
         mock.patch(
             "tpu_inference.kernels.experimental.batched_rpa.wrapper"
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
             "tpu_inference.kernels.experimental.batched_rpa.wrapper"
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
            TunableParams(
                bq_sz=16, bq_c_sz=8, bkv_sz=32, batch_size=4, n_buffer=2),
            TunableParams(
                bq_sz=16, bq_c_sz=8, bkv_sz=32, batch_size=4, n_buffer=2),
            True,
            True,
        ),
        # lo is strictly smaller in every dimension
        (
            TunableParams(
                bq_sz=8, bq_c_sz=4, bkv_sz=16, batch_size=2, n_buffer=1),
            TunableParams(
                bq_sz=16, bq_c_sz=8, bkv_sz=32, batch_size=4, n_buffer=2),
            True,
            False,
        ),
        # lo is strictly larger in every dimension
        (
            TunableParams(
                bq_sz=32, bq_c_sz=16, bkv_sz=64, batch_size=8, n_buffer=4),
            TunableParams(
                bq_sz=16, bq_c_sz=8, bkv_sz=32, batch_size=4, n_buffer=2),
            False,
            True,
        ),
        # mixed: lo has some dims larger – neither le nor ge should be True
        (
            TunableParams(
                bq_sz=8, bq_c_sz=16, bkv_sz=32, batch_size=4, n_buffer=2),
            TunableParams(
                bq_sz=16, bq_c_sz=8, bkv_sz=32, batch_size=4, n_buffer=2),
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
    base = TunableParams(bq_sz=16,
                         bq_c_sz=8,
                         bkv_sz=32,
                         batch_size=4,
                         n_buffer=2)
    bigger_bq = TunableParams(bq_sz=32,
                              bq_c_sz=8,
                              bkv_sz=32,
                              batch_size=4,
                              n_buffer=2)

    # base is ≤ bigger_bq in every dimension (all others are equal)
    assert (base <= bigger_bq) is True
    assert (base >= bigger_bq) is False  # base.bq_sz < bigger_bq.bq_sz
    assert (bigger_bq >= base) is True  # all dims of bigger_bq ≥ base
    assert (bigger_bq <= base) is False  # bigger_bq.bq_sz > base.bq_sz
