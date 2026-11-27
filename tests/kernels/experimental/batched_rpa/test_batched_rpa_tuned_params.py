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

import ast
import dataclasses
import json
from unittest import mock

import jax.numpy as jnp
import pytest

from tpu_inference.kernels.experimental.batched_rpa import configs
from tpu_inference.kernels.experimental.batched_rpa.tuned_params import (
    TunableParams, TuningKey, get_tuned_params)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GET_DEVICE_NAME = (
    "tpu_inference.kernels.experimental.batched_rpa.tuned_params"
    ".get_device_name")


@pytest.fixture(autouse=True)
def _default_device_name():
    """get_device_name() raises off-TPU, so pin it for every test."""
    with mock.patch(_GET_DEVICE_NAME, return_value="TPU v7x"):
        yield


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


def _mock_tpu_info(vmem_mib: int = 192) -> mock.MagicMock:
    info = mock.MagicMock()
    info.vmem_capacity_bytes = vmem_mib * 1024 * 1024
    return info


@pytest.mark.parametrize("num_seqs", [16, 64, 128])
@pytest.mark.parametrize("scale_kv", [None, 1.0, 0.85])
def test_v8i_qwen38_tp2_decode_tuned_entry_bucket_invariant(
        num_seqs, scale_kv):
    model_cfg = configs.ModelConfigs(
        num_q_heads=32,
        num_kv_heads=2,
        head_dim=256,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=num_seqs,
        page_size=256,
        total_q_tokens=num_seqs,
        num_page_indices=num_seqs * 64,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
        scale_k=scale_kv,
        scale_v=scale_kv,
    )
    with mock.patch(_GET_DEVICE_NAME, return_value="TPU v8i"), \
         mock.patch("jax.experimental.pallas.tpu.get_tpu_info",
                    return_value=_mock_tpu_info()), \
         mock.patch(
             "tpu_inference.kernels.experimental.batched_rpa.tuned_params"
             ".calculate_block_sizes"
         ) as mock_calc:
        result = get_tuned_params(model_cfg,
                                  serve_cfg,
                                  case="decode",
                                  decode_query_size=1)

    assert result == configs.BlockSizes(
        bq_sz=1,
        bq_c_sz=1,
        bkv_sz=2048,
        batch_size=4,
    )
    mock_calc.assert_not_called()


def test_v8i_tuned_entry_fallbacks_when_unmatched_or_constrained():
    model_cfg = configs.ModelConfigs(
        num_q_heads=32,
        num_kv_heads=2,
        head_dim=256,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=16,
        page_size=256,
        total_q_tokens=16,
        num_page_indices=16 * 64,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
    )
    fallback_dec = configs.BlockSizes(bq_sz=1,
                                      bq_c_sz=1,
                                      bkv_sz=512,
                                      batch_size=8)
    fallback_pre = configs.BlockSizes(bq_sz=512,
                                      bq_c_sz=128,
                                      bkv_sz=1024,
                                      batch_size=2)

    # 1. Different chip (v7x) -> falls back to calculate_block_sizes.
    with mock.patch(_GET_DEVICE_NAME, return_value="TPU v7x"), \
         mock.patch("jax.experimental.pallas.tpu.get_tpu_info",
                    return_value=_mock_tpu_info(vmem_mib=64)), \
         mock.patch(
             "tpu_inference.kernels.experimental.batched_rpa.tuned_params"
             ".calculate_block_sizes",
             return_value=(fallback_dec, fallback_pre),
         ) as mock_calc:
        assert get_tuned_params(model_cfg, serve_cfg,
                                case="decode") == fallback_dec
        mock_calc.assert_called_once()

    # 2. Speculative decode (decode_query_size > 1) -> falls back.
    with mock.patch(_GET_DEVICE_NAME, return_value="TPU v8i"), \
         mock.patch("jax.experimental.pallas.tpu.get_tpu_info",
                    return_value=_mock_tpu_info()), \
         mock.patch(
             "tpu_inference.kernels.experimental.batched_rpa.tuned_params"
             ".calculate_block_sizes",
             return_value=(fallback_dec, fallback_pre),
         ) as mock_calc:
        assert get_tuned_params(model_cfg,
                                serve_cfg,
                                case="decode",
                                decode_query_size=4) == fallback_dec
        mock_calc.assert_called_once()

    # 3. Reduced vmem_limit_bytes (< hardware VMEM) -> falls back.
    with mock.patch(_GET_DEVICE_NAME, return_value="TPU v8i"), \
         mock.patch("jax.experimental.pallas.tpu.get_tpu_info",
                    return_value=_mock_tpu_info(vmem_mib=192)), \
         mock.patch(
             "tpu_inference.kernels.experimental.batched_rpa.tuned_params"
             ".calculate_block_sizes",
             return_value=(fallback_dec, fallback_pre),
         ) as mock_calc:
        assert get_tuned_params(model_cfg,
                                serve_cfg,
                                vmem_limit_bytes=8 * 1024 * 1024,
                                case="decode") == fallback_dec
        mock_calc.assert_called_once()

    # 4. Short max_seq_len (< 2048) -> falls back.
    short_serve_cfg = configs.ServingConfigs(
        num_seqs=16,
        page_size=256,
        total_q_tokens=16,
        num_page_indices=16 * 2,  # max_seq_len = 512 < 2048
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
    )
    with mock.patch(_GET_DEVICE_NAME, return_value="TPU v8i"), \
         mock.patch("jax.experimental.pallas.tpu.get_tpu_info",
                    return_value=_mock_tpu_info()), \
         mock.patch(
             "tpu_inference.kernels.experimental.batched_rpa.tuned_params"
             ".calculate_block_sizes",
             return_value=(fallback_dec, fallback_pre),
         ) as mock_calc:
        assert get_tuned_params(model_cfg, short_serve_cfg,
                                case="decode") == fallback_dec
        mock_calc.assert_called_once()


# ---------------------------------------------------------------------------
# Kernel tuner compatibility (tools/kernel/tuner/v1/batched_rpa_kernel_tuner.py)
# ---------------------------------------------------------------------------

# A logged case as stored in
# tools/kernel/tuner/v1/tuning_cases/batched_rpa_gemma4_tuning_cases.json.
_LOGGED_TUNING_KEY = {
    "case": "decode",
    "num_q_heads": 8,
    "num_kv_heads": 4,
    "head_dim": 256,
    "num_seqs": 256,
    "page_size": 256,
    "total_q_tokens": 512,
    "num_page_indices": 4096,
    "dtype_q": "bfloat16",
    "dtype_kv": "float8_e4m3fn",
    "dtype_out": "bfloat16",
    "scale_q": None,
    "scale_k": 1.0,
    "scale_v": 1.0,
    "sliding_window": 1024,
}


def test_tuning_key_loads_logged_tuner_cases():
    """The kernel tuner rebuilds TuningKey(**logged_key) and reads the
    per-invocation fields to generate kernel inputs."""
    key = TuningKey(**_LOGGED_TUNING_KEY)
    assert key.num_seqs == 256
    assert key.total_q_tokens == 512
    assert key.num_page_indices == 4096
    assert key.scale_k == 1.0
    assert key.scale_v == 1.0


def test_tuning_key_json_and_repr_round_trip():
    """Keys built at serving time must round-trip through JSON (TuningCase
    logging) and repr() (autotune writes tuned_params_mapping source)."""
    serve_cfg = dataclasses.replace(_SERVE_CONFIG, scale_k=0.5, scale_v=0.25)
    with mock.patch(_GET_DEVICE_NAME, return_value="TPU v8i"), \
         mock.patch("jax.experimental.pallas.tpu.get_tpu_info",
                    return_value=_mock_tpu_info()):
        key = TuningKey.from_config(_MODEL_CONFIG, serve_cfg, case="decode")

    assert key.device_name == "TPU v8i"
    assert TuningKey(**json.loads(json.dumps(dataclasses.asdict(key)))) == key
    for value in dataclasses.asdict(key).values():
        assert ast.literal_eval(repr(value)) == value


def test_tuning_key_shape_only_clears_per_invocation_fields():
    key = TuningKey(**_LOGGED_TUNING_KEY)
    shape_key = key.shape_only()
    assert shape_key.num_seqs is None
    assert shape_key.total_q_tokens is None
    assert shape_key.num_page_indices is None
    assert shape_key.scale_q is None
    assert shape_key.scale_k is None
    assert shape_key.scale_v is None
    assert dataclasses.replace(
        key,
        num_seqs=None,
        total_q_tokens=None,
        num_page_indices=None,
        scale_q=None,
        scale_k=None,
        scale_v=None,
    ) == shape_key


def test_exact_match_takes_priority_over_shape_only_match():
    import tpu_inference.kernels.experimental.batched_rpa.tuned_params as tp_module

    exact_params = TunableParams(bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=2)
    shape_params = TunableParams(bq_sz=1, bq_c_sz=1, bkv_sz=512, batch_size=8)
    with mock.patch(_GET_DEVICE_NAME, return_value="TPU v8i"), \
         mock.patch("jax.experimental.pallas.tpu.get_tpu_info",
                    return_value=_mock_tpu_info()):
        key = TuningKey.from_config(_MODEL_CONFIG,
                                    _SERVE_CONFIG,
                                    case="decode")
        mapping = {key: exact_params, key.shape_only(): shape_params}
        with mock.patch.dict(tp_module.tuned_params_mapping,
                             mapping,
                             clear=True), \
             mock.patch(
                 "tpu_inference.kernels.experimental.batched_rpa.tuned_params"
                 ".calculate_block_sizes") as mock_calc:
            result = get_tuned_params(_MODEL_CONFIG,
                                      _SERVE_CONFIG,
                                      case="decode")

    assert result == exact_params.to_block_sizes()
    mock_calc.assert_not_called()
