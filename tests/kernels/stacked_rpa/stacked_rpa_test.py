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

import dataclasses

import jax.numpy as jnp
import pytest

from tpu_inference.kernels.experimental.stacked_rpa import (configs, envvars,
                                                            tuned_block_sizes,
                                                            wrapper)


def _cfgs(mode=configs.RpaCase.DECODE):
    return configs.RpaConfigs(
        block=configs.BlockSizes(
            bq_sz=1,
            bq_c_sz=1,
            bkv_sz=128,
            batch_size=4,
            n_buffer=2,
        ),
        model=configs.ModelConfigs(
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=128,
            mask_value=-1e30,
        ),
        serve=configs.ServingConfigs(
            num_seqs=4,
            page_size=128,
            total_q_tokens=4,
            num_page_indices=32,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.bfloat16,
            dtype_out=jnp.bfloat16,
            kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        ),
        mode=mode,
        vmem_limit_bytes=128 * 1024 * 1024,
    )


def test_decode_is_stacked():
    assert _cfgs().is_stacked


@pytest.mark.parametrize("mode",
                         [configs.RpaCase.MIXED, configs.RpaCase.PREFILL])
def test_non_decode_is_not_stacked_by_default(mode):
    assert not _cfgs(mode).is_stacked


def test_dense_mixed_is_stacked_but_prefill_is_not():
    mixed = dataclasses.replace(_cfgs(configs.RpaCase.MIXED), dense_pack=True)
    prefill = dataclasses.replace(_cfgs(configs.RpaCase.PREFILL),
                                  dense_pack=True)

    assert mixed.is_stacked
    assert not prefill.is_stacked


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("STACKED_RPA_BKV", "512")
    monkeypatch.setenv("STACKED_RPA_SW_BOUND", "false")

    assert envvars.stacked_rpa_bkv() == 512
    assert not envvars.stacked_rpa_sw_bound()


def test_tuned_lookup_falls_back_when_table_is_empty():
    cfgs = _cfgs()

    assert not tuned_block_sizes.TUNED_BLOCK_SIZES
    assert (tuned_block_sizes.get_tuned_block_sizes(
        cfgs.model,
        cfgs.serve,
        max_model_len=4096,
        mode=cfgs.mode,
    ) is None)


def test_rejects_unsupported_kv_layout():
    with pytest.raises(ValueError, match="SEQ_ALONG_LANE"):
        wrapper._require_seq_along_lane(configs.KVLayout.HEAD_ALONG_SUBLANE)
