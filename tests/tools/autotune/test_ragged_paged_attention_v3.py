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

from unittest.mock import patch

from tpu_inference.tools.autotune.benchmarks import BenchmarkResult
from tpu_inference.tools.autotune.ragged_paged_attention_v3 import (
    RpaBlock, RpaKey, make_rpa_configs, tune_rpa)


def test_make_rpa_configs():
    # Setup simple inputs
    page_sizes = [16]
    q_dtypes = ["bfloat16"]
    kv_dtypes = ["int8"]
    num_q_heads = [32]
    num_kv_heads = [8]  # Divisible
    head_dims = [64]
    max_model_lens = [128]

    bkv_p = [1, 2]
    bq_sz = [32]

    configs = make_rpa_configs(page_sizes, q_dtypes, kv_dtypes, num_q_heads,
                               num_kv_heads, head_dims, max_model_lens, bkv_p,
                               bq_sz)

    # Expect 2 configs (1 for each bkv_p)
    assert len(configs) == 2

    # Verify content
    key, block = configs[0]
    assert isinstance(key, RpaKey)
    assert isinstance(block, RpaBlock)
    assert key.max_model_len == 128
    assert block.num_q_per_block == 32


def test_make_rpa_configs_filters_indivisible():
    # num_q=32, num_kv=7 (Not divisible)
    configs = make_rpa_configs([16], ["bf16"], ["int8"], [32], [7], [64],
                               [128], [1], [32])
    assert len(configs) == 0


def test_make_rpa_configs_filters_invalid_blocks():
    # page_size=128, bkv=32 -> 128*32 = 4096 (OK)
    # page_size=128, bkv=33 -> 4224 > 4096 (Skip)
    configs = make_rpa_configs([128], ["bf16"], ["int8"], [32], [8], [64],
                               [4096], [32, 33], [32])
    # Should only keep bkv=32
    assert len(configs) == 1
    assert configs[0][1].num_kv_pages_per_block == 32


@patch("tpu_inference.tools.autotune.utils.RunContext")  # Mock RunContext
@patch(
    "tpu_inference.tools.autotune.ragged_paged_attention_v3.benchmark_kernel")
@patch("tpu_inference.tools.autotune.utils.update_json_registry")
@patch("tpu_inference.tools.autotune.utils.get_registry_file_name")
@patch("tpu_inference.utils.get_tpu_name_slug")
def test_tune_rpa_flow(mock_slug, mock_registry_name, mock_update,
                       mock_benchmark, mock_run_context):
    # mocks
    mock_slug.return_value = "tpu_v5e"
    mock_registry_name.return_value = "tpu_v5e"

    # Mock benchmark output: (mean, std, compile, lower)
    # Mock benchmark output: needs to be BenchmarkResult objects now
    mock_benchmark.side_effect = [
        BenchmarkResult(10.0, 1.0, 0.1, 0.1, [], {}),
        BenchmarkResult(5.0, 0.5, 0.1, 0.1, [], {}),
    ]

    tune_rpa(
        page_sizes=[16],
        q_dtypes=["bfloat16"],
        kv_dtypes=["int8"],
        num_q_heads_list=[32],
        num_kv_heads_list=[8],
        head_dims=[128],
        max_model_lens=[128],
        kv_block_sizes=[1, 2],  # 2 configs
        q_block_sizes=[32],
        num_iterations=1,
        update_registry=True,
    )

    # Check benchmark calls
    assert mock_benchmark.call_count == 2

    # Check registry update
    assert mock_update.called
    args, _ = mock_update.call_args
    path, data = args
    assert "tpu_v5e.json" in path

    # Verify the best result (latency 5.0) was chosen for the JSON
    assert data["16"]["q_bfloat16_kv_int8"]["q_head-32_kv_head-8_head-128"][
        "max_model_len-128-sw-None"]["stats"]["latency_avg_ns"] == 5.0


@patch("tpu_inference.tools.autotune.utils.RunContext")
@patch(
    "tpu_inference.tools.autotune.ragged_paged_attention_v3.benchmark_kernel")
def test_tune_rpa_tp_scaling(mock_benchmark, mock_run_context):
    # Set return value to prevent unpacking error
    mock_benchmark.return_value = BenchmarkResult(10.0, 0.0, 0.0, 0.0, [], {})

    # Test that TP scaling logic adjusts head counts
    # TP=4, Q=64 -> 16, KV=8 -> 2
    tune_rpa(
        page_sizes=[16],
        q_dtypes=["bfloat16"],
        kv_dtypes=["int8"],
        num_q_heads_list=[64],
        num_kv_heads_list=[8],
        head_dims=[128],
        max_model_lens=[128],
        kv_block_sizes=[1],
        q_block_sizes=[32],
        num_iterations=1,
        tp_size=4,  # ENABLE TP
    )

    # Verify benchmark was called with SCALED key
    assert mock_benchmark.called
    key = mock_benchmark.call_args[0][0]  # first arg is RpaKey
    assert key.num_q_heads == 16
    assert key.num_kv_heads == 2
