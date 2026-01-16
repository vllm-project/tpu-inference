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

from tpu_inference.tools.autotune.quantized_matmul import (factors_of_n,
                                                           make_configs,
                                                           tune_matmul)


def test_factors_of_n():
    # factors of 128 step 128 -> [128]
    assert factors_of_n(128, 128) == [128]

    # factors of 256 step 128 -> [128, 256]
    assert factors_of_n(256, 128) == [128, 256]

    # factors of 100 step 128 -> next multiple 128 -> [128]
    assert factors_of_n(100, 128) == [128]


def test_make_configs():
    batch_sizes = [128]
    out_in_features = [(128, 128)]
    configs = make_configs(batch_sizes, out_in_features, "int8", "int8")

    # batch=128 -> blocks=[128]
    # out=128 -> blocks=[128]
    # in=128 -> blocks=[128]
    # Total 1 config
    assert len(configs) == 1

    key, value = configs[0]
    assert key.n_batch == 128
    assert value.batch_block_size == 128


@patch("tpu_inference.tools.autotune.quantized_matmul.autotune_kernel")
@patch("tpu_inference.tools.autotune.utils.update_json_registry")
@patch("tpu_inference.tools.autotune.utils.get_registry_file_name")
@patch("tpu_inference.utils.get_tpu_name_slug")
@patch(
    "tpu_inference.kernels.quantized_matmul.tuned_block_sizes.get_tpu_version")
def test_tune_matmul_flow(mock_tpu_ver, mock_slug, mock_registry_name,
                          mock_update, mock_autotune):
    # mocks
    mock_tpu_ver.return_value = 5  # Mock TPU v5
    mock_slug.return_value = "tpu_v5e"
    mock_registry_name.return_value = "tpu_v5e"

    # Mock autotune return: (latency, std, compile, lower)
    mock_autotune.side_effect = [
        (10.0, 1.0, 0.1, 0.1),
    ]

    tune_matmul(
        batch_sizes=[128],
        out_in_features=[(128, 128)],
        x_q_dtype="int8",
        w_q_dtype="int8",
        num_iterations=1,
        update_registry=True,
    )

    assert mock_autotune.called
    assert mock_update.called

    # Verify update data
    args, _ = mock_update.call_args
    path, data = args
    assert "tpu_v5e.json" in path

    # Check if data contains our result
    # Key format: "batch,out,in,xdtype,wdtype"
    key = "128,128,128,int8,int8"
    assert key in data
    assert data[key]["stats"]["latency_avg_ns"] == 10.0


@patch("tpu_inference.tools.autotune.quantized_matmul.autotune_kernel")
@patch(
    "tpu_inference.kernels.quantized_matmul.tuned_block_sizes.get_tpu_version")
def test_tune_matmul_tp_scaling(mock_tpu_ver, mock_autotune):
    # Set return value
    mock_tpu_ver.return_value = 5
    mock_autotune.return_value = (10.0, 0.0, 0.0, 0.0)

    # TP=2, Split=out
    tune_matmul(batch_sizes=[128],
                out_in_features=[(256, 128)],
                tp_size=2,
                tp_split_dim="out",
                num_iterations=1)

    # Expect out_features to be scaled 256 -> 128
    assert mock_autotune.called
    args = mock_autotune.call_args[0]
    key = args[0]
    assert key.n_out == 128  # 256 / 2
    assert key.n_in == 128  # Unchanged

    # TP=2, Split=in
    tune_matmul(batch_sizes=[128],
                out_in_features=[(128, 256)],
                tp_size=2,
                tp_split_dim="in",
                num_iterations=1)

    # Expect in_features to be scaled 256 -> 128
    key = mock_autotune.call_args[0][0]
    assert key.n_in == 128
