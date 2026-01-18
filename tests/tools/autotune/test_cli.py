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

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tpu_inference.tools.autotune import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_rpa_command(runner):
    with patch(
            "tpu_inference.tools.autotune.ragged_paged_attention_v3.tune_rpa"
    ) as mock_tune_rpa:
        result = runner.invoke(cli.rpa_v3, [
            "--page-size", "128", "--q-dtype", "bfloat16", "--kv-dtype",
            "bfloat16", "--num-iterations", "1", "--dry-run"
        ])

        assert result.exit_code == 0
        mock_tune_rpa.assert_called_once()
        call_kwargs = mock_tune_rpa.call_args.kwargs
        assert call_kwargs["page_sizes"] == [128]
        assert call_kwargs["dry_run"] is True


def test_quantized_matmul_command(runner):
    with patch("tpu_inference.tools.autotune.quantized_matmul.tune_matmul"
               ) as mock_tune_matmul:
        result = runner.invoke(cli.quantized_matmul, [
            "--batch-sizes", "128", "--out-in-features", "1024/1024",
            "--dry-run", "--num-iterations", "1"
        ])

        assert result.exit_code == 0
        mock_tune_matmul.assert_called_once()
        call_kwargs = mock_tune_matmul.call_args.kwargs
        assert call_kwargs["batch_sizes"] == [128]
        assert call_kwargs["out_in_features"] == [(1024, 1024)]
        assert call_kwargs["dry_run"] is True
