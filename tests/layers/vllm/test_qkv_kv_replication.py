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

from unittest.mock import MagicMock

import torch

from tpu_inference.layers.vllm.custom_ops.linear import VllmQKVParallelLinear


def _call_tile_kv(w, *, total_num_kv_heads, num_kv_head_replicas, output_dim):
    dummy = MagicMock()
    dummy.total_num_kv_heads = total_num_kv_heads
    dummy.num_kv_head_replicas = num_kv_head_replicas
    param = torch.nn.Parameter(torch.empty(0))
    if output_dim is not None:
        param.output_dim = output_dim
    return VllmQKVParallelLinear._tile_kv(dummy, w, param)


def test_tile_kv_2d_weight_along_output_dim_0():
    # 4 KV heads × 2 rows/head along dim 0, hidden_size = 3.
    w = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 1],  # head 0
            [2, 2, 2],
            [2, 2, 2],  # head 1
            [3, 3, 3],
            [3, 3, 3],  # head 2
            [4, 4, 4],
            [4, 4, 4],  # head 3
        ],
        dtype=torch.float32,
    )
    out = _call_tile_kv(w,
                        total_num_kv_heads=4,
                        num_kv_head_replicas=2,
                        output_dim=0)
    expected = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [4, 4, 4],
            [4, 4, 4],
            [4, 4, 4],
            [4, 4, 4],
        ],
        dtype=torch.float32,
    )
    assert out.shape == (16, 3)
    assert torch.equal(out, expected)


def test_tile_kv_along_output_dim_1():
    # 2 KV heads × 3 cols/head along dim 1.
    w = torch.tensor(
        [[1, 1, 1, 2, 2, 2], [10, 10, 10, 20, 20, 20]],
        dtype=torch.float32,
    )
    out = _call_tile_kv(w,
                        total_num_kv_heads=2,
                        num_kv_head_replicas=2,
                        output_dim=1)
    expected = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            [10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(out, expected)
