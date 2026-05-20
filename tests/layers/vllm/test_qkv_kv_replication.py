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

from unittest.mock import MagicMock, patch

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


def _build_qkv(*, mesh_shape, total_num_kv_heads):
    """Construct VllmQKVParallelLinear with parent init stubbed.

    `mesh_shape` is the dict of mesh-axis → size exposed on
    `vllm_config.quant_config.mesh.shape`. Only the ATTN_HEAD axes
    ('model', 'expert', 'dcp') are read by the layer.

    Returns (layer, parent_kv_heads) where `parent_kv_heads` records the
    (inflated) kv-head count handed to `QKVParallelLinear.__init__`.
    """
    cfg = MagicMock()
    cfg.quant_config.mesh.shape = mesh_shape

    parent_kv_heads = []
    mod = "tpu_inference.layers.vllm.custom_ops.linear"
    with patch(f"{mod}.QKVParallelLinear.__init__",
               lambda *args, **_: parent_kv_heads.append(args[4])), \
         patch(f"{mod}.get_current_vllm_config", return_value=cfg):
        layer = VllmQKVParallelLinear(
            hidden_size=256,
            head_size=128,
            total_num_heads=8,
            total_num_kv_heads=total_num_kv_heads,
        )
    return layer, parent_kv_heads


def test_replicas_tp_exceeds_kv_heads():
    # mesh ATTN_HEAD product = 8 > num_kv_heads=2 → replicate each KV head 4x.
    layer, parent_kv_heads = _build_qkv(mesh_shape={"model": 8},
                                        total_num_kv_heads=2)
    assert layer.tp_size == 8
    assert layer.num_kv_head_replicas == 4
    # Inflated kv-heads passed to parent: 2 * 4 = 8.
    assert parent_kv_heads == [8]


def test_replicas_tp_equals_kv_heads():
    # mesh ATTN_HEAD product = 2 == num_kv_heads=2 → no replication.
    layer, parent_kv_heads = _build_qkv(mesh_shape={"model": 2},
                                        total_num_kv_heads=2)
    assert layer.tp_size == 2
    assert layer.num_kv_head_replicas == 1
    assert parent_kv_heads == [2]


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
