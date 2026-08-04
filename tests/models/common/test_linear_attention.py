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
"""Config -> recurrent-state layout for JAX-native hybrid models."""

from types import SimpleNamespace

import torch

from tpu_inference.models.common.linear_attention import (
    compute_linear_attention_layers, compute_linear_attention_layout,
    linear_attn_config)


def _config(**linear_attn):
    return SimpleNamespace(linear_attn_config=dict(linear_attn))


# Kimi-Linear-48B: 27 layers, KDA everywhere except the 7 full-attention ones.
_K48_KDA = [
    1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26
]


def test_no_linear_attention_config_returns_none():
    """Every attention-only model must take the unchanged code path."""
    assert compute_linear_attention_layout(SimpleNamespace(hidden_size=8),
                                           torch.bfloat16) is None
    assert compute_linear_attention_layers(SimpleNamespace(hidden_size=8)) == \
        frozenset()


def test_empty_kda_layer_list_returns_none():
    """A config that declares the sub-config but lists no KDA layers is not
    hybrid -- e.g. an all-full-attention ablation."""
    cfg = _config(kda_layers=[], num_heads=2, head_dim=8)
    assert compute_linear_attention_layout(cfg, torch.bfloat16) is None


def test_kda_layers_are_converted_from_one_indexed():
    """The checkpoint lists layers 1..N; everything downstream indexes 0..N-1.
    An off-by-one here would put the recurrent state on the wrong layers,
    which is silent -- the model would still run."""
    cfg = _config(kda_layers=[1, 2, 3, 5, 6, 7], num_heads=2, head_dim=8)
    assert compute_linear_attention_layers(cfg) == frozenset(
        {0, 1, 2, 4, 5, 6})
    assert 8 not in compute_linear_attention_layers(cfg)


def test_state_layout_matches_the_kda_state_tuple():
    """Three conv windows (q/k/v) at the model dtype, then the fp32
    recurrent matrix -- the field order of `KDAState`."""
    cfg = _config(kda_layers=[1],
                  num_heads=8,
                  head_dim=128,
                  short_conv_kernel_size=4)
    layout = compute_linear_attention_layout(cfg, torch.bfloat16)
    assert layout.shapes == ((3, 1024), (3, 1024), (3, 1024), (8, 128, 128))
    assert layout.dtypes == (torch.bfloat16, torch.bfloat16, torch.bfloat16,
                             torch.float32)
    # 3 * 3 * 1024 * 2 bytes + 8 * 128 * 128 * 4 bytes
    assert layout.page_size_bytes == 3 * 3 * 1024 * 2 + 8 * 128 * 128 * 4


def test_conv_window_depth_follows_the_kernel_size():
    """The conv state holds `kernel_size - 1` past positions."""
    for kernel in (2, 4, 8):
        layout = compute_linear_attention_layout(
            _config(kda_layers=[1],
                    num_heads=2,
                    head_dim=8,
                    short_conv_kernel_size=kernel), torch.bfloat16)
        assert layout.shapes[0] == (kernel - 1, 16)


def test_model_dtype_is_threaded_into_the_conv_states_only():
    """The recurrent state accumulates over the whole sequence and stays fp32
    regardless of the model dtype."""
    layout = compute_linear_attention_layout(
        _config(kda_layers=[1], num_heads=2, head_dim=8), torch.float32)
    assert layout.dtypes == (torch.float32, ) * 4
    layout = compute_linear_attention_layout(
        _config(kda_layers=[1], num_heads=2, head_dim=8), torch.float16)
    assert layout.dtypes[:3] == (torch.float16, ) * 3
    assert layout.dtypes[3] == torch.float32


def test_sub_config_may_be_an_object_rather_than_a_dict():
    """HF hands back a dict from config.json, but a config *class* that
    declares the field exposes an object instead."""
    cfg = SimpleNamespace(linear_attn_config=SimpleNamespace(
        kda_layers=[1, 3], num_heads=2, head_dim=8))
    assert linear_attn_config(cfg)["num_heads"] == 2
    assert compute_linear_attention_layers(cfg) == frozenset({0, 2})


def test_kimi_linear_48b_layer_split():
    """The shipped 48B config: 20 KDA + 7 full-attention out of 27."""
    cfg = SimpleNamespace(num_hidden_layers=27,
                          linear_attn_config={
                              "kda_layers": _K48_KDA,
                              "num_heads": 32,
                              "head_dim": 128,
                              "short_conv_kernel_size": 4
                          })
    layers = compute_linear_attention_layers(cfg)
    assert len(layers) == 20
    assert len(set(range(27)) - layers) == 7
    # The final layer (index 26) is full attention, as in the checkpoint.
    assert 26 not in layers
