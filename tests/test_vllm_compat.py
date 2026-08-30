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

from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock

import torch

from tpu_inference.vllm_compat import (get_compress_ratio,
                                       get_storage_block_size,
                                       legacy_kv_cache_tensors,
                                       make_kv_cache_tensor,
                                       make_mla_attention_spec)


@dataclass
class _NewStyleTensor:
    """Shape of vLLM's post-8bdc70ec7b KVCacheTensor: one strided span over
    all same-spec layers of a cache group."""
    size: int
    layers: List[str]
    layer_stride: int
    block_stride: int
    offset: int = 0


def _config(num_blocks, tensors):
    config = MagicMock()
    config.num_blocks = num_blocks
    config.kv_cache_tensors = tensors
    return config


def _spec(page_size_bytes):
    spec = MagicMock()
    spec.page_size_bytes = page_size_bytes
    return spec


class TestMLASpecCompat:

    def test_round_trips_compress_ratio(self):
        spec = make_mla_attention_spec(block_size=1024,
                                       num_kv_heads=1,
                                       head_size=640,
                                       dtype=torch.uint8,
                                       compress_ratio=4)
        assert get_compress_ratio(spec) == 4
        assert get_storage_block_size(spec) == 1024 // 4

    def test_defaults_to_no_compression(self):
        spec = make_mla_attention_spec(block_size=128,
                                       num_kv_heads=1,
                                       head_size=64,
                                       dtype=torch.bfloat16)
        assert get_compress_ratio(spec) == 1
        assert get_storage_block_size(spec) == 128


class TestLegacyKVCacheTensors:

    def test_old_style_passes_through(self):
        tensors = [
            make_kv_cache_tensor(size=1024, shared_by=["a"]),
            make_kv_cache_tensor(size=1024, shared_by=["b"]),
        ]
        config = _config(num_blocks=8, tensors=tensors)
        assert legacy_kv_cache_tensors(config, {}) == tensors

    def test_layer_outer_groups_alias_by_start(self):
        # Two cache groups, two layers each, layer-outermost layout: layer i
        # of each group starts at i * layer_stride, so layers at the same
        # start must land in one shared_by set, in group order.
        page, num_blocks = 128, 8
        stride = page * num_blocks
        tensors = [
            _NewStyleTensor(size=2 * stride,
                            layers=["g0.l0", "g0.l1"],
                            layer_stride=stride,
                            block_stride=page),
            _NewStyleTensor(size=2 * stride,
                            layers=["g1.l0", "g1.l1"],
                            layer_stride=stride,
                            block_stride=page),
        ]
        specs = {
            name: _spec(page)
            for name in ["g0.l0", "g0.l1", "g1.l0", "g1.l1"]
        }
        legacy = legacy_kv_cache_tensors(_config(num_blocks, tensors), specs)

        assert [t.shared_by for t in legacy] == [["g0.l0", "g1.l0"],
                                                 ["g0.l1", "g1.l1"]]
        for t in legacy:
            assert t.size == num_blocks * page
            assert t.size % page == 0
            assert t.size // page == num_blocks
            assert t.block_stride == 0

    def test_distinct_offsets_stay_separate(self):
        # Block-outermost packed layout: layers of one group are adjacent
        # within a block (layer_stride == their page size), and a second
        # group overlays only the matching start offset.
        num_blocks = 4
        tensors = [
            _NewStyleTensor(size=4096,
                            layers=["main", "idx"],
                            layer_stride=512,
                            block_stride=1024),
            _NewStyleTensor(size=4096,
                            layers=["state"],
                            layer_stride=512,
                            block_stride=1024,
                            offset=0),
        ]
        specs = {"main": _spec(512), "idx": _spec(512), "state": _spec(512)}
        legacy = legacy_kv_cache_tensors(_config(num_blocks, tensors), specs)

        assert [t.shared_by for t in legacy] == [["main", "state"], ["idx"]]
