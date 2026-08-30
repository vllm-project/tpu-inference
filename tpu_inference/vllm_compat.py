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
"""Shims that bridge KV-cache API differences across supported vLLM versions.

tpu-inference must run against both the pinned LKG vLLM and vLLM HEAD. vLLM
commit 8bdc70ec7b ("Standardize KV cache layout") reshaped several KV-cache
interfaces:

- ``MLAAttentionSpec.compress_ratio`` became ``AttentionSpec.tokens_per_state``
  and ``storage_block_size`` became ``num_states``.
- ``KVCacheTensor`` no longer lists the layers aliasing one tensor
  (``shared_by``); instead each tensor spans all same-spec layers of a group
  as one strided allocation (``layers``/``layer_stride``/``block_stride``),
  and cache groups alias by overlapping byte ranges.

The helpers here present the older shapes to the rest of tpu-inference on
either vLLM version.
"""

import dataclasses
from typing import List

from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheSpec,
                                        KVCacheTensor, MLAAttentionSpec)

_MLA_SPEC_HAS_COMPRESS_RATIO = "compress_ratio" in {
    f.name
    for f in dataclasses.fields(MLAAttentionSpec)
}

KV_CACHE_TENSOR_HAS_SHARED_BY = "shared_by" in {
    f.name
    for f in dataclasses.fields(KVCacheTensor)
}


def make_mla_attention_spec(*,
                            compress_ratio: int = 1,
                            **kwargs) -> MLAAttentionSpec:
    """Build an MLAAttentionSpec, mapping ``compress_ratio`` to whichever
    field the running vLLM version uses for it."""
    if _MLA_SPEC_HAS_COMPRESS_RATIO:
        return MLAAttentionSpec(compress_ratio=compress_ratio, **kwargs)
    return MLAAttentionSpec(tokens_per_state=compress_ratio, **kwargs)


def get_compress_ratio(spec: KVCacheSpec) -> int:
    """Tokens compressed into one stored KV state (1 = no compression)."""
    ratio = getattr(spec, "compress_ratio", None)
    if ratio is not None:
        return ratio
    return int(spec.tokens_per_state)


def get_storage_block_size(spec: KVCacheSpec) -> int:
    """Stored states per block (= block_size / compression ratio)."""
    size = getattr(spec, "storage_block_size", None)
    if size is not None:
        return size
    return spec.num_states


@dataclasses.dataclass
class LegacyKVCacheTensor:
    """Old-style KVCacheTensor: one tensor per set of layers aliasing the
    same memory, sized for a single layer's pages."""
    size: int  # size of the KV cache tensor in bytes
    shared_by: List[str]  # layer names that share the same KV cache tensor
    offset: int = 0  # byte offset of this layer within a contiguous block
    block_stride: int = 0  # bytes per block in a packed layout (0 = not packed)


def make_kv_cache_tensor(*,
                         size: int,
                         shared_by: List[str],
                         offset: int = 0,
                         block_stride: int = 0):
    """Build an old-style KV cache tensor: vLLM's own KVCacheTensor when it
    still has ``shared_by``, our stand-in dataclass otherwise (mainly for
    tests emulating the engine core)."""
    if KV_CACHE_TENSOR_HAS_SHARED_BY:
        return KVCacheTensor(size=size,
                             shared_by=shared_by,
                             offset=offset,
                             block_stride=block_stride)
    return LegacyKVCacheTensor(size=size,
                               shared_by=shared_by,
                               offset=offset,
                               block_stride=block_stride)


def legacy_kv_cache_tensors(kv_cache_config: KVCacheConfig,
                            layer_name_to_spec: dict[str, KVCacheSpec]):
    """Present ``kv_cache_config.kv_cache_tensors`` in the old per-aliasing-set
    shape regardless of vLLM version.

    Old-style tensors (with ``shared_by``) pass through unchanged. New-style
    strided tensors are regrouped: layers whose regions start at the same
    absolute byte offset alias each other (that is how the new engine core
    overlays cache groups), so each distinct start becomes one legacy tensor,
    ordered by start offset and sized so ``size // page_size_bytes`` yields
    the config's block count.
    """
    tensors = kv_cache_config.kv_cache_tensors
    if all(hasattr(t, "shared_by") for t in tensors):
        return list(tensors)

    start_to_layers: dict[int, List[str]] = {}
    for tensor in tensors:
        for index, layer_name in enumerate(tensor.layers):
            start = tensor.offset + index * tensor.layer_stride
            start_to_layers.setdefault(start, []).append(layer_name)

    return [
        LegacyKVCacheTensor(size=kv_cache_config.num_blocks *
                            layer_name_to_spec[layer_names[0]].page_size_bytes,
                            shared_by=layer_names)
        for _, layer_names in sorted(start_to_layers.items())
    ]
