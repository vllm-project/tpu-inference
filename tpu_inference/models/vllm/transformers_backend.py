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
"""Make vLLM's Transformers modeling backend build global-view models.

tpu-inference runs vLLM model code on global (unsharded) arrays: vLLM's
tensor-parallel group is initialized with size 1, so vLLM's native models size
their layers for the whole model and the JAX mesh does the sharding.

vLLM's Transformers backend instead sizes two things from
`parallel_config.tensor_parallel_size` directly, i.e. per shard:

1. Its attention layers, through `ModelConfig.get_num_attention_heads` /
   `get_num_kv_heads(parallel_config)`, which divide by the TP size.
2. The split of each fused QKV / merged-column output, generated as
   `[s // layer.tp_size for s in layer.output_sizes]`. With KV-head replication
   (TP > number of KV heads), `VllmQKVParallelLinear.output_sizes` stays
   inflated while its forward returns de-replicated K/V, so this split carves
   q/k/v at the wrong offsets (V ends up empty).

With TP > 1 either mismatch breaks the model (e.g. codegemma-2b, which vLLM
serves through this backend since vllm-project/vllm#66393). `global_view()`
corrects both while the model is built; nothing is patched afterwards.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# The head-count helpers only read `tensor_parallel_size` from the config.
_GLOBAL_VIEW_PARALLEL_CONFIG = SimpleNamespace(tensor_parallel_size=1)


def _split_sizes_source(merged_name: str) -> str:
    """Source for the widths a fused output is split into.

    Prefers the layer's `tpu_split_sizes` (set by `VllmQKVParallelLinear`) and
    otherwise keeps vLLM's own expression, so other merged layers are
    unaffected.
    """
    merged = f"self.{merged_name}"
    return (f"(getattr({merged}, 'tpu_split_sizes', None) or "
            f"[s // {merged}.tp_size for s in {merged}.output_sizes])")


@contextlib.contextmanager
def global_view():
    """Build vLLM Transformers-backend models with global head counts and
    fused-output splits that match tpu-inference's linear layers."""
    try:
        from vllm.config import ModelConfig
        from vllm.model_executor.models.transformers import base as tf_base
        from vllm.model_executor.models.transformers.fusers import \
            base as fusers_base
        from vllm.model_executor.models.transformers.fusers import \
            packed_qkv as fusers_packed_qkv
    except ImportError:
        # A vLLM without the Transformers backend (or with a different
        # layout): nothing to adjust.
        yield
        return

    orig_num_heads = ModelConfig.get_num_attention_heads
    orig_num_kv_heads = ModelConfig.get_num_kv_heads
    orig_create_attention = tf_base.Base._create_attention_instances

    def _global_num_heads(model_config, parallel_config, arch_config=None):
        return orig_num_heads(model_config, _GLOBAL_VIEW_PARALLEL_CONFIG,
                              arch_config)

    def _global_num_kv_heads(model_config, parallel_config, arch_config=None):
        return orig_num_kv_heads(model_config, _GLOBAL_VIEW_PARALLEL_CONFIG,
                                 arch_config)

    def _create_attention_instances(self, *args, **kwargs):
        # Only the backend's attention layers see global head counts; anything
        # else that sizes per-shard state (e.g. the KV-cache spec) is untouched.
        with patch.object(ModelConfig, "get_num_attention_heads",
                          _global_num_heads), \
                patch.object(ModelConfig, "get_num_kv_heads",
                             _global_num_kv_heads):
            return orig_create_attention(self, *args, **kwargs)

    with patch.object(tf_base.Base, "_create_attention_instances",
                      _create_attention_instances), \
            patch.object(fusers_base, "local_output_sizes",
                         _split_sizes_source), \
            patch.object(fusers_packed_qkv, "local_output_sizes",
                         _split_sizes_source):
        yield
