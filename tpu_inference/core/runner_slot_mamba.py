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
"""GDN groups that take no scheduler blocks when prefix caching is off.

With prefix caching off, vLLM's MambaManager gives every request one block per
mamba group from the pool it shares with attention. The TPU runner never reads
those blocks for GDN layers: it keeps GDN state in compact arrays indexed by
its own per-request slot ids (`InputBatch.mamba_state_indices_cpu`, see
gdn_attention_op.py). For Qwen3.8-2.4T (3 GDN groups) at 8k input / 1k output
that is 3 of every 75 blocks a request holds.

With SKIP_MAMBA_SCHEDULER_BLOCKS set, the runner swaps the manager policy on
eligible GDN `KVCacheLayerConfig`s from `MambaManager` to
`RunnerSlotMambaManager`: a `MambaManager` subclass that allocates nothing.
The coordinator, block pool and the rest of the scheduler stay on vLLM's stock
caching-off path.

The policy is swapped only where the runner owns the slots and the path has
been validated: GDN layers, mamba cache mode "none", no speculative decoding
and no KV connector. Everything else keeps vLLM's MambaSpec and MambaManager.
"""

import dataclasses
from typing import TYPE_CHECKING

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import KVCacheLayerConfig, MambaSpec

from tpu_inference import envs
from tpu_inference.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class RunnerSlotMambaManager(MambaManager):
    """A MambaManager that allocates no blocks.

    The overrides take *args/**kwargs so they keep matching vLLM's signatures
    across releases. Freeing and skipped-block removal need no override: they
    walk the request's block list, which stays empty. KV-connector allocation
    is never reached: the policy is not swapped when a connector is configured.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        logger.info_once(
            "[runner_slot_mamba] GDN groups take no scheduler blocks "
            "(SKIP_MAMBA_SCHEDULER_BLOCKS).")

    def get_num_blocks_to_allocate(self, *args, **kwargs) -> int:
        return 0

    def allocate_new_blocks(self, *args, **kwargs) -> list[KVCacheBlock]:
        return []


def _ineligible_reason(layer_config: KVCacheLayerConfig,
                       vllm_config: "VllmConfig") -> str | None:
    spec = layer_config.spec
    if layer_config.manager_cls is not MambaManager:
        return (f"its manager is {layer_config.manager_cls.__name__}, "
                "not MambaManager")
    if type(spec) is not MambaSpec:
        return f"its spec is {type(spec).__name__}, not MambaSpec"
    if spec.mamba_type != MambaAttentionBackendEnum.GDN_ATTN:
        return f"it applies to GDN layers only, not {spec.mamba_type.name}"
    mode = vllm_config.cache_config.mamba_cache_mode
    if mode != "none":
        return f"mamba cache mode is {mode!r}, not 'none' (prefix caching)"
    if vllm_config.speculative_config is not None:
        return "speculative decoding is on"
    if vllm_config.kv_transfer_config is not None:
        return "a KV connector is configured"
    return None


def maybe_runner_slot_mamba_layer_config(
        layer_config: KVCacheLayerConfig,
        vllm_config: "VllmConfig") -> KVCacheLayerConfig:
    """Returns `layer_config` targeting `RunnerSlotMambaManager` when
    `SKIP_MAMBA_SCHEDULER_BLOCKS` applies, otherwise `layer_config` unchanged.
    """
    if (not envs.SKIP_MAMBA_SCHEDULER_BLOCKS
            or not isinstance(layer_config.spec, MambaSpec)):
        return layer_config
    reason = _ineligible_reason(layer_config, vllm_config)
    if reason is not None:
        logger.warning_once(
            "[runner_slot_mamba] Ignoring SKIP_MAMBA_SCHEDULER_BLOCKS for "
            "this layer: %s. It keeps its scheduler blocks.", reason)
        return layer_config
    return dataclasses.replace(
        layer_config,
        manager_config=dataclasses.replace(layer_config.manager_config,
                                           manager_cls=RunnerSlotMambaManager),
    )
