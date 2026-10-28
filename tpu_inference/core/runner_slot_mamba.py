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

With SKIP_MAMBA_SCHEDULER_BLOCKS set, the runner hands vLLM its GDN specs as
`RunnerSlotMambaSpec`. vLLM picks each group's manager from the spec's class,
and TpuPlatform registers `RunnerSlotMambaManager` for this one: a
MambaManager that allocates nothing. The coordinator, block pool and the rest
of the scheduler stay on vLLM's stock caching-off path.

The spec is swapped only where the runner owns the slots and the path has been
validated: GDN layers, mamba cache mode "none", no speculative decoding and no
KV connector. Everything else keeps vLLM's MambaSpec and MambaManager.
"""

import dataclasses
from typing import TYPE_CHECKING

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from tpu_inference import envs
from tpu_inference.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class RunnerSlotMambaSpec(MambaSpec):
    """A GDN MambaSpec whose state slots the TPU runner assigns itself.

    Registered by register_runner_slot_mamba_spec, not @register_kv_cache_spec:
    a registration at import time would run before vLLM's built-in specs and
    keep them from loading.
    """


class RunnerSlotMambaManager(MambaManager):
    """A MambaManager that allocates no blocks.

    The overrides take *args/**kwargs so they keep matching vLLM's signatures
    across releases. Freeing and skipped-block removal need no override: they
    walk the request's block list, which stays empty. KV-connector allocation
    is never reached: the spec is not swapped when a connector is configured.
    """

    def __init__(self, kv_cache_spec: MambaSpec, block_pool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        logger.info_once(
            "[runner_slot_mamba] GDN groups take no scheduler blocks "
            "(SKIP_MAMBA_SCHEDULER_BLOCKS).")

    def get_num_blocks_to_allocate(self, *args, **kwargs) -> int:
        return 0

    def allocate_new_blocks(self, *args, **kwargs) -> list[KVCacheBlock]:
        return []


def _ineligible_reason(spec: MambaSpec,
                       vllm_config: "VllmConfig") -> str | None:
    # Exact class only: a subclass may carry fields RunnerSlotMambaSpec cannot
    # take, or be registered to a manager of its own.
    if type(spec) is not MambaSpec:
        return f"its spec is {type(spec).__name__}, not MambaSpec"
    if spec.mamba_type != MambaAttentionBackendEnum.GDN_ATTN:
        return f"it applies to GDN layers only, not {spec.mamba_type.name}"
    # The spec's own mamba_cache_mode is copied from this config by vLLM.
    mode = vllm_config.cache_config.mamba_cache_mode
    if mode != "none":
        return f"mamba cache mode is {mode!r}, not 'none' (prefix caching)"
    if vllm_config.speculative_config is not None:
        return "speculative decoding is on"
    if vllm_config.kv_transfer_config is not None:
        return "a KV connector is configured"
    return None


def maybe_runner_slot_mamba_spec(spec: KVCacheSpec,
                                 vllm_config: "VllmConfig") -> KVCacheSpec:
    """Returns `spec` as a RunnerSlotMambaSpec when SKIP_MAMBA_SCHEDULER_BLOCKS
    applies to it, otherwise `spec` unchanged."""
    if not envs.SKIP_MAMBA_SCHEDULER_BLOCKS or not isinstance(spec, MambaSpec):
        return spec
    reason = _ineligible_reason(spec, vllm_config)
    if reason is not None:
        logger.warning_once(
            "[runner_slot_mamba] Ignoring SKIP_MAMBA_SCHEDULER_BLOCKS for "
            "this layer: %s. It keeps its scheduler blocks.", reason)
        return spec
    return RunnerSlotMambaSpec(**{
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec) if f.init
    })


def register_runner_slot_mamba_spec() -> None:
    """Maps RunnerSlotMambaSpec to RunnerSlotMambaManager in vLLM's registry.

    Called from TpuPlatform.register_custom_kv_cache_specs, which vLLM runs
    right after registering its built-in specs, in each process that looks a
    manager up. Registering the same pair again is a no-op.
    """
    KVCacheSpecRegistry.register(RunnerSlotMambaSpec,
                                 RunnerSlotMambaManager,
                                 uniform_type_base_spec=MambaSpec)
