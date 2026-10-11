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
"""Single-host executor that installs the TPU hybrid KV-cache coordinator hooks.

vLLM's EngineCore builds the executor, then the KV caches, then the Scheduler.
The Scheduler's KVCacheManager picks its coordinator class at construction
time, so the TPUHybridKVCacheCoordinator hooks must already be installed in the
engine-core process by then. The multiproc and Ray executors do that in
``_init_executor``; the stock ``UniProcExecutor`` (used on a single host without
pipeline parallelism) did not, so any hybrid (mamba/GDN) model served with
prefix caching and without attention data parallelism silently ran vLLM's stock
coordinator: one shared block pool whose block ids exceed the compact TPU mamba
cache, which the model then clamps -- different requests end up sharing one
recurrent-state slot and their outputs bleed into each other. (With attention
DP the forked DPScheduler workers re-run the platform config hook and happened
to install the hooks in time, which is why TP/DP-attention configs were fine.)
"""

from vllm.v1.executor.uniproc_executor import \
    UniProcExecutor as UniProcExecutorV1

from tpu_inference.core.hybrid_coordinator import (
    MambaPoolSyncExecutorMixin, maybe_install_hybrid_coordinator_hooks)
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class UniProcExecutor(MambaPoolSyncExecutorMixin, UniProcExecutorV1):
    """UniProcExecutor that installs the hybrid coordinator hooks before the
    Scheduler (and therefore the KVCacheManager) is constructed."""

    def _init_executor(self) -> None:
        maybe_install_hybrid_coordinator_hooks(self.vllm_config)
        super()._init_executor()
