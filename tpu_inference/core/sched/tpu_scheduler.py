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

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from tpu_inference.logger import init_logger
from tpu_inference.runner.continuous_block_pool import ContinuousFreeQueue

logger = init_logger(__name__)


class DisaggTpuScheduler(Scheduler):
    """Custom vLLM Scheduler for TPUs that uses ContinuousFreeQueue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pool = self.kv_cache_manager.block_pool
        pool.free_block_queue = ContinuousFreeQueue(pool.blocks)
        pool.null_block = pool.free_block_queue.popleft()
        pool.null_block.is_null = True


class DisaggTpuAsyncScheduler(AsyncScheduler):
    """Custom vLLM AsyncScheduler for TPUs that uses ContinuousFreeQueue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pool = self.kv_cache_manager.block_pool
        pool.free_block_queue = ContinuousFreeQueue(pool.blocks)
        pool.null_block = pool.free_block_queue.popleft()
        pool.null_block.is_null = True


def update_vllm_config_for_tpu_scheduler(vllm_config) -> None:
    """
    Update vLLM configuration to use TpuScheduler when in disaggregated mode.
    """
    is_disaggregated = vllm_config.kv_transfer_config is not None

    if is_disaggregated:
        if vllm_config.scheduler_config.async_scheduling:
            logger.warning("Using DisaggTpuAsyncScheduler")
            vllm_config.scheduler_config.scheduler_cls = DisaggTpuAsyncScheduler
        else:
            logger.warning("Using DisaggTpuScheduler")
            vllm_config.scheduler_config.scheduler_cls = DisaggTpuScheduler
