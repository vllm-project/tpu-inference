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
"""Scheduler entry points that install compact Mamba pools in EngineCore."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from tpu_inference.core.compact_mamba_pool import install_compact_mamba_pool


class CompactMambaScheduler(Scheduler):
    """Standard scheduler with TPU compact Mamba pool support installed."""

    def __init__(self, *args, **kwargs) -> None:
        install_compact_mamba_pool()
        super().__init__(*args, **kwargs)


class CompactMambaAsyncScheduler(AsyncScheduler):
    """Async scheduler with TPU compact Mamba pool support installed."""

    def __init__(self, *args, **kwargs) -> None:
        install_compact_mamba_pool()
        super().__init__(*args, **kwargs)
