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

from vllm.v1.executor.multiproc_executor import \
    MultiprocExecutor as MultiprocExecutorV1

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class MultiprocExecutor(MultiprocExecutorV1):
    """
    MultiprocExecutor override to support TPU inference.

    The main change is to support MPMD for Pipeline Parallelism, while keeping
    SPMD for the rest of parallelisms (TP, CP, EP, DP).
    """

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        self.world_size = self.parallel_config.pipeline_parallel_size
        self.local_world_size = self.world_size
        tp_size = 1
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = 1
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        # set up jax transfer connection.
        for rank in range(1, self.world_size):
            self.collective_rpc("initialize_pp_transfer_connect",
                                unique_reply_rank=rank)

    def _get_output_rank(self) -> int:
        return self.world_size - 1

    def _is_driver_worker(self, rank: int) -> bool:
        return True
