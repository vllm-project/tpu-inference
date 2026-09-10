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

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _get_sc_allreduce_allgather_offload_min_size_bytes() -> int:
    """Returns the SparseCore all-reduce/all-gather offload minimum size in bytes.

    Returns 0 if we use default XLA offload threshold.
    """
    sc_threshold_val = envs.SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES
    sc_threshold_bytes = 0
    if sc_threshold_val == "auto":
        from tpu_inference.tpu_info import get_tpu_vmem_size_bytes
        sc_threshold_bytes = get_tpu_vmem_size_bytes()
    else:
        try:
            sc_threshold_bytes = int(sc_threshold_val)
        except ValueError:
            logger.warning(
                f"Invalid value for SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES: "
                f"'{sc_threshold_val}'. Defaulting to 0 (always offload).")
            sc_threshold_bytes = 0
    return sc_threshold_bytes


def get_step_fn_compiler_options():
    """Returns compiler options for the step function."""

    compiler_options = {
        "xla_tpu_all_gather_collective_matmul_mode": "post_spmd_conservative",
        "xla_tpu_reduce_scatter_collective_matmul_mode":
        "post_spmd_conservative",
        "xla_tpu_use_minor_sharding_for_major_trivial_input": "true",
    }
    sc_offload_bytes = _get_sc_allreduce_allgather_offload_min_size_bytes()
    if sc_offload_bytes > 0:
        threshold_bytes = str(sc_offload_bytes)
        compiler_options[
            "xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes"] = (
                threshold_bytes)
        compiler_options[
            "xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"] = (
                threshold_bytes)

    return compiler_options
