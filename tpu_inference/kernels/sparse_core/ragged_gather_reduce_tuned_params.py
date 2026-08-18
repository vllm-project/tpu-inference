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

from pathlib import Path
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TuningKey:
    """Fixed parameters that define a ragged gather reduce problem shape."""
    input_size: int
    hidden_size: int
    reduce_group_size: int
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class TunableParams:
    """Tuning parameters for the ragged gather reduce kernel."""
    num_column_partitions: int
    num_row_partitions: int
    num_row_subchunks: int
    row_chunk_size: int
    aligned_hidden_size: int
    col_size: int
    col_chunk_size: int

    def __ge__(self, other) -> bool:
        if not isinstance(other, TunableParams):
            return NotImplemented
        return (
            self.num_column_partitions >= other.num_column_partitions
            and self.num_row_partitions >= other.num_row_partitions
            and self.num_row_subchunks >= other.num_row_subchunks
            and self.row_chunk_size >= other.row_chunk_size
            and self.aligned_hidden_size >= other.aligned_hidden_size
            and self.col_size >= other.col_size
            and self.col_chunk_size >= other.col_chunk_size
        )

    def __le__(self, other) -> bool:
        if not isinstance(other, TunableParams):
            return NotImplemented
        return (
            self.num_column_partitions <= other.num_column_partitions
            and self.num_row_partitions <= other.num_row_partitions
            and self.num_row_subchunks <= other.num_row_subchunks
            and self.row_chunk_size <= other.row_chunk_size
            and self.aligned_hidden_size <= other.aligned_hidden_size
            and self.col_size <= other.col_size
            and self.col_chunk_size <= other.col_chunk_size
        )

tuned_params_mapping: dict[TuningKey, TunableParams] = {
    TuningKey(
        input_size=131072,
        hidden_size=2816,
        reduce_group_size=8,
        dtype='bfloat16',
    ): TunableParams(
        num_column_partitions=1,
        num_row_partitions=16,
        num_row_subchunks=4,
        row_chunk_size=64,
        aligned_hidden_size=2816,
        col_size=2816,
        col_chunk_size=1408,
    ),
    TuningKey(
        input_size=16384,
        hidden_size=2816,
        reduce_group_size=8,
        dtype='bfloat16',
    ): TunableParams(
        num_column_partitions=2,
        num_row_partitions=8,
        num_row_subchunks=4,
        row_chunk_size=64,
        aligned_hidden_size=2816,
        col_size=1408,
        col_chunk_size=1408,
    ),
    TuningKey(
        input_size=32768,
        hidden_size=2816,
        reduce_group_size=8,
        dtype='bfloat16',
    ): TunableParams(
        num_column_partitions=1,
        num_row_partitions=16,
        num_row_subchunks=4,
        row_chunk_size=64,
        aligned_hidden_size=2816,
        col_size=2816,
        col_chunk_size=1408,
    ),
    TuningKey(
        input_size=4096,
        hidden_size=2816,
        reduce_group_size=8,
        dtype='bfloat16',
    ): TunableParams(
        num_column_partitions=2,
        num_row_partitions=8,
        num_row_subchunks=4,
        row_chunk_size=64,
        aligned_hidden_size=2816,
        col_size=1408,
        col_chunk_size=1408,
    ),
    TuningKey(
        input_size=65536,
        hidden_size=2816,
        reduce_group_size=8,
        dtype='bfloat16',
    ): TunableParams(
        num_column_partitions=1,
        num_row_partitions=16,
        num_row_subchunks=4,
        row_chunk_size=64,
        aligned_hidden_size=2816,
        col_size=2816,
        col_chunk_size=1408,
    ),
    TuningKey(
        input_size=8192,
        hidden_size=2816,
        reduce_group_size=8,
        dtype='bfloat16',
    ): TunableParams(
        num_column_partitions=2,
        num_row_partitions=8,
        num_row_subchunks=4,
        row_chunk_size=64,
        aligned_hidden_size=2816,
        col_size=1408,
        col_chunk_size=1408,
    ),
}



def calculate_tunable_params(
    tuning_key: TuningKey,
    num_cores: int = 16,
    num_lanes: int = 128,
    num_simd_lanes: int = 16,
) -> TunableParams:
    """Calculates default heuristic TunableParams for a given TuningKey."""
    hidden_size = tuning_key.hidden_size
    input_size = tuning_key.input_size

    preferred_num_stages = 4
    num_column_partitions = 1
    while (
        num_cores % (num_column_partitions * 2) == 0
        and hidden_size % (num_lanes * num_column_partitions * 2) == 0
        and hidden_size // (num_column_partitions * 2 * num_lanes)
        >= preferred_num_stages
    ):
        next_candidate = num_column_partitions * 2
        next_row_partitions = num_cores // next_candidate

        base_block_size = num_simd_lanes * next_row_partitions
        row_subchunks = max(1, min(4, pl.cdiv(input_size, base_block_size)))
        row_chunk = num_simd_lanes * row_subchunks
        num_iterations = input_size // (row_chunk * next_row_partitions)

        if num_cores // num_column_partitions > num_simd_lanes:
            num_column_partitions = next_candidate
            continue

        if num_iterations > 40:
            break

        num_column_partitions = next_candidate

    num_row_partitions = num_cores // num_column_partitions
    base_block_size = num_simd_lanes * num_row_partitions
    num_row_subchunks = max(1, min(4, pl.cdiv(input_size, base_block_size)))
    row_chunk_size = num_simd_lanes * num_row_subchunks

    aligned_hidden_size = pl.cdiv(hidden_size, 128 * num_column_partitions) * (
        128 * num_column_partitions
    )
    col_size = aligned_hidden_size // num_column_partitions

    try:
        generation = pltpu.get_tpu_info().generation
    except Exception:
        generation = 7

    match generation:
        case 6:
            target_bytes = int(256 * 1024 * 0.95)
        case 7:
            target_bytes = int(512 * 1024 * 0.95)
        case _:
            target_bytes = int(128 * 1024 * 0.95)

    bytes_per_col = num_simd_lanes * 4 * 2
    max_safe_col = (target_bytes // bytes_per_col // 128) * 128
    max_safe_col = min(max_safe_col, 1024)
    start_col = (min(col_size, max_safe_col) // 128) * 128
    col_chunk_size = 128
    for chunk in range(start_col, 127, -128):
        if col_size % chunk == 0:
            col_chunk_size = chunk
            break

    return TunableParams(
        num_column_partitions=num_column_partitions,
        num_row_partitions=num_row_partitions,
        num_row_subchunks=num_row_subchunks,
        row_chunk_size=row_chunk_size,
        aligned_hidden_size=aligned_hidden_size,
        col_size=col_size,
        col_chunk_size=col_chunk_size,
    )


def log_tunable_params(tuning_key: TuningKey, tunable_params: TunableParams):
    """Logs the given tuning case using TuningCaseLogger."""
    try:
        from tools.kernel.tuner.v1.common.tuning_case_logger import TuningCaseLogger
        log_file_path = (
            Path(__file__).resolve().parents[3]
            / "tools/kernel/tuner/v1/tuning_cases/ragged_gather_reduce_tuning_cases.json"
        )
        if log_file_path.parent.exists():
            case_logger = TuningCaseLogger(
                str(log_file_path),
                key_class=TuningKey,
                params_class=TunableParams,
            )
            existing_cases = (
                case_logger.get_logged_tuning_cases()
                if log_file_path.exists()
                else []
            )
            if not any(c.tuning_key == tuning_key for c in existing_cases):
                case_logger.log_tuning_case(tuning_key, tunable_params)
    except Exception as e:
        logger.warning(f"Failed to log tuning case for {tuning_key}: {e}")


def get_tuned_params(tuning_key: TuningKey) -> TunableParams:
    """Looks up tuned parameters for ragged_gather_reduce."""
    tuned_params = tuned_params_mapping.get(tuning_key)
    if tuned_params is None:
        logger.warning_once(
            f"No tuned params found for ragged_gather_reduce with key: {tuning_key}, "
            "falling back to default heuristic."
        )
        tuned_params = calculate_tunable_params(tuning_key)
        log_tunable_params(tuning_key, tuned_params)
    return tuned_params

