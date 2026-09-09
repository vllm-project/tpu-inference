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
        return (self.num_column_partitions >= other.num_column_partitions
                and self.num_row_partitions >= other.num_row_partitions
                and self.num_row_subchunks >= other.num_row_subchunks
                and self.row_chunk_size >= other.row_chunk_size
                and self.aligned_hidden_size >= other.aligned_hidden_size
                and self.col_size >= other.col_size
                and self.col_chunk_size >= other.col_chunk_size)

    def __le__(self, other) -> bool:
        if not isinstance(other, TunableParams):
            return NotImplemented
        return (self.num_column_partitions <= other.num_column_partitions
                and self.num_row_partitions <= other.num_row_partitions
                and self.num_row_subchunks <= other.num_row_subchunks
                and self.row_chunk_size <= other.row_chunk_size
                and self.aligned_hidden_size <= other.aligned_hidden_size
                and self.col_size <= other.col_size
                and self.col_chunk_size <= other.col_chunk_size)


tuned_params_mapping: dict[TuningKey, TunableParams] = {
    TuningKey(
        input_size=4096,
        hidden_size=2816,
        reduce_group_size=8,
        dtype='bfloat16',
    ):
    TunableParams(
        num_column_partitions=2,
        num_row_partitions=16,
        num_row_subchunks=4,
        row_chunk_size=64,
        aligned_hidden_size=2816,
        col_size=1408,
        col_chunk_size=1408,
    ),
    TuningKey(
        input_size=8192,
        hidden_size=2816,
        reduce_group_size=8,
        dtype='bfloat16',
    ):
    TunableParams(
        num_column_partitions=2,
        num_row_partitions=16,
        num_row_subchunks=4,
        row_chunk_size=64,
        aligned_hidden_size=2816,
        col_size=1408,
        col_chunk_size=1408,
    ),
    TuningKey(
        input_size=16384,
        hidden_size=2816,
        reduce_group_size=8,
        dtype='bfloat16',
    ):
    TunableParams(
        num_column_partitions=2,
        num_row_partitions=16,
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
    ):
    TunableParams(
        num_column_partitions=2,
        num_row_partitions=16,
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
    ):
    TunableParams(
        num_column_partitions=2,
        num_row_partitions=16,
        num_row_subchunks=4,
        row_chunk_size=64,
        aligned_hidden_size=2816,
        col_size=1408,
        col_chunk_size=1408,
    ),
    TuningKey(
        input_size=131072,
        hidden_size=2816,
        reduce_group_size=8,
        dtype='bfloat16',
    ):
    TunableParams(
        num_column_partitions=2,
        num_row_partitions=16,
        num_row_subchunks=4,
        row_chunk_size=64,
        aligned_hidden_size=2816,
        col_size=1408,
        col_chunk_size=1408,
    ),
}


def get_tuned_params(tuning_key: TuningKey) -> TunableParams | None:
    """Looks up tuned parameters for ragged_gather_reduce."""
    tuned_params = tuned_params_mapping.get(tuning_key)
    if tuned_params is None:
        logger.warning_once(
            f"No tuned params found for ragged_gather_reduce with ke0y: {tuning_key}, "
            "falling back to default heuristic.")
    return tuned_params
