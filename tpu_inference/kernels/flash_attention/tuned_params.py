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
    batch_size: int
    num_heads: int
    q_seq_len: int  # Assumes q_seq_len == kv_seq_len (self-attention).
    kv_seq_len: int
    head_dim: int
    q_dtype: str = "bfloat16"
    kv_dtype: str = "bfloat16"


@dataclass(frozen=True)
class TunableParams:
    block_q: int
    block_k_major: int
    block_k: int
    block_b: int

    def __ge__(self, other) -> bool:
        return False

    def __le__(self, other) -> bool:
        return False


tuned_params_mapping: dict[TuningKey, TunableParams] = {
    TuningKey(
        batch_size=8,
        num_heads=16,
        q_seq_len=2560,
        kv_seq_len=2560,
        head_dim=72,
    ):
    TunableParams(
        block_q=640,
        block_k_major=2560,
        block_k=512,
        block_b=1,
    ),
    TuningKey(
        batch_size=16,
        num_heads=16,
        q_seq_len=2560,
        kv_seq_len=2560,
        head_dim=72,
    ):
    TunableParams(
        block_q=640,
        block_k_major=2560,
        block_k=512,
        block_b=1,
    ),
}


def get_tuned_params(tuning_key: TuningKey) -> TunableParams | None:
    """Looks up tuned flash attention block sizes for the given shape.

    Returns None on a miss to fall back to the default.
    """
    tuned_params = tuned_params_mapping.get(tuning_key)
    if tuned_params is None:
        logger.warning_once(
            f"No tuned block sizes found for flash attention with tuning key: {tuning_key}, falling back to the default block size heuristic."
        )
    return tuned_params
