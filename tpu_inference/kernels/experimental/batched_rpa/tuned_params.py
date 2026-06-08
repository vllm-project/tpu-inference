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

from dataclasses import asdict, dataclass
from typing import Literal

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.batched_rpa import configs
from tpu_inference.kernels.experimental.batched_rpa.configs_from_log import \
    LOG_ENTRIES
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TuningKey:
    case: Literal['decode', 'prefill']
    # model config
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    # mask_value: float

    # serve config parameters
    num_seqs: int
    page_size: int
    total_q_tokens: int
    num_page_indices: int
    dtype_q: str
    dtype_kv: str
    dtype_out: str
    scale_q: int | None = None
    scale_k: int | None = None
    scale_v: int | None = None

    # model config default params
    # sm_scale: float = 1.0
    # soft_cap: float | None = None
    sliding_window: int | None = None

    @staticmethod
    def from_config(model_config: configs.ModelConfigs,
                    serve_config: configs.ServingConfigs,
                    case: Literal['decode', 'prefill']) -> 'TuningKey':
        return TuningKey(
            num_q_heads=model_config.num_q_heads,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
            # mask_value=model_config.mask_value,
            # sm_scale=model_config.sm_scale,
            # soft_cap=model_config.soft_cap,
            sliding_window=model_config.sliding_window,
            num_seqs=serve_config.num_seqs,
            page_size=serve_config.page_size,
            total_q_tokens=serve_config.total_q_tokens,
            num_page_indices=serve_config.num_page_indices,
            dtype_q=jnp.dtype(serve_config.dtype_q).name,
            dtype_kv=jnp.dtype(serve_config.dtype_kv).name,
            dtype_out=jnp.dtype(serve_config.dtype_out).name,
            scale_q=serve_config.scale_q,
            scale_k=serve_config.scale_k,
            scale_v=serve_config.scale_v,
            case=case)


@dataclass(frozen=True)
class TunableParams:
    """Tuning parameters for the RPA kernel."""
    bq_sz: int
    bq_c_sz: int
    bkv_sz: int
    batch_size: int
    n_buffer: int
    is_baseline: bool = False

    def to_block_sizes(self) -> configs.BlockSizes:
        return configs.BlockSizes(bq_sz=self.bq_sz,
                                  bq_c_sz=self.bq_c_sz,
                                  bkv_sz=self.bkv_sz,
                                  batch_size=self.batch_size,
                                  n_buffer=self.n_buffer)

    def __ge__(self, other) -> bool:
        return self.bq_sz >= other.bq_sz and self.bq_c_sz >= other.bq_c_sz and self.bkv_sz >= other.bkv_sz and self.batch_size >= other.batch_size


def get_tuned_params(
    model_config: configs.ModelConfigs,
    serve_config: configs.ServingConfigs,
    vmem_limit_bytes: int | None = None
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    decode_tuning_key = TuningKey.from_config(model_config,
                                              serve_config,
                                              case='decode')
    prefill_tuning_key = TuningKey.from_config(model_config,
                                               serve_config,
                                               case='prefill')
    if decode_tuning_key not in tuned_params_mapping:
        from tpu_inference.kernels.experimental.batched_rpa.wrapper import \
            calculate_block_sizes
        decode_block_sizes, _ = calculate_block_sizes(model_config,
                                                      serve_config,
                                                      vmem_limit_bytes)
        # logger.warning(f"No tuned parameters found for the given tuning keys: {decode_tuning_key}, using calculated decode tuned params {decode_tuned_params}")
    else:
        # logger.info(f"Found tuned parameters for decode tuning key: {decode_tuning_key}")
        decode_block_sizes = tuned_params_mapping[
            decode_tuning_key].to_block_sizes()
    if prefill_tuning_key not in tuned_params_mapping:
        from tpu_inference.kernels.experimental.batched_rpa.wrapper import \
            calculate_block_sizes
        _, prefill_block_sizes = calculate_block_sizes(model_config,
                                                       serve_config,
                                                       vmem_limit_bytes)
        # logger.warning(f"No tuned parameters found for the given tuning keys: {prefill_tuning_key}, using calculated prefill tuned params {prefill_tuned_params}")
    else:
        # logger.info(f"Found tuned parameters for prefill tuning key: {prefill_tuning_key}")
        prefill_block_sizes = tuned_params_mapping[
            prefill_tuning_key].to_block_sizes()
    return decode_block_sizes, prefill_block_sizes


tuned_params_mapping: dict[TuningKey, TunableParams] = {}
for log_entry in LOG_ENTRIES:
    model_config, serve_config, decode_tuned_block_size, prefill_tuned_block_size = log_entry.model, log_entry.serve, log_entry.decode_block_sizes, log_entry.prefill_block_sizes
    decode_tuning_key = TuningKey.from_config(model_config,
                                              serve_config,
                                              case='decode')
    prefill_tuning_key = TuningKey.from_config(model_config,
                                               serve_config,
                                               case='prefill')
    decode_tuned_params = TunableParams(**asdict(decode_tuned_block_size))
    prefill_tuned_params = TunableParams(**asdict(prefill_tuned_block_size))
    tuned_params_mapping[decode_tuning_key] = decode_tuned_params
    tuned_params_mapping[prefill_tuning_key] = prefill_tuned_params
logger.info(
    f"Loaded tuned parameters for {len(tuned_params_mapping)} tuning keys.")
