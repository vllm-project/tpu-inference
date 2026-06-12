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
''' How to use this kernel tuner:
1. Through BuildKite new build UI:
    1.1 Select the correct Commit and Branch
    1.2 Add below env variables in the "Environment Variables" section
        KERNEL_TUNING_KERNEL_NAME=batched_rpa_kernel_tuner
        KERNEL_TUNING_CASE_SET_ID=batched_rpa_kernel_YOUR_ID
        KERNEL_TUNING_RUN_ID=000
        KERNEL_TUNING_CASE_SET_DESC=YOUR_DESC
        KERNEL_TUNING_TPU_VERSION=tpu7x
        KERNEL_TUNING_TPU_CORES=2
2. Run Locally through local command line:
    python -m tools.kernel.tuner.v1.kernel_tuner_runner --run_locally \
        --kernel_tuner_name=batched_rpa_kernel_tuner \
        --case_set_desc=batched_rpa_kernel_tuning_setup \
        --case_set_id=batched_rpa_2 --run_id=0 \
        --tpu_version=tpu7x --tpu_cores=2
'''

import logging
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from vllm.utils.math_utils import cdiv

from tools.kernel.tuner.v1.batched_rpa_tuned_cases import ENTRIES
from tools.kernel.tuner.v1.common.kernel_tuner_base import (KernelTunerBase,
                                                            RunConfig,
                                                            TunerConfig,
                                                            TuningCase,
                                                            TuningStatus)
from tpu_inference.kernels.experimental.batched_rpa.tuned_params import (
    TunableParams, TuningKey)
from tpu_inference.kernels.experimental.batched_rpa.wrapper import \
    ragged_paged_attention
from tpu_inference.utils import get_dtype_packing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _generate_batched_rpa_inputs_prefill(tuning_key: TuningKey,
                                         rng: np.random.Generator
                                         | None = None):
    """Generates inputs for the batched RPA kernel Prefill case ONLY.

  Args:
    tuning_key: TuningKey object containing the configuration for the kernel.

  Returns: dictionary of inputs for the kernel, including:
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
  """
    if rng is None:
        rng = np.random.default_rng(1234)

    def gen_random(shape, dtype):
        return jnp.array(rng.random(size=shape,
                                    dtype=np.float32)).astype(dtype)

    num_q_heads = tuning_key.num_q_heads
    num_kv_heads = tuning_key.num_kv_heads
    head_dim = tuning_key.head_dim
    num_seqs = tuning_key.num_seqs
    page_size = tuning_key.page_size
    total_q_tokens = tuning_key.total_q_tokens
    num_page_indices = tuning_key.num_page_indices
    q_dtype = tuning_key.dtype_q
    kv_dtype = tuning_key.dtype_kv
    out_dtype = tuning_key.dtype_out
    q_scale = tuning_key.scale_q
    k_scale = tuning_key.scale_k
    v_scale = tuning_key.scale_v
    sliding_window = tuning_key.sliding_window

    kv_packing = get_dtype_packing(kv_dtype)

    queries = gen_random((total_q_tokens, num_q_heads, head_dim), q_dtype)
    keys = gen_random((total_q_tokens, num_kv_heads, head_dim), kv_dtype)
    values = gen_random((total_q_tokens, num_kv_heads, head_dim), kv_dtype)
    kv_cache = gen_random(
        (num_page_indices, page_size, cdiv(
            num_kv_heads * 2, kv_packing), kv_packing, head_dim), kv_dtype)

    max_input_len = 8192  # (TODO): This depends on the bench serve command line input-len flag
    max_prefill_seqs = min(num_seqs, total_q_tokens // max_input_len)
    remaining_tokens = total_q_tokens - max_prefill_seqs * max_input_len
    max_decode_seqs = min(num_seqs - max_prefill_seqs, remaining_tokens)
    assert max_decode_seqs == 0, "Only pack prefill sequences in prefill case, expect max_decode_seqs to be 0"

    pages_per_seq = num_page_indices // num_seqs
    kv_lens = [max_input_len
               ] * max_prefill_seqs + [0] * (num_seqs - max_prefill_seqs)

    cu_q_lens = [1] * (num_seqs + 1)
    cu_q_lens[0] = 0
    for i in range(1, max_prefill_seqs + 1):
        cu_q_lens[i] = cu_q_lens[i - 1] + max_input_len

    page_indices = []
    starting_page_index = 1
    for i in range(max_decode_seqs + max_prefill_seqs):
        num_pages_for_seq = cdiv(kv_lens[i], page_size)
        assert num_pages_for_seq <= pages_per_seq, f"num_pages_for_seq should not exceed pages_per_seq, but got num_pages_for_seq={num_pages_for_seq}, pages_per_seq={pages_per_seq}"
        page_indices.extend(
            list(
                range(starting_page_index, starting_page_index +
                      num_pages_for_seq)) + [0] *
            (pages_per_seq - num_pages_for_seq))
        starting_page_index += num_pages_for_seq
    assert len(
        page_indices
    ) <= num_page_indices, f"len(page_indices) should not exceed num_page_indices, but got len(page_indices)={len(page_indices)}, num_page_indices={num_page_indices}"
    page_indices.extend([0] * (num_page_indices - len(page_indices)))

    kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
    cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
    distribution = jnp.array([0, 0, max_prefill_seqs], dtype=jnp.int32)
    page_indices = jnp.array(page_indices, dtype=jnp.int32)

    return {
        'queries': queries,
        'keys': keys,
        'values': values,
        'kv_cache': kv_cache,
        'kv_lens': kv_lens,
        'page_indices': page_indices,
        'cu_q_lens': cu_q_lens,
        'distribution': distribution,
        'sm_scale': 1.0,  # (TODO) Value is from log
        'sliding_window': sliding_window,
        'soft_cap': None,  # (TODO) Value is from log
        'mask_value': -3.38953e+38,  # (TODO) Value is from log
        'q_scale': q_scale,
        'k_scale': k_scale,
        'v_scale': v_scale,
        'out_dtype': out_dtype,
        'use_causal_mask': True,  # Only support causal mask
        'update_kv_cache': True,  # (TODO) Value is from log
    }


class BatchedRpaKernelTuner(KernelTunerBase):

    def __init__(self, run_config: RunConfig):
        self.tuner_config = TunerConfig(
            tuning_key_class=TuningKey,
            tunable_params_class=TunableParams,
            kernel_tuner_name="batched_rpa_kernel_tuner",
            jit_kernel_pattern=r"(jit_ragged_paged_attention\()",
        )
        super().__init__(tuner_config=self.tuner_config, run_config=run_config)

    def generate_cases(self) -> list[TuningCase]:
        tuning_cases = []
        for log_entry in ENTRIES:
            model_config = log_entry.model
            serve_config = log_entry.serve
            prefill_tuned_block_size = log_entry.prefill_block_sizes
            prefill_tuning_key = TuningKey.from_config(model_config,
                                                       serve_config,
                                                       case='prefill')
            prefill_tunable_params = TunableParams(**asdict(
                prefill_tuned_block_size),
                                                   is_baseline=True)
            # We setup this for tuning Gemme4's rpa block sizes. When total_q_tokens is smaller than sequence input length,
            # it cannot be used for scheduling prefill case. We only tuned for prefill case at this moment.
            if serve_config.total_q_tokens < 8192:
                continue
            tuning_cases.append(
                TuningCase(tuning_key=prefill_tuning_key,
                           tunable_params=prefill_tunable_params))

            bq_c_sz = prefill_tunable_params.bq_c_sz
            bkv_sz = prefill_tunable_params.bkv_sz
            n_buffer = prefill_tunable_params.n_buffer

            for prefill_batch_size in [1, 2]:
                for bq_sz in range(512, 8193, 512):
                    for bq_c_sz in range(128, bq_sz + 1, 128):
                        if bq_sz % bq_c_sz != 0:
                            continue
                        for bkv_sz in range(512, 8193, 512):
                            if bkv_sz % prefill_tuning_key.page_size != 0:  # requirement from scheduler
                                continue
                            for n_buffer in [2, 3]:
                                tuning_cases.append(
                                    TuningCase(
                                        tuning_key=prefill_tuning_key,
                                        tunable_params=TunableParams(
                                            bq_sz=bq_sz,
                                            bq_c_sz=bq_c_sz,
                                            bkv_sz=bkv_sz,
                                            batch_size=prefill_batch_size,
                                            n_buffer=n_buffer)))

        logger.info(f"Generated {len(tuning_cases)} tuning cases.")
        return tuning_cases

    def generate_inputs(self, tuning_key: TuningKey):
        # Generate inputs for the kernel based on the tuning key.
        if tuning_key == self._tuning_key:
            return self._kernel_inputs_cache
        self._tuning_key = tuning_key
        self._kernel_inputs_cache = _generate_batched_rpa_inputs_prefill(
            tuning_key)

        return self._kernel_inputs_cache

    def run(self,
            tuning_key: TuningKey,
            tunable_params: TunableParams,
            iters: int = 1) -> tuple[TuningStatus, float, float]:
        if iters == 1:
            logger.info(
                f"Running batched RPA kernel for tuning key & tunable params:\nTuningKey=\n{tuning_key}, TunableParams=\n{tunable_params}"
            )
        input_cache = self.generate_inputs(tuning_key)
        prefill_block_sizes = tunable_params.to_block_sizes()
        try:
            start_ns = time.perf_counter_ns()
            for _ in range(iters):
                _, input_cache['kv_cache'] = jax.block_until_ready(
                    ragged_paged_attention(
                        queries=input_cache['queries'],
                        keys=input_cache['keys'],
                        values=input_cache['values'],
                        kv_cache=input_cache['kv_cache'],
                        kv_lens=input_cache['kv_lens'],
                        page_indices=input_cache['page_indices'],
                        cu_q_lens=input_cache['cu_q_lens'],
                        distribution=input_cache['distribution'],
                        sm_scale=input_cache['sm_scale'],
                        sliding_window=input_cache['sliding_window'],
                        soft_cap=input_cache['soft_cap'],
                        mask_value=input_cache['mask_value'],
                        q_scale=input_cache['q_scale'],
                        k_scale=input_cache['k_scale'],
                        v_scale=input_cache['v_scale'],
                        chunk_prefill_size=None,  # not used inside
                        decode_block_sizes=None,
                        prefill_block_sizes=prefill_block_sizes,
                        vmem_limit_bytes=
                        None,  # use default vmem limit from the wrapper
                        out_dtype=input_cache['out_dtype'],
                        use_causal_mask=input_cache['use_causal_mask'],
                        update_kv_cache=input_cache['update_kv_cache'],
                    ))
            end_ns = time.perf_counter_ns()
            latency_ns = (end_ns - start_ns)
            if iters > 1:
                logger.debug(
                    f"latency_ns={latency_ns}, average_latency_ns={latency_ns / iters}"
                )
            return TuningStatus.SUCCESS, latency_ns // iters, latency_ns  # status, average latency, total latency
        except Exception as err:
            if "RESOURCE_EXHAUSTED:" in str(err):
                logger.warning(
                    f"Kernel run failed with OOM for {tuning_key=}, {tunable_params=}"
                )
                return TuningStatus.FAILED_OOM, float("inf"), float("inf")
            logger.warning(
                f"Failed with {tuning_key=}, {tunable_params=}, got error: {err=}"
            )
            return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")
