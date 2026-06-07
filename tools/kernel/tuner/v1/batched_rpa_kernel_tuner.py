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

import logging
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from vllm.utils.math_utils import cdiv


from tools.kernel.tuner.v1.utils import print_dataclasses_as_table
from tools.kernel.tuner.v1.common.kernel_tuner_base import (KernelTunerBase,
                                                            RunConfig,
                                                            TunerConfig,
                                                            TuningCase,
                                                            TuningStatus)
from tpu_inference.kernels.experimental.batched_rpa.configs_from_log import \
    LOG_ENTRIES
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
    """Generates inputs for the batched RPA kernel Prefill case.

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
    case = tuning_key.case

    kv_packing = get_dtype_packing(kv_dtype)

    queries = gen_random((total_q_tokens, num_q_heads, head_dim), q_dtype)
    keys = gen_random((total_q_tokens, num_kv_heads, head_dim), kv_dtype)
    values = gen_random((total_q_tokens, num_kv_heads, head_dim), kv_dtype)
    num_pages = 1296  # get this number from log
    kv_cache = gen_random(
        (num_pages, page_size, cdiv(
            num_kv_heads * 2, kv_packing), kv_packing, head_dim), kv_dtype)

    # distribution: [3]. Cumulative sum of number of decode, prefill, and mixed
    #     sequences. distribution[2] represents total number of sequences.

    # step 1: calculate how many prefill and decode can have seprately, fit prefill first
    # max input length is 1024, max number of input tokens is total_q_tokens, for prefill case, we want to pack as many prefill sequences as possible.
    #    if there are still space left, we add decode sequence until we reach the total_q_tokens limit. also need to make sure the total sequences not
    #    exceed num_seqs.
    max_input_len = 1024  # (TODO): We cannot get this number from the TuningKey, need to find a way to pass this information in. Currently this is from the vllm serve bench command line args
    max_prefill_seqs = min(num_seqs, total_q_tokens // max_input_len)
    remaining_tokens = total_q_tokens - max_prefill_seqs * max_input_len
    max_decode_seqs = min(
        num_seqs - max_prefill_seqs, remaining_tokens
    )  # assume decode sequence has 1 token input, this is the best case for decode heavy case
    # remove decode sequence first for debug
    assert max_decode_seqs == 0, f"For Prefill case, we only want prefill sequence, so max_decode_seqs should be 0, but got max_decode_seqs={max_decode_seqs}"

    # step 2: generate the distribution based on the step 1 result
    # for decode case, the kv has lens of max_input_len(actaully should be more, should be max_model_len or sliding_window), for prefill case, the kv lens should be 0.
    # max_model_len can be calculated from num_page_indices, num_page_indices = num_seqs * pages_per_seq, so
    pages_per_seq = num_page_indices // num_seqs
    max_model_len = pages_per_seq * page_size
    kv_lens = [max_input_len] * max_prefill_seqs + [0] * (
        num_seqs - max_prefill_seqs
    )  # for prefill sequence, the kv lens is max_input_len, ignore decode sequence for now
    logger.debug(f"{max_input_len=}, {num_seqs=}, {max_prefill_seqs=}")
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
    # tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py using MIX rather than Prefill for Prefill case
    distribution = jnp.array([0, 0, max_prefill_seqs], dtype=jnp.int32)
    page_indices = jnp.array(page_indices, dtype=jnp.int32)

    logger.debug(f'queries shape: {queries.shape}, dtype: {queries.dtype}')
    logger.debug(f'keys shape: {keys.shape}, dtype: {keys.dtype}')
    logger.debug(f'values shape: {values.shape}, dtype: {values.dtype}')
    logger.debug(f'kv_cache shape: {kv_cache.shape}, dtype: {kv_cache.dtype}')
    logger.debug(f'kv_lens shape: {kv_lens.shape}, {kv_lens[:32]=}')
    logger.debug(f'page_indices shape: {page_indices.shape}, {page_indices[:32]=}')
    logger.debug(f'cu_q_lens shape: {cu_q_lens.shape}, {cu_q_lens[:32]=}')
    logger.debug(f'distribution shape: {distribution.shape}, {distribution=}')
    return {
        'queries': queries,
        'keys': keys,
        'values': values,
        'kv_cache': kv_cache,
        'kv_lens': kv_lens,
        'page_indices': page_indices,
        'cu_q_lens': cu_q_lens,
        'distribution': distribution,
        'sm_scale':
        1.0,  # (TODO) value is from log, needs to confirm with others
        'sliding_window': sliding_window,
        'soft_cap':
        None,  # (TODO) value is from log, needs to confirm with others
        'mask_value':
        -3.38953e+38,  # (TODO) value is from log, needs to confirm with others
        'q_scale': q_scale,
        'k_scale': k_scale,
        'v_scale': v_scale,
        'out_dtype': out_dtype,
        'use_causal_mask': True,  # only support causal mask for now,
        'update_kv_cache': True,  # (TODO) Need to check from log
    }


class BatchedRpaKernelTuner(KernelTunerBase):

    def __init__(self, run_config: RunConfig):
        self.tuner_config = TunerConfig(
            tuning_key_class=TuningKey,
            tunable_params_class=TunableParams,
            kernel_tuner_name="batched_rpa_kernel_tuner")
        super().__init__(tuner_config=self.tuner_config, run_config=run_config)

    def generate_cases(self) -> list[TuningCase]:
        tuning_cases = []
        for log_entry in LOG_ENTRIES:
            model_config = log_entry.model
            serve_config = log_entry.serve
            decode_tuned_block_size = log_entry.decode_block_sizes
            prefill_tuned_block_size = log_entry.prefill_block_sizes
            decode_tuning_key = TuningKey.from_config(model_config,
                                                      serve_config,
                                                      case='decode')
            prefill_tuning_key = TuningKey.from_config(model_config,
                                                       serve_config,
                                                       case='prefill')
            decode_tunable_params = TunableParams(**asdict(
                decode_tuned_block_size),
                                                  is_baseline=True)
            prefill_tunable_params = TunableParams(**asdict(
                prefill_tuned_block_size),
                                                   is_baseline=True)
            if serve_config.total_q_tokens < 1024:  # batched_rpa_0
                continue
            tuning_cases.append(
                TuningCase(tuning_key=prefill_tuning_key,
                           tunable_params=prefill_tunable_params))

            bq_c_sz = prefill_tunable_params.bq_c_sz
            bkv_sz = prefill_tunable_params.bkv_sz
            batch_size = prefill_tunable_params.batch_size
            n_buffer = prefill_tunable_params.n_buffer
            total_q_tokens = serve_config.total_q_tokens

            for prefill_batch_size in [1, 2]:
                for bq_sz in range(8, 513, 8):
                    for bq_c_sz in range(bq_sz, bq_sz + 1, 8):
                        if bq_sz % bq_c_sz != 0:
                            continue
                        for bkv_sz in range(256, 1025, 256):
                            if bkv_sz % prefill_tuning_key.page_size != 0: # requirement from scheduler
                                continue
                            for n_buffer in [2, 3]: # when n_buffer is 1, it stucks at the second iteration.
                                tuning_cases.append(
                                    TuningCase(tuning_key=prefill_tuning_key,
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
                f"Running batched RPA kernel for tuning key & tunable params:\n"
            )
            print_dataclasses_as_table(tuning_key)
            print_dataclasses_as_table(tunable_params)
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
                logger.info(
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
