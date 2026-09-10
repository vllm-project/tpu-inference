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

import itertools
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from vllm.utils.math_utils import cdiv

from tools.kernel.tuner.v1.common.kernel_tuner_base import KernelTunerBase
from tools.kernel.tuner.v1.common.tuner_datatypes import (RunConfig,
                                                          TunerConfig,
                                                          TuningCase,
                                                          TuningStatus)
from tpu_inference.kernels.experimental.batched_rpa.tuned_params import (
    TunableParams, TuningKey)
from tpu_inference.kernels.experimental.batched_rpa.wrapper import \
    ragged_paged_attention
from tpu_inference.utils import get_dtype_packing

logger = logging.getLogger(__name__)


def _get_page_indices(kv_lens, page_size, max_decode_seqs, max_prefill_seqs,
                      pages_per_seq, num_page_indices):
    # Total number of sequences being processed
    num_seqs = max_decode_seqs + max_prefill_seqs
    seq_kv_lens = kv_lens[:num_seqs]

    # 1. Vectorized cdiv equivalent: ceil(kv_lens / page_size)
    num_pages = (seq_kv_lens + page_size - 1) // page_size

    # 2. Create a boolean mask of shape (num_seqs, pages_per_seq)
    # True means a page index belongs there, False means it should be 0 padding
    col_indices = jnp.arange(pages_per_seq)
    mask = col_indices < num_pages[:, None]

    # 3. Flatten the mask into a 1D array of length (num_seqs * pages_per_seq)
    valid_slots = mask.flatten()

    # 4. Generate consecutive indices using cumsum
    # Example: valid_slots = [True, True, False, True]
    # cumsum -> [1, 2, 2, 3]
    consecutive_indices = jnp.cumsum(valid_slots)

    # 5. Mask out the invalid slots with 0
    # Result: [1, 2, 0, 3]
    page_indices_unpadded = jnp.where(valid_slots, consecutive_indices, 0)

    # 6. Pad out the rest of the array to num_page_indices
    current_length = page_indices_unpadded.shape[0]
    pad_amount = num_page_indices - current_length

    page_indices = jnp.pad(page_indices_unpadded, (0, pad_amount),
                           mode='constant',
                           constant_values=0)

    return page_indices


def _generate_batched_rpa_inputs(tuning_key: TuningKey,
                                 rng: np.random.Generator
                                 | None = None):
    """Generates inputs for the batched RPA kernel based on tuning_key.case.

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
    assert tuning_key.case in [
        'decode', 'prefill'
    ], f'Only support tuning for prefill and decode case but receives {tuning_key.case=}'
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

    # prefill and decode context length is from Gemma-4 1k/500 Case
    prefill_input_len = 1024 if total_q_tokens >= 1024 else total_q_tokens
    decode_kv_len = 1523

    # The 1524 here is because in Gemma-4 1k/500 case, prefill 1024 tokens and decode 500 tokens
    pages_per_seq = (1524 + page_size - 1) // page_size
    if tuning_key.case == 'prefill':
        assert total_q_tokens % prefill_input_len == 0, f'Expect prefill_input_len to align with total_q_tokens, got {prefill_input_len=} and {total_q_tokens=} '
        max_prefill_seqs = min(num_seqs, total_q_tokens // prefill_input_len)
        assert max_prefill_seqs * pages_per_seq <= num_page_indices, f'Expect max_prefill_seqs * pages_per_seq <= num_page_indices, got {total_q_tokens=}, {prefill_input_len=}, {num_seqs=} and {pages_per_seq=} and {num_page_indices=}'
        max_decode_seqs = 0
        kv_lens = jnp.pad(jnp.full((max_prefill_seqs, ),
                                   prefill_input_len,
                                   dtype=jnp.int32),
                          (0, num_seqs - max_prefill_seqs),
                          constant_values=0)
        cu_q_lens = jnp.pad(
            jnp.arange(max_prefill_seqs + 1) * prefill_input_len,
            (0, num_seqs - max_prefill_seqs),
            constant_values=max_prefill_seqs * prefill_input_len)
    else:
        max_prefill_seqs = 0
        max_decode_seqs = min(num_seqs, total_q_tokens)
        kv_lens = jnp.pad(jnp.full((max_decode_seqs, ),
                                   decode_kv_len,
                                   dtype=jnp.int32),
                          (0, num_seqs - max_decode_seqs),
                          constant_values=0)
        cu_q_lens = jnp.pad(jnp.arange(max_decode_seqs + 1),
                            (0, num_seqs - max_decode_seqs),
                            constant_values=max_decode_seqs)

    distribution = jnp.array(
        [max_decode_seqs, max_decode_seqs, max_decode_seqs + max_prefill_seqs],
        dtype=jnp.int32)
    page_indices = _get_page_indices(kv_lens, page_size, max_decode_seqs,
                                     max_prefill_seqs, pages_per_seq,
                                     num_page_indices)

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

    def __init__(self, run_config: RunConfig, lightweight: bool = False):
        self.tuner_config = TunerConfig(
            tuning_key_class=TuningKey,
            tunable_params_class=TunableParams,
            kernel_tuner_name="batched_rpa_kernel_tuner",
            support_bayesian_optimization=True,
            n_bayesian_trials=100,
            jit_kernel_pattern=lambda tuning_key: r"RPAm-"
            if tuning_key.case == 'prefill' else r"RPAd-",
        )
        super().__init__(tuner_config=self.tuner_config,
                         run_config=run_config,
                         lightweight=lightweight)

    def generate_cases(self) -> list[TuningCase]:
        from tools.kernel.tuner.v1.common.tuning_case_logger import \
            TuningCaseLogger

        current_dir = Path(__file__).parent
        tuning_case_logger = TuningCaseLogger(
            current_dir / 'tuning_cases/batched_rpa_gemma4_tuning_cases.json',
            key_class=TuningKey,
            params_class=TunableParams)
        # This is just a setup for tuning for Gemma4 Specific Prefill Case Only.
        # For tuning more cases, modify this and add more tuning cases to the
        # tuning_cases/batched_rpa_gemma4_tuning_cases.json file via using the TuningCaseLogger class.
        seen_keys: set[TuningKey] = set()
        unique_keys: list[TuningKey] = []
        cases: list[TuningCase] = []
        for case in tuning_case_logger.get_logged_tuning_cases():
            if (case.tuning_key not in seen_keys):
                seen_keys.add(case.tuning_key)
                unique_keys.append(case.tuning_key)
                cases.append(case)
        # Build the full Cartesian product of the search space for every key.
        for tuning_key in unique_keys:
            space = self.get_search_space(tuning_key)
            param_names = list(space.keys())
            for combo in itertools.product(*space.values()):
                cases.append(
                    TuningCase(tuning_key=tuning_key,
                               tunable_params=TunableParams(
                                   **dict(zip(param_names, combo)))))
        logger.info(f"Generated {len(cases)} tuning cases from log file.")
        return cases

    def get_search_space(self, tuning_key: TuningKey) -> dict[str, list]:
        """Returns independent lists of candidate values for each tunable param.

        For batched RPA, all bq_sz values are multiples of 256 and all bq_c_sz
        values are powers of 2 ≤ 128, so the constraint bq_sz % bq_c_sz == 0
        is always satisfied and no filtering is required for those two params.
        bkv_sz values are filtered to multiples of page_size as required by the
        scheduler.
        """
        bkv_sz_list = [
            v for v in range(256, 2049, 256) if v % tuning_key.page_size == 0
        ]
        return {
            'batch_size': [1, 2, 3, 4]
            if tuning_key.case == 'prefill' else [1, 2, 4, 8, 16, 32],
            'bq_sz':
            list(range(256, 2049, 256))
            if tuning_key.case == 'prefill' else [1],
            'bq_c_sz':
            [8, 16, 32, 64, 128] if tuning_key.case == 'prefill' else [1],
            'bkv_sz':
            bkv_sz_list,
            'n_buffer': [2, 3],
        }

    def generate_inputs(self, tuning_key: TuningKey):
        # Generate inputs for the kernel based on the tuning key.
        if tuning_key == self._tuning_key:
            return self._kernel_inputs_cache
        self._tuning_key = tuning_key
        self._kernel_inputs_cache = _generate_batched_rpa_inputs(tuning_key)

        return self._kernel_inputs_cache

    def run(self,
            tuning_key: TuningKey,
            tunable_params: TunableParams,
            iters: int = 1) -> tuple[TuningStatus, float, float]:
        input_cache = self.generate_inputs(tuning_key)
        block_sizes = tunable_params.to_block_sizes()
        try:
            start_ns = time.perf_counter_ns()
            for _ in range(iters):
                copied_inputs = jax.tree.map(
                    lambda x: x.copy()
                    if isinstance(x, jax.Array) else x, input_cache)
                queries = copied_inputs.pop('queries')
                keys = copied_inputs.pop('keys')
                values = copied_inputs.pop('values')
                kv_cache = copied_inputs.pop('kv_cache')
                kv_lens = copied_inputs.pop('kv_lens')
                page_indices = copied_inputs.pop('page_indices')
                cu_q_lens = copied_inputs.pop('cu_q_lens')
                distribution = copied_inputs.pop('distribution')

                input_cache['queries'], input_cache[
                    'kv_cache'] = jax.block_until_ready(
                        ragged_paged_attention(
                            queries,
                            keys,
                            values,
                            kv_cache,
                            kv_lens,
                            page_indices,
                            cu_q_lens,
                            distribution,
                            **copied_inputs,
                            chunk_prefill_size=None,  # not used inside
                            decode_block_sizes=block_sizes
                            if tuning_key.case == 'decode' else None,
                            prefill_block_sizes=block_sizes
                            if tuning_key.case == 'prefill' else None,
                            vmem_limit_bytes=
                            None,  # use default vmem limit from the wrapper
                        ))
            end_ns = time.perf_counter_ns()
            latency_ns = (end_ns - start_ns)
            return TuningStatus.SUCCESS, latency_ns // iters, latency_ns  # status, average latency, total latency
        except Exception as err:
            if "RESOURCE_EXHAUSTED:" in str(err):
                logger.info(
                    f"Kernel run failed with OOM for {tuning_key=}, {tunable_params=}"
                )
                return TuningStatus.FAILED_OOM, float("inf"), float("inf")
            logger.critical(
                f"Unknown exception happened for {tuning_key}, {tunable_params}, {iters=} got error: {err=}"
            )
            return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")
