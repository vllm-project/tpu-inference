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

import jax
import jax.numpy as jnp
import numpy as np
from absl import flags
from vllm.utils.math_utils import cdiv

from tools.kernel.tuner.v1.common.kernel_tuner_base import (KernelTunerBase,
                                                            RunConfig,
                                                            TunerConfig,
                                                            TuningCase,
                                                            TuningStatus)
from tpu_inference.kernels.mla.v2.kernel import mla_ragged_paged_attention
from tpu_inference.kernels.mla.v2.tuned_params import TunableParams, TuningKey
from tpu_inference.utils import align_to, get_dtype_packing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

flags.DEFINE_integer("mla_total_num_pages", 1506,
                     "Total number of pages in the cache.")
flags.DEFINE_integer("mla_page_size_per_kv_packing", 256,
                     "Page size per KV packing.")
flags.DEFINE_integer("mla_kv_packing", 4, "Packing factor for KV.")
flags.DEFINE_integer("mla_max_num_seqs", 160,
                     "Maximum number of sequences in the batch.")
flags.DEFINE_integer("mla_pages_per_seq", 9, "Number of pages per sequence.")
flags.DEFINE_integer("mla_actual_num_q_heads", 128,
                     "Actual number of Q heads.")
flags.DEFINE_integer("mla_actual_lkv_dim", 512, "Actual NOPE head dimension.")
flags.DEFINE_integer("mla_actual_r_dim", 64, "Actual ROPE head dimension.")
flags.DEFINE_string("mla_kv_dtype", "float8_e4m3fn", "KV cache data type.")
flags.DEFINE_string("mla_q_dtype", "float8_e4m3fn", "Q activation dtype.")


def _generate_mla_inputs(
    seq_lens,  # List[(q_len, kv_len)]
    num_heads,
    lkv_dim,
    r_dim,
    page_size,
    q_dtype,
    kv_dtype,
    num_pages,
    rng=None,
):
    """Generates inputs for the MLA kernel.

  Args:
    seq_lens: List of (q_len, kv_len) for each sequence.
    num_heads: Number of attention heads.
    lkv_dim: Dimension of the linear KV part.
    r_dim: Dimension of the rotary embedding part.
    page_size: Size of each page in the KV cache.
    q_dtype: Data type for queries.
    kv_dtype: Data type for keys and values.
    num_pages: Total number of pages in the cache.
    rng: Optional numpy random number generator.

  Returns:
    A tuple containing:
      - ql_nope: Query linear part without positional encoding.
      - q_pe: Query positional encoding part.
      - new_kv_c: New KV cache data.
      - new_k_pe: New Key positional encoding.
      - cache_kv: The existing KV cache.
      - kv_lens: Array of KV lengths for each sequence.
      - page_indices: Indices mapping sequence pages to cache pages.
      - cu_q_lens: Cumulative query lengths.
      - distribution: Mode distribution (e.g., prefill, decode, mixed).
  """
    if rng is None:
        rng = np.random.default_rng(1234)

    def gen_random(shape, dtype):
        return jnp.array(rng.random(size=shape,
                                    dtype=np.float32)).astype(dtype)

    padded_r_dim = align_to(r_dim, 128)
    padded_lkv_dim = align_to(lkv_dim, 128)
    padded_kv_dim = padded_lkv_dim + padded_r_dim
    packing = get_dtype_packing(kv_dtype)
    q_lens = [s[0] for s in seq_lens]
    kv_lens_list = [s[1] for s in seq_lens]
    total_q_len = sum(q_lens)
    cu_q_lens_list = [0]
    for q_len in q_lens:
        cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)

    max_kv_len = max(kv_lens_list) if kv_lens_list else 0
    pages_per_seq = cdiv(max_kv_len, page_size)

    page_indices_list = []
    page_count = 0
    for kv_len in kv_lens_list:
        num_seq_pages = cdiv(kv_len, page_size)
        indices = list(range(page_count, page_count + num_seq_pages))
        page_indices_list.extend(indices + [-1] *
                                 (pages_per_seq - num_seq_pages))
        page_count += num_seq_pages

    total_num_pages = max(num_pages, page_count)

    ql_nope = gen_random((num_heads, total_q_len, lkv_dim), q_dtype)
    q_pe = gen_random((total_q_len, num_heads, r_dim), q_dtype)
    new_kv_c = gen_random((total_q_len, lkv_dim), kv_dtype)
    new_k_pe = gen_random((total_q_len, r_dim), kv_dtype)

    assert page_size % packing == 0, f"page_size ({page_size}) must be a multiple of packing ({packing})"

    cache_kv = gen_random(
        (total_num_pages, page_size // packing, packing, padded_kv_dim),
        kv_dtype,
    )
    kv_lens = jnp.array(kv_lens_list, dtype=jnp.int32)
    page_indices = jnp.array(page_indices_list, dtype=jnp.int32)
    cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)

    # Find the number of decode sequences at the beginning of the batch.
    num_decode_seqs = 0
    for s in seq_lens:
        if s[0] == 1:
            num_decode_seqs += 1
        else:
            break
    distribution = jnp.array([num_decode_seqs, num_decode_seqs,
                              len(seq_lens)],
                             dtype=jnp.int32)

    return {
        'ql_nope': ql_nope,
        'q_pe': q_pe,
        'new_kv_c': new_kv_c,
        'new_k_pe': new_k_pe,
        'cache_kv': cache_kv,
        'kv_lens': kv_lens,
        'page_indices': page_indices,
        'cu_q_lens': cu_q_lens,
        'distribution': distribution,
    }


class MlaKernelTuner(KernelTunerBase):

    def __init__(self, run_config: RunConfig):
        self.tuner_config = TunerConfig(tuning_key_class=TuningKey,
                                        tunable_params_class=TunableParams,
                                        kernel_tuner_name="mla_kernel_tuner")
        super().__init__(tuner_config=self.tuner_config, run_config=run_config)

    def support_autotune(self) -> bool:
        return True

    def generate_cases(self) -> list[TuningCase]:
        tuning_set_from_log = []
        # The tuningKey and tunableParams in this list is constructed based on the real kernel execution logs during benchmarking.
        # The tunableParams currently represent the baseline configuration from the attention_interface.py.
        # From the log, only the max_num_tokens is different across different cases
        for max_num_tokens in [
                4, 8, 16, 32, 64, 128, 160, 256, 512, 1024, 2048
        ]:
            tuning_set_from_log.append([
                TuningKey(
                    max_num_tokens=max_num_tokens,
                    actual_num_q_heads=flags.FLAGS.mla_actual_num_q_heads,
                    actual_lkv_dim=flags.FLAGS.mla_actual_lkv_dim,
                    actual_r_dim=flags.FLAGS.mla_actual_r_dim,
                    kv_dtype=flags.FLAGS.mla_kv_dtype,
                    q_dtype=flags.FLAGS.mla_q_dtype,
                    total_num_pages=flags.FLAGS.mla_total_num_pages,
                    page_size_per_kv_packing=flags.FLAGS.
                    mla_page_size_per_kv_packing,
                    kv_packing=flags.FLAGS.mla_kv_packing,
                    max_num_seqs=flags.FLAGS.mla_max_num_seqs,
                    pages_per_seq=flags.FLAGS.mla_pages_per_seq,
                    s_dtype="bfloat16",
                    case="batched_decode",
                    soft_cap=None,
                    chunk_prefill_size=None,
                    sliding_window=None,
                    p_same_dtype_as_v=True,
                ),
                TunableParams(
                    decode_batch_size=4,
                    num_kv_pages_per_block=3,
                    num_queries_per_block=1,
                    vmem_limit_bytes=64 * 1024 * 1024,
                )
            ])

        tuning_cases = []
        for tuning_key, baseline_tunable_params in tuning_set_from_log:
            # for batched decode, the batch size is the same as the max num of sequences
            batch_size = tuning_key.max_num_seqs
            # add the baseline case from log for comparison
            tuning_cases.append(
                TuningCase(tuning_key=tuning_key,
                           tunable_params=baseline_tunable_params))

            # increase number of KV pages per block(flashatt processing block) by powers of two
            for num_kv_pages_per_block in [
                    2**j for j in range(tuning_key.pages_per_seq.bit_length())
            ] + ([3] if tuning_key.pages_per_seq >= 3 else []):
                decode_batch_size = 1
                while decode_batch_size <= batch_size:
                    tunable_params = TunableParams(
                        decode_batch_size=decode_batch_size,
                        num_kv_pages_per_block=
                        num_kv_pages_per_block,  # process one sequence (one block) at a time
                        num_queries_per_block=
                        1,  # for batched decode, process one query at a time to minimize latency
                        vmem_limit_bytes=60 * 1024 *
                        1024,  # set a medium vmem limit to allow some flexibility for the kernel to choose the best configuration
                    )
                    logger.debug(
                        f"Generated tuning case with tuning_key={tuning_key} and tunable_params={tunable_params}"
                    )
                    tuning_cases.append(
                        TuningCase(tuning_key=tuning_key,
                                   tunable_params=tunable_params))
                    decode_batch_size *= 2  # increase decode batch size by powers of two
        logger.info(f"Generated {len(tuning_cases)} tuning cases.")
        return tuning_cases

    def generate_inputs(self, tuning_key: TuningKey):
        # Generate inputs for the kernel based on the tuning key.
        if self._tuning_key and tuning_key == self._tuning_key:
            return self._kernel_inputs_cache
        self._tuning_key = tuning_key
        # recover kv_len from tuning_key
        kv_len = tuning_key.pages_per_seq * tuning_key.page_size_per_kv_packing * tuning_key.kv_packing
        rng = np.random.default_rng(0)
        self._kernel_inputs_cache = _generate_mla_inputs(
            # the q_len and kv_len in the seq_lens impact the kernel run time so this need to set correctly according to benchmark
            seq_lens=[[1, kv_len] for _ in range(tuning_key.max_num_seqs)],
            num_heads=tuning_key.actual_num_q_heads,
            lkv_dim=tuning_key.actual_lkv_dim,
            r_dim=tuning_key.actual_r_dim,
            page_size=tuning_key.page_size_per_kv_packing *
            tuning_key.kv_packing,
            q_dtype=jnp.dtype(tuning_key.q_dtype),
            kv_dtype=jnp.dtype(tuning_key.kv_dtype),
            num_pages=tuning_key.total_num_pages,
            rng=rng,
        )

        return self._kernel_inputs_cache

    def run(self,
            tuning_key: TuningKey,
            tunable_params: TunableParams,
            iters: int = 1) -> tuple[TuningStatus, float, float]:
        logger.debug(
            f"Running mla kernel with tuning_key={tuning_key}, tunable_params={tunable_params}, iters={iters}"
        )
        input_cache = self.generate_inputs(tuning_key)
        try:
            start_ns = time.perf_counter_ns()
            for _ in range(iters):
                _, input_cache['cache_kv'] = jax.block_until_ready(
                    mla_ragged_paged_attention(
                        ql_nope=input_cache['ql_nope'],
                        q_pe=input_cache['q_pe'],
                        new_kv_c=input_cache['new_kv_c'],
                        new_k_pe=input_cache['new_k_pe'],
                        cache_kv=input_cache['cache_kv'],
                        kv_lens=input_cache['kv_lens'],
                        page_indices=input_cache['page_indices'],
                        cu_q_lens=input_cache['cu_q_lens'],
                        distribution=input_cache['distribution'],
                        sliding_window=tuning_key.sliding_window,
                        soft_cap=tuning_key.soft_cap,
                        q_scale=None,
                        k_scale=None,
                        v_scale=None,
                        chunk_prefill_size=tuning_key.chunk_prefill_size,
                        s_dtype=tuning_key.s_dtype,
                        p_same_dtype_as_v=tuning_key.p_same_dtype_as_v,
                        decode_batch_size=tunable_params.decode_batch_size,
                        num_kv_pages_per_block=tunable_params.
                        num_kv_pages_per_block,
                        num_queries_per_block=tunable_params.
                        num_queries_per_block,
                        vmem_limit_bytes=tunable_params.vmem_limit_bytes,
                    ))
            end_ns = time.perf_counter_ns()
            latency_ns = (end_ns - start_ns)
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
