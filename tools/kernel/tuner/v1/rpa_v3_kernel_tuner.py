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

import dataclasses
import itertools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from tools.kernel.tuner.v1.common.kernel_tuner_base import (KernelTunerBase,
                                                            TuningCase,
                                                            TuningStatus)
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    dynamic_validate_inputs, get_kv_cache_shape, get_smem_estimate_bytes,
    get_vmem_estimate_bytes, ragged_paged_attention)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def get_dtype_packing(dtype):
    return 32 // jax.dtypes.itemsize_bits(dtype)


def align_to(x, alignment):
    return cdiv(x, alignment) * alignment


def next_power_of_2(x):
    assert x > 0
    return 1 << (x - 1).bit_length()


def get_simplified_raw_key(
    page_size,
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    max_model_len,
    sliding_window,
):
    """Get the simplified key."""
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads_x2 = align_to(actual_num_kv_heads * 2, kv_packing)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head,
                                       q_packing)
    assert num_kv_heads_x2 % 2 == 0

    return (
        next_power_of_2(page_size),
        jnp.dtype(q_dtype).name,
        jnp.dtype(kv_dtype).name,
        next_power_of_2(num_q_heads_per_kv_head * actual_num_kv_heads),
        next_power_of_2(num_kv_heads_x2) // 2,
        align_to(head_dim, 128),
        next_power_of_2(max_model_len),
        sliding_window,
    )


# Temporarily set a large vmem limit for autotuning.
VMEM_LIMIT_BYTES = 60 * 1024 * 1024
SMEM_LIMIT_BYTES = 0.9 * 1024 * 1024
jax.config.parse_flags_with_absl()


def get_decode_heavy_example(max_num_tokens, max_model_len, actual_num_seqs):
    """Returns a decode-heavy example: N-1 decode sequences, 1 prefill sequence."""
    assert max_num_tokens >= actual_num_seqs
    decode_end = actual_num_seqs - 1
    if actual_num_seqs == 1:
        cu_q_lens = [0, max_num_tokens]
    else:
        cu_q_lens = list(range(actual_num_seqs))
        prefill_q_len = max_num_tokens - (actual_num_seqs - 1)
        cu_q_lens.append(cu_q_lens[-1] + prefill_q_len)
    kv_lens = []
    for i in range(actual_num_seqs):
        q_len = cu_q_lens[i + 1] - cu_q_lens[i]
        if q_len == 1:
            kv_lens.append(max_model_len)
        else:
            kv_lens.append(q_len)
    return cu_q_lens, kv_lens, decode_end


def get_prefill_heavy_example(max_num_tokens, max_model_len, actual_num_seqs):
    """Returns a prefill-heavy example: 1 decode sequence, N-1 prefill sequences."""
    assert max_num_tokens >= actual_num_seqs
    if actual_num_seqs == 1:
        decode_end = 0
        cu_q_lens = [0, max_num_tokens]
    else:
        decode_end = 1
        cu_q_lens = [0, 1]
        num_prefill_seqs = actual_num_seqs - 1
        tokens_for_prefill = max_num_tokens - 1
        q_len_per_seq = tokens_for_prefill // num_prefill_seqs
        r = tokens_for_prefill % num_prefill_seqs
        for i in range(num_prefill_seqs):
            q_len = q_len_per_seq + (1 if i < r else 0)
            cu_q_lens.append(cu_q_lens[-1] + q_len)
    kv_lens = []
    for i in range(actual_num_seqs):
        q_len = cu_q_lens[i + 1] - cu_q_lens[i]
        if q_len == 1:
            kv_lens.append(max_model_len)
        else:
            kv_lens.append(q_len)
    return cu_q_lens, kv_lens, decode_end


@dataclasses.dataclass
class TuningKey:
    page_size: int
    q_dtype: str
    kv_dtype: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    max_model_len: int
    sliding_window: int


@dataclasses.dataclass
class TunableParams:
    bkv_p: int
    bq_sz: int


class RpaV3KernelTuner(KernelTunerBase):
    # This is a reference implementation of a KernelTuner for testing purposes.
    # It defines a simple tuning key and tunable parameters, and simulates running
    # a kernel by sleeping for a random short duration. The latency returned is
    # not based on any real computation, but rather is just a placeholder to
    # demonstrate the tuning pipeline.

    def __init__(self, storage_manager):
        super().__init__(tuning_key_class=TuningKey,
                         tunable_params_class=TunableParams,
                         storage_manager=storage_manager,
                         job_bucket_size=100,
                         kernel_tuner_name="rpa_v3_kernel_tuner"
                         )  # Use a small bucket size for testing
        self.max_num_tokens = 128
        self.max_model_len = 2048
        self.max_num_seqs = 128
        self.bkv_p_lst = [64, 128]
        self.bq_sz_lst = [128]
        self.page_size = 128
        self.q_dtype = jnp.bfloat16
        self.kv_dtype = jnp.float8_e4m3fn
        self.num_q_heads = 8
        self.num_kv_heads = 4
        self.head_dim = 256
        self.total_num_pages = 1000
        self.sliding_window = [512, 1024]

    def generate_cases(self) -> list[TuningCase]:
        # tuning keys
        max_model_len = self.max_model_len if isinstance(
            self.max_model_len, list) else [self.max_model_len]
        sliding_window = self.sliding_window if isinstance(
            self.sliding_window, list) else [self.sliding_window]
        page_size = self.page_size if isinstance(self.page_size,
                                                 list) else [self.page_size]
        q_dtype = self.q_dtype if isinstance(self.q_dtype,
                                             list) else [self.q_dtype]
        kv_dtype = self.kv_dtype if isinstance(self.kv_dtype,
                                               list) else [self.kv_dtype]
        num_q_heads = self.num_q_heads if isinstance(
            self.num_q_heads, list) else [self.num_q_heads]
        num_kv_heads = self.num_kv_heads if isinstance(
            self.num_kv_heads, list) else [self.num_kv_heads]
        head_dim = self.head_dim if isinstance(self.head_dim,
                                               list) else [self.head_dim]
        # tunable parameters
        bkv_p_lst = self.bkv_p_lst if isinstance(self.bkv_p_lst,
                                                 list) else [self.bkv_p_lst]
        bq_sz_lst = self.bq_sz_lst if isinstance(self.bq_sz_lst,
                                                 list) else [self.bq_sz_lst]

        cases = []
        for page_size, q_dtype, kv_dtype, num_q_heads, num_kv_heads, head_dim, max_model_len, sliding_window, bkv_p, bq_sz in itertools.product(
                page_size, q_dtype, kv_dtype, num_q_heads, num_kv_heads,
                head_dim, max_model_len, sliding_window, bkv_p_lst, bq_sz_lst):

            tuning_key = TuningKey(
                page_size=page_size,
                q_dtype=jnp.dtype(q_dtype).name,
                kv_dtype=jnp.dtype(kv_dtype).name,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_model_len=max_model_len,
                sliding_window=sliding_window,
            )
            tunable_params = TunableParams(
                bkv_p=bkv_p,
                bq_sz=bq_sz,
            )
            (
                page_size,
                q_dtype_name,
                kv_dtype_name,
                num_q_heads,
                num_kv_heads,
                head_dim,
                max_model_len,
                sliding_window,
            ) = get_simplified_raw_key(
                tuning_key.page_size,
                tuning_key.q_dtype,
                tuning_key.kv_dtype,
                tuning_key.num_q_heads,
                tuning_key.num_kv_heads,
                tuning_key.head_dim,
                tuning_key.max_model_len,
                tuning_key.sliding_window,
            )
            pages_per_seq = cdiv(max_model_len, page_size)

            if bkv_p > pages_per_seq:
                logger.info(f"[Debug] Skip ({page_size=}, {bkv_p=}) because"
                            f" {bkv_p=} > {pages_per_seq=}")
                continue
            if page_size * bkv_p > 4096:
                logger.info(
                    f"[Debug] Skip because ({page_size=}) * ({bkv_p=}) ="
                    f" {page_size * bkv_p} > 4096")
                continue

            cases.append(TuningCase(tuning_key, tunable_params))
        return cases

    def generate_inputs(self, tuning_key: TuningKey):
        # Generate some mock inputs for the kernel based on the tuning key.
        if self._TUNING_KEY and tuning_key == self._TUNING_KEY:
            return self._KERNEL_INPUTS_CACHE
        self._TUNING_KEY = tuning_key

        cu_q_lens, kv_lens, decode_end = get_decode_heavy_example(
            self.max_num_tokens,
            self.max_model_len,
            actual_num_seqs=35,
        )

        (
            page_size,
            q_dtype_name,
            kv_dtype_name,
            num_q_heads,
            num_kv_heads,
            head_dim,
            max_model_len,
            sliding_window,
        ) = get_simplified_raw_key(
            tuning_key.page_size,
            tuning_key.q_dtype,
            tuning_key.kv_dtype,
            tuning_key.num_q_heads,
            tuning_key.num_kv_heads,
            tuning_key.head_dim,
            tuning_key.max_model_len,
            tuning_key.sliding_window,
        )
        q_dtype = jnp.dtype(q_dtype_name)
        kv_dtype = jnp.dtype(kv_dtype_name)
        self.pages_per_seq = cdiv(max_model_len, page_size)
        actual_num_seqs = len(kv_lens)
        cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
        kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
        cu_q_lens = jnp.pad(cu_q_lens,
                            (0, self.max_num_seqs + 1 - cu_q_lens.shape[0]))
        kv_lens = jnp.pad(kv_lens, (0, self.max_num_seqs - kv_lens.shape[0]))

        q_shape = (self.max_num_tokens, num_q_heads, head_dim)
        kv_shape = (self.max_num_tokens, num_kv_heads, head_dim)
        kv_cache_shape = get_kv_cache_shape(
            self.total_num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            kv_dtype,
        )
        q = jnp.array(
            np.random.rand(*q_shape),
            dtype=q_dtype,
        )
        k = jnp.array(
            np.random.rand(*kv_shape),
            dtype=kv_dtype,
        )
        v = jnp.array(
            np.random.rand(*kv_shape),
            dtype=kv_dtype,
        )
        kv_cache = jnp.array(
            np.random.rand(*kv_cache_shape),
            dtype=kv_dtype,
        )
        page_indices = np.random.randint(0,
                                         self.total_num_pages,
                                         size=(self.max_num_seqs *
                                               self.pages_per_seq, ),
                                         dtype=jnp.int32)

        distribution = jnp.array(
            [decode_end, actual_num_seqs, actual_num_seqs], dtype=jnp.int32)
        logger.info(f"[Debug] {distribution=}")

        self._KERNEL_INPUTS_CACHE = {
            "cu_q_lens": cu_q_lens,
            "kv_lens": kv_lens,
            "decode_end": decode_end,
            "q": q,
            "k": k,
            "v": v,
            "kv_cache": kv_cache,
            "page_indices": page_indices,
            "distribution": distribution,
        }
        return self._KERNEL_INPUTS_CACHE

    def run(self,
            tuning_key: TuningKey,
            tunable_params: TunableParams,
            iters: int = 1) -> tuple[TuningStatus, float, float]:
        # Run the kernel with the given tuning key and tunable params, and return the latency.
        logger.info(
            f"Running rpa_v3 kernel with tuning_key={tuning_key}, tunable_params={tunable_params}, iters={iters}"
        )
        (
            page_size,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = get_simplified_raw_key(
            tuning_key.page_size,
            tuning_key.q_dtype,
            tuning_key.kv_dtype,
            tuning_key.num_q_heads,
            tuning_key.num_kv_heads,
            tuning_key.head_dim,
            tuning_key.max_model_len,
            tuning_key.sliding_window,
        )
        inputs = self.generate_inputs(tuning_key)
        args = [
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["kv_cache"],
            inputs["kv_lens"],
            inputs["page_indices"],
            inputs["cu_q_lens"],
            inputs["distribution"],
        ]
        kwargs = {
            "sliding_window": tuning_key.sliding_window,
            "num_kv_pages_per_block": tunable_params.bkv_p,
            "num_queries_per_block": tunable_params.bq_sz,
            # Temporarily set a large vmem limit for autotuning.
            "vmem_limit_bytes": VMEM_LIMIT_BYTES,
        }

        try:
            dynamic_validate_inputs(*args, **kwargs)
        except Exception as err:
            logger.info(
                f"[Debug] Failed with ({page_size=}, {tunable_params.bkv_p=},"
                f" {tunable_params.bq_sz=}), got error: {err=}")
            return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")

        vmem_estimate = get_vmem_estimate_bytes(
            tuning_key.num_q_heads,
            tuning_key.num_kv_heads,
            tuning_key.head_dim,
            tunable_params.bq_sz,
            tunable_params.bkv_p,
            tuning_key.q_dtype,
            tuning_key.kv_dtype,
        )
        if vmem_estimate > VMEM_LIMIT_BYTES:
            logger.info(f"[Debug] Skip ({page_size=}, {tunable_params.bkv_p=},"
                        f" {tunable_params.bq_sz=}) because {vmem_estimate=} >"
                        f" {VMEM_LIMIT_BYTES=}")
            return TuningStatus.SKIPPED, float("inf"), float("inf")

        smem_estimate = get_smem_estimate_bytes(
            self.max_num_seqs,
            self.pages_per_seq,
        )
        if smem_estimate > SMEM_LIMIT_BYTES:
            logger.info(f"[Debug] Skip ({page_size=}, {tunable_params.bkv_p=},"
                        f" {tunable_params.bq_sz=}) because {smem_estimate=} >"
                        f" {SMEM_LIMIT_BYTES=}")
            return TuningStatus.SKIPPED, float("inf"), float("inf")

        try:
            start_ns = time.perf_counter_ns()
            for i in range(iters):
                _, args[3] = jax.block_until_ready(
                    ragged_paged_attention(*args, **kwargs))

            end_ns = time.perf_counter_ns()
            latency_ns = (end_ns - start_ns)
            return TuningStatus.SUCCESS, latency_ns / iters, latency_ns  # status, average latency, total latency
        except Exception as err:
            logger.info(
                f"[Debug] Failed with ({page_size=}, {tunable_params.bkv_p=},"
                f" {tunable_params.bq_sz=}), got error: {err=}")
            return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")
