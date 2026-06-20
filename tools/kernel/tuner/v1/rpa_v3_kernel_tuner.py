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
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

# yapf: disable
from tools.kernel.tuner.v1.common.kernel_tuner_base import (KernelTunerBase,
                                                            RunConfig,
                                                            RunResult,
                                                            TunerConfig,
                                                            TuningCase,
                                                            TuningStatus)
# yapf: enable
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    dynamic_validate_inputs, get_smem_estimate_bytes, get_vmem_estimate_bytes,
    ragged_paged_attention)

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

    def __init__(self, run_config: RunConfig):
        self.tuner_config = TunerConfig(
            tuning_key_class=TuningKey,
            tunable_params_class=TunableParams,
            kernel_tuner_name="rpa_v3_kernel_tuner")
        self.run_config = run_config

        super().__init__(tuner_config=self.tuner_config,
                         run_config=self.run_config)

        self.max_model_len = 384
        self.max_num_tokens = 384
        self.max_num_seqs = 128
        self.bkv_p_lst = [64, 128]
        self.bq_sz_lst = [128]
        self.page_size = 16
        self.q_dtype = jnp.bfloat16
        self.kv_dtype = jnp.bfloat16
        self.num_q_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 128
        # Default workload mirrors the v3 unit test (three chunked-prefill
        # sequences with kv_len > q_len). Each seq's KV occupies a unique
        # contiguous block of pages — see generate_inputs. The cache size
        # is sized in generate_inputs to fit.
        self.seq_lens = [(192, 328), (128, 180), (64, 255)]
        self.distribution_kind = "mixed"
        _pages_per_seq = cdiv(self.max_model_len, self.page_size)
        self.total_num_pages = max(1000, self.max_num_seqs * _pages_per_seq)
        self.sliding_window = None

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
        """Build kernel inputs that the eager reference can also score.

        Mirrors the construction in
        ``tests/kernels/ragged_paged_attention_kernel_v3_test.py``: each
        sequence's KV occupies a contiguous block of pages in ``kv_cache``,
        padded with NaN so the kernel and the reference both agree on
        "no-data-here". The legacy version of this method used
        ``np.random.rand`` for ``kv_cache`` and a random permutation for
        ``page_indices``; both pass through the latency loop but diverge
        from the reference's per-sequence write/read pattern, defeating
        smart-search's correctness gate.

        The default workload is the test's three-sequence chunked-prefill
        pattern. Override ``self.seq_lens`` (list of (q_len, kv_len)) and
        ``self.distribution_kind`` (\"mixed\" / \"decode_heavy\" /
        \"prefill_heavy\") before calling to use a different workload.
        """
        if self._tuning_key and tuning_key == self._tuning_key:
            return self._kernel_inputs_cache
        self._tuning_key = tuning_key

        seq_lens = getattr(self, "seq_lens", None)
        if seq_lens is None:
            seq_lens = [(192, 328), (128, 180), (64, 255)]
        cu_q_lens_raw = [0]
        kv_lens_raw = []
        for q_len, kv_len in seq_lens:
            assert q_len <= kv_len, (
                f"q_len ({q_len}) must be <= kv_len ({kv_len})")
            cu_q_lens_raw.append(cu_q_lens_raw[-1] + q_len)
            kv_lens_raw.append(kv_len)
        actual_num_seqs = len(seq_lens)
        # Pin max_num_tokens to the workload's total query length (aligned to
        # the TPU's lane-of-128). The kernel pads its output to this; ref
        # concatenates per-seq outputs (sum-of-q-lens). When they don't
        # match, ``assertAllClose`` fails on shape — the v3 unit test sets
        # ``max_num_batched_tokens = align_to(cu_q_lens[-1], 128)`` for the
        # same reason.
        total_q = cu_q_lens_raw[-1]
        self.max_num_tokens = max(align_to(total_q, 128),
                                  align_to(actual_num_seqs, 128))
        if actual_num_seqs > self.max_num_seqs:
            self.max_num_seqs = align_to(actual_num_seqs, 8)
        # Distribution: ``mixed`` means all seqs go through the m_block_sizes
        # kernel branch — the general-case path that the test exercises and
        # that smart-search by default tunes via the m_block_sizes tuple.
        dist_kind = getattr(self, "distribution_kind", "mixed")
        if dist_kind == "mixed":
            decode_end = 0
            prefill_end = 0
        elif dist_kind == "decode_heavy":
            decode_end = max(0, actual_num_seqs - 1)
            prefill_end = actual_num_seqs
        elif dist_kind == "prefill_heavy":
            decode_end = 0
            prefill_end = actual_num_seqs
        else:
            raise ValueError(f"unknown distribution_kind: {dist_kind!r}")

        (
            page_size,
            q_dtype_name,
            kv_dtype_name,
            num_q_heads,
            num_kv_heads,
            head_dim,
            max_model_len,
            _sliding_window,
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
        kv_packing = get_dtype_packing(kv_dtype)
        num_kv_heads_x2 = align_to(num_kv_heads * 2, kv_packing)
        padded_head_dim = align_to(head_dim, 128)
        self.pages_per_seq = cdiv(max_model_len, page_size)

        rng = np.random.default_rng(0)

        def _gen_random(shape, dtype):
            arr = rng.random(size=shape, dtype=np.float32)
            return jnp.asarray(arr, dtype=dtype)

        q = _gen_random((self.max_num_tokens, num_q_heads, head_dim), q_dtype)
        k = _gen_random((self.max_num_tokens, num_kv_heads, head_dim),
                        kv_dtype)
        v = _gen_random((self.max_num_tokens, num_kv_heads, head_dim),
                        kv_dtype)

        page_cnt = 0
        kv_pages_list = []
        page_indices_list = []
        for kv_len in kv_lens_raw:
            kv = _gen_random(
                (
                    kv_len,
                    num_kv_heads_x2 // kv_packing,
                    kv_packing,
                    padded_head_dim,
                ),
                kv_dtype,
            )
            pad_tokens = cdiv(kv_len, page_size) * page_size - kv_len
            kv = jnp.pad(
                kv,
                ((0, pad_tokens), (0, 0), (0, 0), (0, 0)),
                constant_values=jnp.nan,
            )
            kv = kv.reshape(
                -1,
                page_size,
                num_kv_heads_x2 // kv_packing,
                kv_packing,
                padded_head_dim,
            )
            num_pages_for_seq = kv.shape[0]
            indices = page_cnt + jnp.arange(num_pages_for_seq, dtype=jnp.int32)
            indices = jnp.pad(
                indices,
                ((0, self.pages_per_seq - num_pages_for_seq), ),
                constant_values=jnp.nan,
            )
            kv_pages_list.append(kv)
            page_indices_list.append(indices)
            page_cnt += num_pages_for_seq

        kv_cache = jnp.concatenate(kv_pages_list, axis=0)
        if kv_cache.shape[0] > self.total_num_pages:
            self.total_num_pages = int(kv_cache.shape[0]) + 16
        kv_cache = jnp.pad(
            kv_cache,
            (
                (0, self.total_num_pages - kv_cache.shape[0]),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
            ),
            constant_values=jnp.nan,
        )

        # Stack-pad-reshape matches the test's exact construction so the
        # kernel and reference see the same NaN-padded slot table.
        page_indices_stack = jnp.stack(page_indices_list, axis=0)
        page_indices_stack = jnp.pad(
            page_indices_stack,
            ((0, self.max_num_seqs - page_indices_stack.shape[0]), (0, 0)),
            constant_values=jnp.nan,
        )
        page_indices = page_indices_stack.reshape(-1)

        cu_q_lens = jnp.pad(
            jnp.array(cu_q_lens_raw, dtype=jnp.int32),
            (0, self.max_num_seqs + 1 - len(cu_q_lens_raw)),
        )
        kv_lens = jnp.pad(
            jnp.array(kv_lens_raw, dtype=jnp.int32),
            (0, self.max_num_seqs - len(kv_lens_raw)),
        )

        distribution = jnp.array([decode_end, prefill_end, actual_num_seqs],
                                 dtype=jnp.int32)
        logger.info(f"[Debug] {distribution=}")

        self._kernel_inputs_cache = {
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
        return self._kernel_inputs_cache

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

    # ------------------------------------------------------------------
    # Smart-search hooks (Phase 0 + 1). These are used when the runner is
    # invoked with ``--search-strategy={tpe,evolutionary}`` and bypass the
    # legacy grid path (``generate_cases``/``run`` above) entirely. The
    # search space is over a 4-tuple ``m_block_sizes`` matching the v3
    # kernel API (``bq_sz``, ``bkv_sz``, ``bq_csz``, ``bkv_csz``).
    # ------------------------------------------------------------------

    def get_default_tuning_key(self) -> TuningKey:
        sliding_window = (self.sliding_window[0] if isinstance(
            self.sliding_window, list) else self.sliding_window)
        return TuningKey(
            page_size=self.page_size,
            q_dtype=jnp.dtype(self.q_dtype).name,
            kv_dtype=jnp.dtype(self.kv_dtype).name,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_model_len=self.max_model_len,
            sliding_window=sliding_window,
        )

    def get_search_space(self):
        # Defer the import so that callers that only use the legacy grid
        # path don't have to import the smart-search package.
        from tools.kernel.tuner.v1.search.strategy import IntChoice
        return {
            "bq_sz": IntChoice("bq_sz", options=[16, 32, 64, 128]),
            "bkv_sz": IntChoice("bkv_sz", options=[128, 256, 512, 1024]),
            "bq_csz": IntChoice("bq_csz", options=[8, 16, 32, 64]),
            "bkv_csz": IntChoice("bkv_csz", options=[128, 256, 512, 1024]),
        }

    def get_oracle(self):
        from tools.kernel.tuner.v1.verifier.reference_oracle import RpaV3Oracle
        tk = self.get_default_tuning_key()
        return RpaV3Oracle(semantic_kwargs={
            "sliding_window": tk.sliding_window,
        })

    def get_cost_model(self):
        from tools.kernel.tuner.v1.bench.cost_estimate import (CostEstimate,
                                                               CostModel)

        def _estimate(tuning_key: TuningKey,
                      params: dict[str, Any]) -> CostEstimate:
            page_size = tuning_key.page_size
            bq_sz = params["bq_sz"]
            bkv_sz = params["bkv_sz"]
            bq_csz = params["bq_csz"]
            bkv_csz = params["bkv_csz"]
            # Kernel-level shape constraints (dynamic_validate_inputs). Catch
            # these before the TPU run so the search loop spends budget on
            # feasible candidates.
            if bq_csz > bq_sz:
                return CostEstimate(reason=f"bq_csz={bq_csz} > bq_sz={bq_sz}")
            if bkv_csz > bkv_sz:
                return CostEstimate(
                    reason=f"bkv_csz={bkv_csz} > bkv_sz={bkv_sz}")
            if bkv_sz % page_size != 0:
                return CostEstimate(
                    reason=
                    f"bkv_sz={bkv_sz} not multiple of page_size={page_size}")
            bkv_p = bkv_sz // page_size
            pages_per_seq = (tuning_key.max_model_len + page_size -
                             1) // page_size
            if bkv_p > pages_per_seq:
                return CostEstimate(
                    reason=
                    f"bkv_p={bkv_p} exceeds pages_per_seq={pages_per_seq}")
            try:
                vmem = get_vmem_estimate_bytes(
                    tuning_key.num_q_heads,
                    tuning_key.num_kv_heads,
                    tuning_key.head_dim,
                    bq_sz,
                    bkv_p,
                    tuning_key.q_dtype,
                    tuning_key.kv_dtype,
                )
            except Exception as err:  # pragma: no cover - estimator change
                logger.warning("VMEM estimator raised: %s", err)
                return CostEstimate(reason=f"vmem-estimator error: {err}")
            if vmem > VMEM_LIMIT_BYTES:
                return CostEstimate(
                    vmem_bytes=int(vmem),
                    reason=f"VMEM estimate {vmem} > limit {VMEM_LIMIT_BYTES}",
                )
            return CostEstimate(vmem_bytes=int(vmem))

        return CostModel(_estimate)

    def build_kernel_fn(
        self,
        tuning_key: TuningKey,
        params: dict[str, Any],
        inputs: dict[str, Any],
    ) -> Callable[[], Any]:
        # Pin the initial state in closure so each timed iteration starts
        # from the same kv_cache (the legacy ``run`` rebinds args[3] across
        # iters; smart-search wants stable inputs for verifier comparison).
        #
        # ``ragged_paged_attention`` v3 internally donates k/v/kv_cache via
        # pallas_call's input_output_aliases. After the first invocation the
        # original buffers are invalid, so calling fn() a second time fails
        # with "Donation requested for invalid buffer". ``jnp.copy`` forces
        # fresh device buffers so each iter is independent.
        q = inputs["q"]
        k = inputs["k"]
        v = inputs["v"]
        initial_kv_cache = inputs["kv_cache"]
        kv_lens = inputs["kv_lens"]
        page_indices = inputs["page_indices"]
        cu_q_lens = inputs["cu_q_lens"]
        distribution = inputs["distribution"]
        sliding_window = tuning_key.sliding_window
        m_block_sizes = (params["bq_sz"], params["bkv_sz"], params["bq_csz"],
                         params["bkv_csz"])

        def fn():
            return ragged_paged_attention(
                jnp.copy(q),
                jnp.copy(k),
                jnp.copy(v),
                jnp.copy(initial_kv_cache),
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                sliding_window=sliding_window,
                m_block_sizes=m_block_sizes,
                vmem_limit_bytes=VMEM_LIMIT_BYTES,
            )

        return fn

    def run_with_outputs(
        self,
        tuning_key: TuningKey,
        params: dict[str, Any],
        iters: int,
    ) -> RunResult:
        from tools.kernel.tuner.v1.bench.harness import measure

        try:
            inputs = self.generate_inputs(tuning_key)
        except Exception as err:
            logger.exception("generate_inputs failed")
            return RunResult(
                status=TuningStatus.UNKNOWN_ERROR,
                avg_latency_ns=float("inf"),
                total_latency_ns=float("inf"),
                aux={
                    "phase": "generate_inputs",
                    "error": str(err)
                },
            )
        try:
            fn = self.build_kernel_fn(tuning_key, params, inputs)
        except Exception as err:
            logger.exception("build_kernel_fn failed")
            return RunResult(
                status=TuningStatus.UNKNOWN_ERROR,
                avg_latency_ns=float("inf"),
                total_latency_ns=float("inf"),
                aux={
                    "phase": "build_kernel_fn",
                    "error": str(err)
                },
            )
        try:
            bench = measure(fn, warmup=2, iters=max(2, iters))
        except Exception as err:
            msg = str(err)
            status = (TuningStatus.FAILED_OOM if "RESOURCE_EXHAUSTED" in msg or
                      "OUT_OF_MEMORY" in msg else TuningStatus.UNKNOWN_ERROR)
            logger.info("run_with_outputs failed (%s): %s", status.value,
                        msg[:200])
            return RunResult(
                status=status,
                avg_latency_ns=float("inf"),
                total_latency_ns=float("inf"),
                aux={
                    "phase": "measure",
                    "error": msg
                },
            )
        return RunResult(
            status=TuningStatus.SUCCESS,
            avg_latency_ns=float(bench.mean_ns),
            total_latency_ns=float(bench.mean_ns * bench.iters),
            output=bench.output,
            bench=bench,
        )
