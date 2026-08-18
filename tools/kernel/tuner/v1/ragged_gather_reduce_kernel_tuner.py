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
"""Kernel tuner for ragged_gather_reduce.

How to use this kernel tuner locally:
    python -m tools.kernel.tuner.v1.kernel_tuner_runner --run_locally \
        --kernel_tuner_name=ragged_gather_reduce_kernel_tuner \
        --case_set_desc=ragged_gather_reduce_tuning \
        --case_set_id=ragged_gather_reduce_1 --run_id=0 \
        --tpu_version=tpu7x --tpu_cores=2
"""

import itertools
import logging
import time
from pathlib import Path

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from tools.kernel.tuner.v1.common.kernel_tuner_base import KernelTunerBase
from tools.kernel.tuner.v1.common.tuner_datatypes import (RunConfig,
                                                          TunerConfig,
                                                          TuningCase,
                                                          TuningStatus)
from tools.kernel.tuner.v1.common.tuning_case_logger import TuningCaseLogger
from tpu_inference.kernels.sparse_core.ragged_gather_reduce_tuned_params import (
    TunableParams, TuningKey)
from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import \
    ragged_gather_reduce

logger = logging.getLogger(__name__)


def _generate_ragged_gather_reduce_inputs(tuning_key: TuningKey,
                                           rng: np.random.Generator
                                           | None = None):
    """Generates synthetic input arrays for ragged_gather_reduce."""
    if rng is None:
        rng = np.random.default_rng(1234)

    out_size = tuning_key.input_size
    hidden_size = tuning_key.hidden_size
    reduce_group_size = tuning_key.reduce_group_size
    dtype = jnp.dtype(tuning_key.dtype)

    start = min(int(0.01 * out_size), out_size)
    end = min(int(0.1 * out_size), out_size)

    x = jnp.array(rng.standard_normal((out_size, hidden_size), dtype=np.float32)).astype(dtype)
    indices = jnp.array(rng.permutation(out_size), dtype=jnp.int32)
    topk_weights = jnp.array(rng.standard_normal((out_size, ), dtype=np.float32)).astype(jnp.bfloat16)
    valid_rows_mask = jnp.where(
        jnp.logical_and(
            jnp.array([start], jnp.int32) <= indices,
            indices < jnp.array([end], jnp.int32),
        ),
        True,
        False,
    )

    return {
        'x': x,
        'indices': indices,
        'topk_weights': topk_weights,
        'valid_rows_mask': valid_rows_mask,
        'reduce_group_size': reduce_group_size,
    }


class RaggedGatherReduceKernelTuner(KernelTunerBase):
    """Tuner for the SparseCore ragged_gather_reduce kernel."""

    def __init__(self, run_config: RunConfig, lightweight: bool = False):
        self.tuner_config = TunerConfig(
            tuning_key_class=TuningKey,
            tunable_params_class=TunableParams,
            kernel_tuner_name="ragged_gather_reduce_kernel_tuner",
            support_bayesian_optimization=True,
            n_bayesian_trials=100,
            jit_kernel_pattern=r"jit_ragged_gather_reduce\(",
        )
        super().__init__(tuner_config=self.tuner_config,
                         run_config=run_config,
                         lightweight=lightweight)

    def _generate_valid_tunable_params(
            self, tuning_key: TuningKey) -> list[TunableParams]:
        """Generates valid TunableParams configurations that satisfy all hard assertions."""
        num_simd_lanes = 16
        num_cores = 16

        hidden_size = tuning_key.hidden_size

        valid_params = []
        num_col_part_candidates = [
            c for c in [1, 2, 4, 8, 16] if num_cores % c == 0
        ]

        for num_col_part in num_col_part_candidates:
            num_row_part = num_cores // num_col_part
            # Assertion 1: num_row_partitions <= num_simd_lanes
            if num_row_part > num_simd_lanes:
                continue

            for num_row_subchunks in [1, 2, 3, 4, 8]:
                row_chunk_size = num_simd_lanes * num_row_subchunks

                base_aligned = pl.cdiv(hidden_size, 128 * num_col_part) * (
                    128 * num_col_part)
                aligned_candidates = [base_aligned]

                for aligned_hidden_size in aligned_candidates:
                    col_size = aligned_hidden_size // num_col_part
                    if col_size % 128 != 0:
                        continue

                    max_col_chunk = min(col_size, 1024)
                    col_chunk_candidates = []
                    for chunk in range(128, max_col_chunk + 1, 128):
                        if col_size % chunk == 0:
                            col_chunk_candidates.append(chunk)

                    for col_chunk_size in col_chunk_candidates:
                        vmem_needed_bytes = (
                            num_simd_lanes * col_chunk_size + col_size + 6 * row_chunk_size
                        ) * 4
                        if vmem_needed_bytes > 450 * 1024:
                            continue

                        valid_params.append(
                            TunableParams(
                                num_column_partitions=num_col_part,
                                num_row_partitions=num_row_part,
                                num_row_subchunks=num_row_subchunks,
                                row_chunk_size=row_chunk_size,
                                aligned_hidden_size=aligned_hidden_size,
                                col_size=col_size,
                                col_chunk_size=col_chunk_size,
                            ))
        return valid_params

    def generate_cases(self) -> list[TuningCase]:
        current_dir = Path(__file__).parent
        tuning_case_logger = TuningCaseLogger(
            current_dir / 'tuning_cases/ragged_gather_reduce_tuning_cases.json',
            key_class=TuningKey,
            params_class=TunableParams,
        )

        seen_keys: set[TuningKey] = set()
        unique_keys: list[TuningKey] = []
        cases: list[TuningCase] = []

        for case in tuning_case_logger.get_logged_tuning_cases():
            # if case.tuning_key not in seen_keys:
                # seen_keys.add(case.tuning_key)
            unique_keys.append(case.tuning_key)
            cases.append(case)

        # for tuning_key in unique_keys:
        #     valid_params = self._generate_valid_tunable_params(tuning_key)
        #     for tp in valid_params:
        #         cases.append(
        #             TuningCase(tuning_key=tuning_key, tunable_params=tp))

        logger.info(
            f"Generated {len(cases)} tuning cases for ragged_gather_reduce from log file.")
        return cases

    def get_search_space(self, tuning_key: TuningKey) -> dict[str, list]:
        valid_cases = self._generate_valid_tunable_params(tuning_key)
        if not valid_cases:
            return {}
        return {
            'num_column_partitions':
            sorted(list(set(tp.num_column_partitions for tp in valid_cases))),
            'num_row_partitions':
            sorted(list(set(tp.num_row_partitions for tp in valid_cases))),
            'num_row_subchunks':
            sorted(list(set(tp.num_row_subchunks for tp in valid_cases))),
            'row_chunk_size':
            sorted(list(set(tp.row_chunk_size for tp in valid_cases))),
            'aligned_hidden_size':
            sorted(list(set(tp.aligned_hidden_size for tp in valid_cases))),
            'col_size':
            sorted(list(set(tp.col_size for tp in valid_cases))),
            'col_chunk_size':
            sorted(list(set(tp.col_chunk_size for tp in valid_cases))),
        }

    def generate_inputs(self, tuning_key: TuningKey):
        if tuning_key == self._tuning_key and self._kernel_inputs_cache is not None:
            return self._kernel_inputs_cache
        inputs = _generate_ragged_gather_reduce_inputs(tuning_key)
        self._tuning_key = tuning_key
        self._kernel_inputs_cache = inputs
        return self._kernel_inputs_cache

    def run(self,
            tuning_key: TuningKey,
            tunable_params: TunableParams,
            iters: int = 1) -> tuple[TuningStatus, float, float]:
        input_cache = self.generate_inputs(tuning_key)
        try:
            if tunable_params.num_row_partitions > 16:
                return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")
            if tunable_params.col_size % tunable_params.col_chunk_size != 0:
                return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")

            start_ns = time.perf_counter_ns()
            for _ in range(iters):
                copied_inputs = jax.tree.map(
                    lambda item: item.copy()
                    if isinstance(item, jax.Array) else item, input_cache)
                out = ragged_gather_reduce(
                    x=copied_inputs['x'],
                    indices=copied_inputs['indices'],
                    topk_weights=copied_inputs['topk_weights'],
                    valid_rows_mask=copied_inputs['valid_rows_mask'],
                    reduce_group_size=copied_inputs['reduce_group_size'],
                    tunable_params=tunable_params,
                )
                out.block_until_ready()
            end_ns = time.perf_counter_ns()
            total_ns = end_ns - start_ns
            return TuningStatus.SUCCESS, total_ns // iters, total_ns
        except Exception as err:
            if "RESOURCE_EXHAUSTED" in str(err) or "OOM" in str(err):
                logger.info(
                    f"Kernel run failed with OOM for {tuning_key=}, {tunable_params=}"
                )
                return TuningStatus.FAILED_OOM, float("inf"), float("inf")
            logger.warning(
                f"Kernel run failed for {tuning_key=}, {tunable_params=}: {err}"
            )
            return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")
