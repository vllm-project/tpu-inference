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

import itertools
import logging
import time

import jax
import jax.numpy as jnp

from tools.kernel.tuner.v1.common.kernel_tuner_base import (KernelTunerBase,
                                                            RunConfig,
                                                            TunerConfig,
                                                            TuningCase,
                                                            TuningStatus)
from tpu_inference.kernels.flash_attention.kernel import (BlockSizes,
                                                          SegmentIds,
                                                          flash_attention)
from tpu_inference.kernels.flash_attention.tuned_params import (TunableParams,
                                                                TuningKey)

logger = logging.getLogger(__name__)


class FlashAttentionKernelTuner(KernelTunerBase):

    def __init__(self, run_config: RunConfig, lightweight: bool = False):
        self.tuner_config = TunerConfig(
            tuning_key_class=TuningKey,
            tunable_params_class=TunableParams,
            kernel_tuner_name="flash_attention_kernel_tuner")
        self.run_config = run_config
        super().__init__(tuner_config=self.tuner_config,
                         run_config=self.run_config,
                         lightweight=lightweight)

    def generate_cases(self) -> list[TuningCase]:
        tuning_key = TuningKey(batch_size=8,
                               num_heads=16,
                               q_seq_len=2560,
                               kv_seq_len=2560,
                               head_dim=72)

        block_q_values = [32, 64, 128, 256, 512, 640]
        block_k_values = [128, 256, 512, 640, 1280, 2560]
        block_b_values = [1, 2, 4, 8]

        cases = []
        for bq, bk, bb in itertools.product(block_q_values, block_k_values,
                                            block_b_values):
            if bk % 128 != 0:
                continue
            for bkm in block_k_values:
                if bkm < bk:
                    continue
                if bkm % bk != 0:
                    continue
                if 2560 % bkm != 0:
                    continue
                if bk == 2560 and bkm != 2560:
                    continue
                tunable_params = TunableParams(block_q=bq,
                                               block_k_major=bkm,
                                               block_k=bk,
                                               block_b=bb)
                cases.append(TuningCase(tuning_key, tunable_params))
        return cases

    def generate_inputs(self, tuning_key: TuningKey):
        if self._tuning_key and tuning_key == self._tuning_key:
            return self._kernel_inputs_cache
        self._tuning_key = tuning_key

        q = jnp.ones((tuning_key.batch_size, tuning_key.num_heads,
                      tuning_key.q_seq_len, tuning_key.head_dim),
                     dtype=jnp.bfloat16)
        k = jnp.ones((tuning_key.batch_size, tuning_key.num_heads,
                      tuning_key.kv_seq_len, tuning_key.head_dim),
                     dtype=jnp.bfloat16)
        v = jnp.ones((tuning_key.batch_size, tuning_key.num_heads,
                      tuning_key.kv_seq_len, tuning_key.head_dim),
                     dtype=jnp.bfloat16)

        # segment_ids
        seg_q = jnp.zeros((tuning_key.batch_size, tuning_key.q_seq_len),
                          dtype=jnp.int32)
        seg_kv = jnp.zeros((tuning_key.batch_size, tuning_key.kv_seq_len),
                           dtype=jnp.int32)
        segment_ids = SegmentIds(q=seg_q, kv=seg_kv)

        self._kernel_inputs_cache = {
            'q': q,
            'k': k,
            'v': v,
            'segment_ids': segment_ids,
            'ab': None,
        }
        return self._kernel_inputs_cache

    def run(self,
            tuning_key: TuningKey,
            tunable_params: TunableParams,
            iters: int = 1) -> tuple[TuningStatus, float, float]:
        inputs = self.generate_inputs(tuning_key)

        block_sizes = BlockSizes(block_q=tunable_params.block_q,
                                 block_k_major=tunable_params.block_k_major,
                                 block_k=tunable_params.block_k,
                                 block_b=tunable_params.block_b)

        try:
            f = jax.jit(flash_attention,
                        static_argnames=['block_sizes', 'vmem_limit_bytes'])
            vmem_limit = 16 * 1024 * 1024

            # Warm up
            out = f(q=inputs['q'],
                    k=inputs['k'],
                    v=inputs['v'],
                    ab=inputs['ab'],
                    segment_ids=inputs['segment_ids'],
                    block_sizes=block_sizes,
                    vmem_limit_bytes=vmem_limit)
            out.block_until_ready()

            start_ns = time.perf_counter_ns()
            for _ in range(iters):
                out = f(q=inputs['q'],
                        k=inputs['k'],
                        v=inputs['v'],
                        ab=inputs['ab'],
                        segment_ids=inputs['segment_ids'],
                        block_sizes=block_sizes,
                        vmem_limit_bytes=vmem_limit)
                out.block_until_ready()
            end_ns = time.perf_counter_ns()

            latency_ns = (end_ns - start_ns)
            return TuningStatus.SUCCESS, latency_ns / iters, latency_ns
        except Exception as e:
            logger.warning(
                f"Kernel failed with block_q={tunable_params.block_q}, block_k_major={tunable_params.block_k_major}, block_k={tunable_params.block_k}, block_b={tunable_params.block_b}: {e}"
            )
            if "out of memory" in str(e).lower() or "vmem" in str(e).lower():
                return TuningStatus.FAILED_OOM, 0.0, 0.0
            return TuningStatus.UNKNOWN_ERROR, 0.0, 0.0
