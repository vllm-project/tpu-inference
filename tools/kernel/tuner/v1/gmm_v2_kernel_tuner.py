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
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tools.kernel.tuner.v1.common.kernel_tuner_base import KernelTunerBase
from tools.kernel.tuner.v1.common.tuner_datatypes import (RunConfig,
                                                          TunerConfig,
                                                          TuningCase,
                                                          TuningStatus)
from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class TuningKey:
    m: int
    k: int
    n: int
    cg: int
    ldt: str
    rdt: str
    q_block: int
    fuse_act: str | None = None


@dataclass(frozen=True)
class TunableParams:
    tm: int
    tk: int
    tn: int
    bucket_base: int

    def __ge__(self, other: "TunableParams") -> bool:
        return self.tm >= other.tm and self.tk >= other.tk and self.tn >= other.tn

    def __le__(self, other: "TunableParams") -> bool:
        return self.tm <= other.tm and self.tk <= other.tk and self.tn <= other.tn

    def to_tile_sizes(self) -> Any:
        from tpu_inference.kernels.megablox.gmm_v2 import TileSizes

        return TileSizes(tile_m=self.tm,
                         tile_k=self.tk,
                         tile_n=self.tn,
                         bucket_base=self.bucket_base)


class GmmV2KernelTuner(KernelTunerBase):

    def __init__(self, run_config: RunConfig, lightweight: bool = False):
        self.tuner_config = TunerConfig(
            tuning_key_class=TuningKey,
            tunable_params_class=TunableParams,
            kernel_tuner_name="gmm_v2_kernel_tuner",
            jit_kernel_pattern=r"^jit_gmm_v2",
        )
        super().__init__(tuner_config=self.tuner_config,
                         run_config=run_config,
                         lightweight=lightweight)

    def get_search_space(self, tuning_key: TuningKey) -> dict[str, list[int]]:
        tm_values = [t for t in [64, 128, 256] if tuning_key.m % t == 0]
        tk_values = [
            t for t in [128, 256, 512, 1024] if t <= tuning_key.k and (
                t % tuning_key.q_block == 0 or tuning_key.q_block % t == 0)
        ]
        tn_values = [t for t in [128, 256, 512, 1024] if t <= tuning_key.n]
        return {
            "tm": tm_values,
            "tk": tk_values,
            "tn": tn_values,
            "bucket_base": tm_values
        }

    def generate_cases(self) -> list[TuningCase]:
        problem_specs = [
            TuningKey(
                m=128,
                k=2560,
                n=6144,
                cg=
                20,  # current group size, which is the local expert counts. tg = 160, ep = 8, then cg = 20
                ldt="bfloat16",
                rdt="float8_e4m3fn",
                q_block=2560,
                fuse_act=None),
            TuningKey(m=256,
                      k=6144,
                      n=2560,
                      cg=20,
                      ldt="bfloat16",
                      rdt="float8_e4m3fn",
                      q_block=6144,
                      fuse_act="silu"),
        ]
        cases: list[TuningCase] = []
        for tuning_key in problem_specs:
            space = self.get_search_space(tuning_key)
            for tm, tk, tn, bucket_base in itertools.product(
                    space["tm"], space["tk"], space["tn"],
                    space['bucket_base']):
                if bucket_base > tm or tm % bucket_base != 0:
                    continue
                tunable_params = TunableParams(tm=tm,
                                               tk=tk,
                                               tn=tn,
                                               bucket_base=bucket_base)
                cases.append(
                    TuningCase(tuning_key=tuning_key,
                               tunable_params=tunable_params))
        logger.info("Generated %d tuning cases for GMM v2.", len(cases))
        return cases

    def generate_inputs(self, tuning_key: TuningKey):
        if self._tuning_key and tuning_key == self._tuning_key:
            return self._kernel_inputs_cache

        self._tuning_key = tuning_key
        rng = np.random.default_rng(1234)

        def _make_array(shape, dtype_name):
            values = rng.standard_normal(shape).astype(np.float32)
            return jnp.array(values).astype(jnp.dtype(dtype_name))

        lhs = _make_array((tuning_key.m, tuning_key.k), tuning_key.ldt)
        rhs = _make_array((tuning_key.cg, tuning_key.k, tuning_key.n),
                          tuning_key.rdt)
        rhs_scale = jnp.array(
            rng.uniform(size=(tuning_key.cg,
                              tuning_key.k // tuning_key.q_block, 1,
                              tuning_key.n)).astype(np.float32))
        group_sizes = jnp.full((tuning_key.cg, ),
                               tuning_key.m // tuning_key.cg,
                               dtype=jnp.int32)
        group_offset = jnp.array([0], dtype=jnp.int32)

        self._kernel_inputs_cache = {
            "lhs": lhs,
            "rhs": rhs,
            "rhs_scale": rhs_scale,
            "group_sizes": group_sizes,
            "group_offset": group_offset,
        }
        return self._kernel_inputs_cache

    def run(self,
            tuning_key: TuningKey,
            tunable_params: TunableParams,
            iters: int = 1) -> tuple[TuningStatus, float, float]:
        logger.info(
            "Running GMM v2 kernel with tuning_key=%s and tunable_params=%s",
            tuning_key,
            tunable_params,
        )
        try:
            inputs = self.generate_inputs(tuning_key)
            start_ns = time.perf_counter_ns()
            for _ in range(iters):
                _ = jax.block_until_ready(
                    gmm_v2(
                        lhs=inputs["lhs"],
                        rhs=inputs["rhs"],
                        group_sizes=inputs["group_sizes"],
                        rhs_scale=inputs["rhs_scale"],
                        group_offset=inputs["group_offset"],
                        tile_info=tunable_params.to_tile_sizes(),
                        preferred_element_type=inputs["lhs"].dtype,
                        maybe_quantize_lhs=False,
                        zero_initialize=True,
                        fuse_act=tuning_key.fuse_act,
                    ))
            end_ns = time.perf_counter_ns()
            avg_latency_ns = (end_ns - start_ns) // max(iters, 1)
            return TuningStatus.SUCCESS, avg_latency_ns, (end_ns - start_ns)
        except Exception as err:
            if "RESOURCE_EXHAUSTED:" in str(err):
                logger.warning(
                    f"Kernel run failed with OOM for {tuning_key=}, {tunable_params=}"
                )
                return TuningStatus.FAILED_OOM, float("inf"), float("inf")
            logger.warning(
                f"Failed with {tuning_key=}, {tunable_params=}, got error: {err=}"
            )
            raise Exception(
                f"Kernel run failed with tuning key & tunable params:\nTuningKey=\n{tuning_key}, TunableParams=\n{tunable_params}, got error: {err=}"
            )
