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
"""The fused expert-parallel kernel's build cache: one program per config."""

import inspect
import threading

from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fused_moe.v2.host import (WEIGHT_FORMAT_NAMES,
                                                     WeightFormat)
from tpu_inference.kernels.fused_moe.v2.kernel import (
    _BUILD_CACHE, build_fused_ep_moe_kernel)

# The build needs a VMEM bound, which a CPU cannot answer.
SERVED_CHIP = pltpu.ChipVersion.TPU_7X


def _restore_cache(saved):
    _BUILD_CACHE.clear()
    _BUILD_CACHE.update(saved)


class KernelBuildCacheTest(jtu.JaxTestCase, parameterized.TestCase):
    """Pallas keys its tracing cache on the kernel function's identity, so a
    second build of one configuration retraces every layer using it."""

    BASE_KWARGS = dict(g_local=1,
                       capacity=256,
                       hidden=512,
                       inter=512,
                       ep=8,
                       ragged_rows_alloc=512,
                       weight_format=WeightFormat.FP8,
                       rhs_qb=512)

    def setUp(self):
        super().setUp()
        info = pltpu.get_tpu_info_for_chip(SERVED_CHIP, 1)
        original = pltpu.get_tpu_info
        pltpu.get_tpu_info = lambda: info
        self.addCleanup(setattr, pltpu, "get_tpu_info", original)
        saved = dict(_BUILD_CACHE)
        self.addCleanup(_restore_cache, saved)

    def build(self, **overrides):
        kwargs = dict(self.BASE_KWARGS, **overrides)
        return build_fused_ep_moe_kernel(**kwargs)

    # yapf: disable
    @parameterized.named_parameters(
        dict(testcase_name="_one_key_returns_one_program",
             overrides=[{}, {}], programs=1),
        dict(testcase_name="_a_changed_shape_returns_a_different_program",
             overrides=[{}, dict(capacity=512)], programs=2),
        # The format decides the element type of every buffer and which
        # operands exist at all, so it cannot share a program with another.
        dict(testcase_name="_each_weight_format_gets_its_own_program",
             overrides=[dict(weight_format=f) for f in WEIGHT_FORMAT_NAMES],
             programs=len(WEIGHT_FORMAT_NAMES)),
        # Whether a bias is present decides the operand list and the
        # buffers, so it has to be in the key.
        dict(testcase_name="_each_bias_operand_gets_its_own_program",
             overrides=[{}, dict(has_w1_bias=True), dict(has_w2_bias=True),
                        dict(has_w1_bias=True, has_w2_bias=True)],
             programs=4),
    )
    # yapf: enable
    def test_the_key_decides_the_program(self, overrides, programs):
        """Everything that changes the program the builder emits is in the
        key, and nothing that does not is."""
        built = [self.build(**override) for override in overrides]
        self.assertLen({id(program) for program in built}, programs)

    @parameterized.parameters(*WEIGHT_FORMAT_NAMES)
    def test_every_weight_format_builds_a_program(self, weight_format):
        """Nothing in the tree had ever built an int8 or a bf16 program. The
        two formats appeared only in pure-JAX FFN bodies with no Pallas
        anywhere, in VMEM hand-arithmetic that calls the accounting and
        never the builder, and in adapter acceptance tests that stop at a
        reason string -- while the builder's own sweep pinned fp8.

        What that left unexercised is exactly what is unique to them: the
        HBM operand tuple drops operands out of its MIDDLE, three for bf16
        and one for int8, and the builder's operand arithmetic and the
        body's positional unpack have to agree on that for the four OUTPUT
        refs unpacked right after them to land on the right arrays. The
        geometry here divides the integer widening chunk, so all four are
        legal at one shape."""
        program = self.build(weight_format=weight_format)
        self.assertIsNotNone(program)
        self.assertIs(program, self.build(weight_format=weight_format))

    def test_no_environment_variable_reaches_the_build(self):
        """An environment read is a program switch outside the key's reach."""
        source = inspect.getsource(
            inspect.getmodule(build_fused_ep_moe_kernel))
        self.assertNotIn("os.environ", source)
        self.assertNotIn("getenv", source)

    def test_racing_builds_settle_on_one_program(self):
        _BUILD_CACHE.clear()
        start = threading.Barrier(8)
        built = []
        lock = threading.Lock()

        def race():
            start.wait()
            program = self.build()
            with lock:
                built.append(program)

        threads = [threading.Thread(target=race) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertLen(built, 8)
        self.assertLen({id(program) for program in built}, 1)
        self.assertLen(_BUILD_CACHE, 1)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
