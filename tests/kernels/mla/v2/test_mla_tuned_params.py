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

import pytest

from tpu_inference.kernels.mla.v2.tuned_params import TunableParams


@pytest.mark.parametrize(
    "lo,hi,expect_le,expect_ge",
    [
        # identical params → both comparisons are True
        (
            TunableParams(
                decode_batch_size=8,
                num_kv_pages_per_block=3,
                num_queries_per_block=1,
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
            TunableParams(
                decode_batch_size=8,
                num_kv_pages_per_block=3,
                num_queries_per_block=1,
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
            True,
            True,
        ),
        # lo is strictly smaller in the resource-demand dimensions,
        # while also having a larger vmem limit.
        (
            TunableParams(
                decode_batch_size=4,
                num_kv_pages_per_block=2,
                num_queries_per_block=1,
                vmem_limit_bytes=128 * 1024 * 1024,
            ),
            TunableParams(
                decode_batch_size=8,
                num_kv_pages_per_block=3,
                num_queries_per_block=2,
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
            True,
            False,
        ),
        # lo is strictly larger in the resource-demand dimensions,
        # while also having a smaller vmem limit.
        (
            TunableParams(
                decode_batch_size=16,
                num_kv_pages_per_block=4,
                num_queries_per_block=2,
                vmem_limit_bytes=32 * 1024 * 1024,
            ),
            TunableParams(
                decode_batch_size=8,
                num_kv_pages_per_block=3,
                num_queries_per_block=1,
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
            False,
            True,
        ),
        # mixed dimensions should not satisfy either ordering relation
        (
            TunableParams(
                decode_batch_size=8,
                num_kv_pages_per_block=4,
                num_queries_per_block=1,
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
            TunableParams(
                decode_batch_size=16,
                num_kv_pages_per_block=2,
                num_queries_per_block=2,
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
            False,
            False,
        ),
    ],
)
def test_tunable_params_ge_le(lo, hi, expect_le, expect_ge):
    assert (lo <= hi) is expect_le, f"Expected lo<=hi to be {expect_le}"
    assert (lo >= hi) is expect_ge, f"Expected lo>=hi to be {expect_ge}"


def test_tunable_params_ge_le_single_dim_difference():
    """A single larger dimension should make the larger object >= and the
    smaller object <=, because both operators require all dimensions to match
    the ordering relation."""

    base = TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=2,
        num_queries_per_block=1,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    larger_decode_batch = TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=2,
        num_queries_per_block=1,
        vmem_limit_bytes=64 * 1024 * 1024,
    )

    assert (base <= larger_decode_batch) is True
    assert (base >= larger_decode_batch) is False
    assert (larger_decode_batch >= base) is True
    assert (larger_decode_batch <= base) is False
