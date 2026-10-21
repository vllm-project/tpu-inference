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
"""flywheel_tpu backend wrapper against ref_ragged_paged_attention, the eager
reference of the RPA v3 tests, on one cache."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ref_ragged_paged_attention
from tpu_inference.layers.common import flywheel_attention as fa

jax.config.parse_flags_with_absl()

HEAD_DIM = 256


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class FlywheelTpuWrapperTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=6):
            self.skipTest("Expect TPUv6+")

    def run_case(self,
                 seq_lens: list[tuple[int, int]],
                 num_decode: int,
                 num_heads: tuple[int, int],
                 page_size: int,
                 *,
                 max_num_seqs: int = 8,
                 num_tokens: int = 1024,
                 should_check_cache_copies: bool = True,
                 is_dense: bool = False,
                 phase: str | None = None) -> None:
        """Run one step of seq_lens [(q_len, kv_len)], decode rows first, and
        compare output and updated cache against ref_ragged_paged_attention.

        is_dense routes the prefill through the dense kernel, which takes one
        fresh request as a (1, num_tokens) row. phase
        overrides the one the runner picks for these lengths, as a DP step
        does when its ranks disagree.
        """
        num_q_heads, num_kv_heads = num_heads
        rng = np.random.default_rng(1234)

        def _gen(shape):
            return jnp.asarray(rng.standard_normal(
                shape, dtype=np.float32)).astype(jnp.bfloat16)

        cu_q = [0]
        kv_lens = []
        for q_len, kv_len in seq_lens:
            assert q_len <= kv_len
            cu_q.append(cu_q[-1] + q_len)
            kv_lens.append(kv_len)
        assert cu_q[-1] <= num_tokens and len(seq_lens) <= max_num_seqs
        pages_per_seq = -(-max(kv_lens) // page_size)
        num_pages = max_num_seqs * pages_per_seq + 3

        q = _gen((num_tokens, num_q_heads, HEAD_DIM))
        k = _gen((num_tokens, num_kv_heads, HEAD_DIM))
        v = _gen((num_tokens, num_kv_heads, HEAD_DIM))
        kv_cache_v3 = _gen((num_pages, page_size, num_kv_heads, 2, HEAD_DIM))
        page_indices = jnp.asarray(
            rng.permutation(num_pages)[:max_num_seqs * pages_per_seq].astype(
                np.int32))
        cu_q_lens = jnp.asarray(
            np.pad(cu_q, (0, max_num_seqs + 1 - len(cu_q)),
                   mode="edge").astype(np.int32))
        kv_lens_arr = jnp.asarray(
            np.pad(kv_lens, (0, max_num_seqs - len(kv_lens))).astype(np.int32))
        distribution = jnp.array(
            [num_decode, num_decode, len(seq_lens)], jnp.int32)
        # Note: the runner decides the kernel, an index into the bucket's
        # step_kernels, and the writer's slice table on the host, so the test
        # has to do the same.
        if phase is None:
            phase = fa.select_attention_phase(num_decode, np.asarray(cu_q),
                                              np.asarray(kv_lens))
        sm_scale = HEAD_DIM**-0.5
        slice_table = np.zeros(
            (3, fa.kv_write_slice_bound(num_tokens, max_num_seqs, page_size)),
            np.int32)
        # Note: the decode kernel appends the leading decode rows itself,
        # so the writer's table holds only the rows after them.
        num_slices = fa.build_kv_write_slices(
            np.asarray(cu_q_lens)[num_decode:],
            np.asarray(kv_lens_arr)[num_decode:],
            np.asarray(page_indices).reshape(max_num_seqs,
                                             pages_per_seq)[num_decode:],
            page_size=page_size,
            out=slice_table)
        kv_write_slices = jnp.asarray(slice_table.reshape(-1))
        kv_write_num_slices = jnp.array([num_slices], jnp.int32)

        expected, expected_cache_v3 = ref_ragged_paged_attention(
            q,
            k,
            v,
            kv_cache_v3,
            kv_lens_arr,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale)

        kernel = phase
        if is_dense:
            assert num_decode == 0 and len(seq_lens) == 1
            assert seq_lens[0][0] == seq_lens[0][1]
            kernel = fa._DENSE_KERNEL
        elif phase == fa._PREFILL_PHASE:
            kernel = fa.varlen_block_span([[q_len for q_len, _ in seq_lens]],
                                          num_tokens)
        kernel_index = jnp.array([fa.step_kernels(num_tokens).index(kernel)],
                                 jnp.int32)
        static_kwargs = dict(sm_scale=sm_scale)
        run = jax.jit(fa.flywheel_tpu_attention,
                      static_argnames=tuple(static_kwargs),
                      donate_argnums=(3, ))
        # Note: the backend's [k0, v0, k1, v1, ...] head axis is the
        # v3 (pages, page_size, kv_heads, 2, head_dim) cache with its (K, V)
        # pair axis folded in, so a reshape converts either way.
        cache = kv_cache_v3.reshape(num_pages, page_size, 2 * num_kv_heads,
                                    HEAD_DIM)
        if should_check_cache_copies:
            hlo = run.lower(q, k, v, cache, kv_lens_arr, page_indices,
                            cu_q_lens, kv_write_slices, kv_write_num_slices,
                            distribution, kernel_index,
                            **static_kwargs).compile().as_text()
            cache_type = "bf16[" + ",".join(map(str, cache.shape)) + "]"
            copies = [
                line for line in hlo.splitlines()
                if " copy(" in line and cache_type in line
            ]
            self.assertEqual(copies, [], msg="\n".join(copies))
        pointer = cache.unsafe_buffer_pointer()
        output, updated = run(q, k, v, cache, kv_lens_arr, page_indices,
                              cu_q_lens, kv_write_slices, kv_write_num_slices,
                              distribution, kernel_index, **static_kwargs)
        self.assertEqual(updated.unsafe_buffer_pointer(), pointer)

        total = cu_q[-1]
        self.assertAllClose(output[:total].astype(jnp.float32),
                            expected[:total].astype(jnp.float32),
                            atol=5e-2,
                            rtol=5e-2)
        self.assertArraysEqual(updated.reshape(kv_cache_v3.shape),
                               expected_cache_v3)

    # Note: (8, 1) covers one KV head per chip, whose cache token row
    # is a single (K, V) row pair; (24, 4) and (16, 4) put four pairs in a row.
    @parameterized.product(num_heads=[(24, 4), (16, 4), (8, 1)],
                           page_size=[128, 256])
    def test_decode_only(self, num_heads, page_size):
        self.run_case([(1, 300), (1, 129), (1, 1), (1, 1000)], 4, num_heads,
                      page_size)

    @parameterized.product(num_heads=[(24, 4), (16, 4), (8, 1)],
                           page_size=[128, 256])
    def test_prefill_only(self, num_heads, page_size):
        self.run_case([(300, 300), (129, 129), (513, 513)], 0, num_heads,
                      page_size)

    def test_single_decode_row_is_batch_of_one(self):
        self.run_case([(1, 777)], 1, (24, 4), 128, num_tokens=16)

    @parameterized.named_parameters(
        ("span_256", [(100, 100), (37, 37), (128, 128), (64, 64)]),
        ("span_512", [(300, 300), (129, 129), (200, 200), (5, 5)]),
        # Note: the span bounds nothing, so a 1000-token request may
        # run on 512-token blocks.
        ("span_below_longest", [(1000, 1000), (24, 24)]),
    )
    def test_prefill_at_a_host_chosen_block_span(self, seq_lens):
        self.run_case(seq_lens, 0, (24, 4), 128)

    # Note: the request need not fill the bucket; the causal mask
    # keeps its queries off the padded tail.
    @parameterized.product(num_heads=[(24, 4), (16, 4)], seq_len=[1024, 700])
    def test_dense_prefill(self, num_heads, seq_len):
        self.run_case([(seq_len, seq_len)], 0, num_heads, 128, is_dense=True)

    @parameterized.named_parameters(
        ("decode", [(1, 300), (1, 129)], 2),
        ("prefill", [(250, 250), (300, 300)], 0),
    )
    def test_cache_is_updated_in_place_without_copies(self, seq_lens,
                                                      num_decode):
        self.run_case(seq_lens,
                      num_decode, (24, 4),
                      128,
                      should_check_cache_copies=True)

    # Note: a mixed step: the writer puts every scheduled row in the
    # pages, the decode kernel serves the leading one-token rows and the
    # extend kernel the chunk on its cached prefix, all on one donated cache.
    @parameterized.product(num_heads=[(24, 4), (8, 1)])
    def test_extend_with_decode_rows_and_a_cached_prefix(self, num_heads):
        self.run_case([(1, 300), (1, 129), (40, 700), (200, 457)],
                      2,
                      num_heads,
                      128,
                      should_check_cache_copies=True)

    # Note: one long prompt under chunked prefill: no decode row, but
    # its keys are partly in the pages, so the runner picks extend.
    @parameterized.product(num_heads=[(24, 4), (8, 1)])
    def test_chunk_on_a_cached_prefix_without_decode_rows(self, num_heads):
        self.run_case([(200, 457)], 0, num_heads, 128)

    # Note: with DP, a rank whose own batch is all decode runs the
    # step's extend executable whenever another rank prefills.
    @parameterized.product(num_heads=[(24, 4), (8, 1)])
    def test_decode_only_rank_on_the_extend_executable(self, num_heads):
        self.run_case([(1, 300), (1, 129)],
                      2,
                      num_heads,
                      128,
                      phase=fa._EXTEND_PHASE)


if __name__ == "__main__":
    absltest.main()
