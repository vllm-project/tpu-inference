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
"""PCP ring cache phase on the batched RPA kernel.

The ring streams page-interleaved KV cache shards around the pcp mesh axis
while every rank attends with its local Q, accumulating all rounds in one
online softmax. Since softmax is order-invariant, the reference is plain
CACHE_ONLY attention over the full cache laid out in token order.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as PS

from tpu_inference.kernels.experimental.batched_rpa import \
    configs as brpa_configs
from tpu_inference.kernels.experimental.batched_rpa.utils import (
    align_to, cp_local_cache_len, get_dtype_packing)
from tpu_inference.kernels.experimental.batched_rpa.wrapper import \
    ragged_paged_attention


def cdiv(a, b):
    return (a + b - 1) // b


def merge_kv(k, v):
    """[n, nkv, hd] x2 -> [n, nkv_x2 // packing, packing, padded_hd]."""
    n, nkv, hd = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    nkv_x2 = nkv * 2
    nkv_x2_aligned = align_to(nkv_x2, kv_packing)
    padded_hd = align_to(hd, 128)
    kv = jnp.pad(
        jnp.concat([k, v], axis=-1).reshape(n, nkv_x2, hd),
        ((0, 0), (0, nkv_x2_aligned - nkv_x2), (0, padded_hd - hd)),
        constant_values=0,
    ).reshape(n, nkv_x2_aligned // kv_packing, kv_packing, padded_hd)
    return kv


class BatchedRpaPcpRingTest(jtu.JaxTestCase):

    @parameterized.product(
        dtype=[jnp.float32, jnp.bfloat16],
        P=[2, 4, 8],
        layout=["head", "seq"],
    )
    def test_pcp_ring_cache_phase_matches_full_cache(self, dtype, P, layout):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        if jax.device_count() < P:
            self.skipTest(f"Needs {P} devices")

        lp = 128  # local (per-rank) page size
        nq, nkv, hd = 8, 2, 128
        q_len = 256
        # Two full super-pages plus a 200-token remainder so rank shard
        # lengths differ (rank 0 gets 128 of it, rank 1 gets 72, rest 0).
        prev_len = 2 * P * lp + 200
        kv_len = prev_len + q_len
        max_num_seqs = 8
        pages_per_seq = cdiv(prev_len, lp)
        num_pages = pages_per_seq + 8
        sm = hd**-0.5

        rng = np.random.default_rng(1234)

        def gen(shape):
            return jnp.array(rng.random(size=shape,
                                        dtype=np.float32)).astype(dtype)

        q_all = gen((P, q_len, nq, hd))
        k_prev = gen((prev_len, nkv, hd))
        v_prev = gen((prev_len, nkv, hd))
        kv_layout = (brpa_configs.KVLayout.SEQ_ALONG_LANE if layout == "seq"
                     else brpa_configs.KVLayout.HEAD_ALONG_SUBLANE)
        if layout == "seq":
            # Token-major [n, nkv_x2, ahd // packing, packing]; the cache
            # builder moves tokens to the lane (last) dim per page.
            kv_packing = get_dtype_packing(dtype)
            ahd = align_to(hd, 8 * kv_packing)
            merged = jnp.pad(
                jnp.concat([k_prev, v_prev],
                           axis=-1).reshape(prev_len, nkv * 2, hd),
                ((0, 0), (0, 0), (0, ahd - hd)),
            ).reshape(prev_len, nkv * 2, ahd // kv_packing, kv_packing)
        else:
            merged = merge_kv(k_prev, v_prev)
        dummy_kv = jnp.zeros((q_len, nkv, hd), dtype)

        # Small blocks so both the lane serialization (2 ring micro-steps per
        # kernel step) and the multi-block ring (2 blocks of 2 pages) are
        # exercised.
        blocks = brpa_configs.BlockSizes(bq_sz=128,
                                         bq_c_sz=128,
                                         bkv_sz=256,
                                         batch_size=2,
                                         n_buffer=3)

        kv_lens = jnp.zeros(max_num_seqs, jnp.int32).at[0].set(kv_len)
        cu_q_lens = jnp.zeros(max_num_seqs + 1, jnp.int32).at[1:].set(q_len)
        distribution = jnp.array([0, 0, 1], jnp.int32)
        page_indices = jnp.zeros(max_num_seqs * pages_per_seq,
                                 jnp.int32).at[:pages_per_seq].set(
                                     jnp.arange(pages_per_seq,
                                                dtype=jnp.int32))

        def cache_from_tokens(tokens):
            """Token-major page content -> (num_pages, ...) cache."""
            n = tokens.shape[0]
            npg = max(cdiv(n, lp), 1)
            padded = jnp.pad(tokens,
                             ((0, npg * lp - n), (0, 0), (0, 0), (0, 0)))
            paged = padded.reshape(npg, lp, *tokens.shape[1:])
            if layout == "seq":
                # (npg, lp, planes, hd / p, p) -> (npg, planes, hd / p, p, lp)
                paged = paged.transpose(0, 2, 3, 4, 1)
            cache = jnp.zeros((num_pages, *paged.shape[1:]), dtype)
            return cache.at[:npg].set(paged)

        common = dict(
            sm_scale=sm,
            attention_scope=brpa_configs.AttentionScope.CACHE_ONLY,
            update_kv_cache=False,
            return_lse=True,
            decode_block_sizes=blocks,
            prefill_block_sizes=blocks,
            kv_layout=kv_layout,
        )

        # Reference: full cache in token order, no CP. kv_cache is donated,
        # so each call gets its own copy.
        ref_outs, ref_lses = [], []
        for r in range(P):
            o, _, lse = ragged_paged_attention(
                q_all[r],
                dummy_kv,
                dummy_kv,
                cache_from_tokens(merged),
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                **common,
            )
            ref_outs.append(np.asarray(o[:q_len], np.float32))
            ref_lses.append(np.asarray(lse[:q_len], np.float32))
        ref_out = np.stack(ref_outs)
        ref_lse = np.stack(ref_lses)

        # Page-interleaved shards: rank r's local slot j holds global token
        # (j // lp) * P * lp + r * lp + j % lp, matching cp_local_cache_len.
        isz = lp

        def rank_tokens(r):
            slices = []
            for j in range(cdiv(prev_len, P * isz)):
                start = (j * P + r) * isz
                end = min(start + isz, prev_len)
                if start < prev_len:
                    slices.append(merged[start:end])
            tokens = (jnp.concatenate(slices, axis=0)
                      if slices else merged[:0])
            expected = cp_local_cache_len(jnp.int32(prev_len), P, r, isz)
            assert tokens.shape[0] == int(expected), (tokens.shape[0],
                                                      int(expected))
            return tokens

        cache_sh = jnp.stack(
            [cache_from_tokens(rank_tokens(r)) for r in range(P)])

        mesh = Mesh(np.array(jax.devices()[:P]), ("pcp", ))

        @functools.partial(
            shard_map,
            mesh=mesh,
            in_specs=(PS("pcp"), PS("pcp")),
            out_specs=(PS("pcp"), PS("pcp")),
            check_rep=False,
        )
        def ring(q_l, c_l):
            r = jax.lax.axis_index("pcp")
            o, _, lse = ragged_paged_attention(
                q_l[0],
                dummy_kv,
                dummy_kv,
                c_l[0],
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                cp_rank=jax.lax.reshape(r, (1, )).astype(jnp.int32),
                cp_group_size=P,
                pcp_ring_axis_name="pcp",
                pcp_ring_mesh_axis_names=("pcp", ),
                **common,
            )
            return o[None], lse[None]

        out, lse = jax.jit(ring)(q_all, cache_sh)
        out = np.asarray(out[:, :q_len], np.float32)
        lse = np.asarray(lse[:, :q_len], np.float32)

        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreater(float(np.abs(out).max()), 0.0)
        tol = 2e-3 if dtype == jnp.float32 else 2e-2
        self.assertAllClose(out, ref_out, atol=tol, rtol=tol)
        self.assertAllClose(lse, ref_lse, atol=tol, rtol=tol)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
