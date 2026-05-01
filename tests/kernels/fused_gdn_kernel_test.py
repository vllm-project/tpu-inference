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
"""Correctness tests for fused GDN kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.gdn import fused_gdn
from tpu_inference.layers.common.ragged_gated_delta_rule_ref import \
    ragged_gated_delta_rule as ragged_gated_delta_rule_ref

jax.config.parse_flags_with_absl()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    rng,
    decode_N,
    mixed_seqlens,
    H_qk,
    H_v,
    K,
    V,
    dtype=jnp.bfloat16,
    max_num_req=None,
):
    """Build inputs for fused_gdn tests."""
    all_seqlens = [1] * decode_N + list(mixed_seqlens)
    N = len(all_seqlens)
    T = sum(all_seqlens)
    cu_seqlens = np.cumsum([0] + all_seqlens).astype(np.int32)

    if max_num_req is not None:
        padded_cu = np.full(max_num_req + 1, T, dtype=np.int32)
        padded_cu[:len(cu_seqlens)] = cu_seqlens
        cu_seqlens = padded_cu

    q = rng.randn(T, H_qk, K).astype(np.float32)
    k = rng.randn(T, H_qk, K).astype(np.float32)
    v = rng.randn(T, H_v, V).astype(np.float32)
    a = rng.randn(T, H_v).astype(np.float32)
    b = rng.randn(T, H_v).astype(np.float32)
    A_log = rng.randn(H_v).astype(np.float32)

    h0_N = max_num_req if max_num_req is not None else N
    h0 = rng.randn(h0_N, H_v, K, V).astype(np.float32)
    state_indices = np.arange(h0_N, dtype=np.int32)

    if dtype != np.float32:
        q, k, v, a, b = (jnp.array(x, dtype=dtype) for x in [q, k, v, a, b])
    else:
        q, k, v, a, b = (jnp.array(x) for x in [q, k, v, a, b])

    return (
        q,
        k,
        v,
        a,
        b,
        jnp.array(A_log),
        jnp.array(h0),
        jnp.array(cu_seqlens),
        jnp.array(state_indices),
        N,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class FusedGdnKernelTest(jtu.JaxTestCase):

    def _test_fused_gdn(
        self,
        decode_N,
        mixed_seqlens,
        H_qk,
        H_v,
        K,
        V,
        *,
        max_num_req=None,
        use_dt_bias=False,
        lower_bound=None,
        atol=1e-2,
        has_initial_state_active=None,
    ):
        rng = np.random.RandomState(42)
        q, k, v, a, b, A_log, h0, cu_seqlens, state_indices, N = _make_inputs(
            rng,
            decode_N,
            mixed_seqlens,
            H_qk,
            H_v,
            K,
            V,
            max_num_req=max_num_req,
        )
        T = q.shape[0]

        dt_bias = (jnp.array(rng.randn(H_v).astype(np.float32))
                   if use_dt_bias else jnp.zeros(H_v, dtype=jnp.float32))

        # `has_initial_state_active` is a python list of bools for the
        # ACTIVE sequences only (length N); we pad to `max_num_req` with
        # True for the padded slots (their state isn't read by either
        # impl, but the array shape must match `state_indices`). Default
        # behaviour (None) is all-True, which matches the pre-fix path
        # and keeps the existing baseline tests valid.
        max_num_req_padded = state_indices.shape[0]
        if has_initial_state_active is None:
            has_initial_state = jnp.ones((max_num_req_padded, ),
                                         dtype=jnp.bool_)
        else:
            assert len(has_initial_state_active) == N, (
                f"has_initial_state_active must have length N={N}, got "
                f"{len(has_initial_state_active)}")
            full = np.ones(max_num_req_padded, dtype=np.bool_)
            full[:N] = np.array(has_initial_state_active, dtype=np.bool_)
            has_initial_state = jnp.array(full)

        # ── Reference (ragged_gated_delta_rule_ref) ──
        mixed_qkv = jnp.concatenate(
            [q.reshape(T, -1),
             k.reshape(T, -1),
             v.reshape(T, -1)],
            axis=-1,
        )
        distribution_ref = jnp.array([decode_N, N, N], dtype=jnp.int32)

        ref_state, ref_o = ragged_gated_delta_rule_ref(
            mixed_qkv.astype(jnp.float32),
            b.astype(jnp.float32),
            a.astype(jnp.float32),
            h0.astype(jnp.float32),
            A_log[None, None, :],  # (1,1,H_v) to match curr_a rank in ref
            dt_bias[None, None, :],  # (1,1,H_v) to match curr_a rank in ref
            cu_seqlens,
            state_indices,
            distribution_ref,
            has_initial_state,
            n_kq=H_qk,
            n_v=H_v,
            d_k=K,
            d_v=V,
        )
        ref_o = ref_o.reshape(T, H_v, V)

        # ── Kernel ──
        pallas_o, pallas_state = fused_gdn(
            q,
            k,
            v,
            cu_seqlens,
            a,  # [T, H_v] — broadcast to [T, H_v, K] inside fused_gdn
            h0,
            state_indices,
            b=b,
            has_initial_state=has_initial_state,
            distribution=jnp.array([decode_N, N], dtype=jnp.int32),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias if use_dt_bias else None,
            lower_bound=lower_bound,
        )

        # ── Compare ──
        self.assertAllClose(pallas_o,
                            ref_o,
                            atol=atol,
                            rtol=atol,
                            check_dtypes=False)
        self.assertAllClose(
            pallas_state[:N],
            ref_state[:N],
            atol=atol,
            rtol=atol,
            check_dtypes=False,
        )

    # ── Distribution forward (decode / mixed) ──

    @parameterized.parameters(
        (13, [8, 16, 24]),
        (8, []),
        (0, [9, 15]),
        (3, [9, 15]),
    )
    def test_basic(self, decode_N, mixed_seqlens):
        self._test_fused_gdn(decode_N, mixed_seqlens, 2, 2, 128, 128)

    # ── Padded max_num_req ──

    @parameterized.parameters(
        (3, [9, 15], 5),
        (8, [], 8),
        (0, [9, 15], 4),
    )
    def test_padded_max_num_req(self, decode_N, mixed_seqlens, extra_pad):
        actual_N = decode_N + len(mixed_seqlens)
        self._test_fused_gdn(
            decode_N,
            mixed_seqlens,
            2,
            2,
            128,
            128,
            max_num_req=actual_N + extra_pad,
        )

    # ── GQA (H_v > H_qk) ──

    @parameterized.parameters(
        (5, [9, 15]),
        (8, []),
        (0, [9, 15]),
    )
    def test_gqa(self, decode_N, mixed_seqlens):
        self._test_fused_gdn(decode_N, mixed_seqlens, 2, 8, 128, 128)

    # ── has_initial_state masking ──

    def test_has_initial_state_zeros_stale_slot(self):
        """When ``has_initial_state[i]`` is False, the recurrent kernel
        must treat the slot's prior contents as zero — even though the
        DMA-loaded h0 holds whatever a previous tenant left there.

        Compares two runs over identical token inputs:
          * fresh: h0 is zero everywhere (slot already cleared).
          * stale: h0 is random nonzero at the active slots, with
            has_initial_state=False so the kernel zeros h0 in VMEM.

        The per-token outputs and the written-back state for active
        slots must match — if they don't, stale state is leaking into
        the recurrent update and a freshly-allocated mamba slot would
        corrupt the new request's trajectory.
        """
        H_qk, H_v, K, V = 2, 8, 128, 128
        # Two prefill sequences, no decodes.
        decode_N = 0
        mixed_seqlens = [9, 15]
        rng = np.random.RandomState(123)
        q, k, v, a, b, A_log, _h0_ignored, cu_seqlens, state_indices, N = (
            _make_inputs(rng, decode_N, mixed_seqlens, H_qk, H_v, K, V))
        max_num_req = state_indices.shape[0]

        h0_fresh = jnp.zeros((max_num_req, H_v, K, V), dtype=jnp.float32)
        # Stale h0: nonzero at every active slot. The kernel must
        # ignore these values when has_initial_state is False.
        h0_stale = jnp.array(
            rng.randn(max_num_req, H_v, K, V).astype(np.float32))

        has_initial_state = jnp.zeros((max_num_req, ), dtype=jnp.bool_)
        distribution = jnp.array([decode_N, N], dtype=jnp.int32)

        common_kwargs = dict(
            cu_seqlens=cu_seqlens,
            g=a,
            state_indices=state_indices,
            distribution=distribution,
            b=b,
            has_initial_state=has_initial_state,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=A_log,
        )

        # Re-create q, k, v each call — fused_gdn donates `v` so we
        # must hand it a fresh buffer per invocation.
        o_fresh, state_fresh = fused_gdn(jnp.array(q),
                                         jnp.array(k),
                                         jnp.array(v),
                                         initial_state=h0_fresh,
                                         **common_kwargs)
        o_stale, state_stale = fused_gdn(jnp.array(q),
                                         jnp.array(k),
                                         jnp.array(v),
                                         initial_state=h0_stale,
                                         **common_kwargs)

        self.assertAllClose(o_stale,
                            o_fresh,
                            atol=1e-4,
                            rtol=1e-4,
                            check_dtypes=False)
        # Compare the active slots; padding slots are not exercised.
        self.assertAllClose(state_stale[:N],
                            state_fresh[:N],
                            atol=1e-4,
                            rtol=1e-4,
                            check_dtypes=False)

    @parameterized.named_parameters(
        # Two patterns chosen for orthogonal coverage:
        #   * `all_true` — every slot uses its loaded h0 (no
        #     zeroing); the continuation case.
        #   * `alternating` — has_initial_state varies by slot, so
        #     the kernel must look up the right SMEM entry per
        #     seq; catches off-by-one or uniformly-applied
        #     conditional bugs.
        # The all-False case is covered by
        # `test_has_initial_state_zeros_stale_slot`, which directly
        # compares fused-with-stale vs fused-with-fresh rather than
        # matching the ref. Extra patterns (first-only, last-only,
        # ...) add CI time without new coverage.
        dict(testcase_name="all_true", pattern=[True, True, True, True]),
        dict(testcase_name="alternating", pattern=[True, False, True, False]),
    )
    def test_has_initial_state_pattern_matches_ref(self, pattern):
        """fused_gdn vs ragged_gated_delta_rule_ref across per-slot
        has_initial_state patterns. The ref impl's masking is well-
        validated (PR #2408), so any mismatch points at the kernel.
        """
        self._test_fused_gdn(decode_N=0,
                             mixed_seqlens=[9, 15, 33, 7],
                             H_qk=2,
                             H_v=8,
                             K=128,
                             V=128,
                             has_initial_state_active=pattern)

    def test_has_initial_state_long_sequence_multi_block(self):
        """A single sequence of 2200 tokens spans 2 blocks (bt=2048 for
        H_qk=2, H_v=8, K=V=128). The kernel must zero h0 only at the
        first block (`is_new_seq`); subsequent blocks of the same seq
        carry the running state in `h_bufs[buf_idx]` and must NOT
        re-zero, otherwise the second-block tokens see zero state and
        output diverges from ref. The continuation case (True) is
        already covered by `pattern_matches_ref_all_true`.
        """
        self._test_fused_gdn(decode_N=0,
                             mixed_seqlens=[2200],
                             H_qk=2,
                             H_v=8,
                             K=128,
                             V=128,
                             has_initial_state_active=[False])

    def test_has_initial_state_decode_plus_new_prefill(self):
        """Decodes (always has_initial_state=True) interleaved with new
        prefills (has_initial_state=False). The decode kernel writes
        state for decode positions first; the recurrent kernel must
        zero only the prefill slots, leaving decode slots' updated
        state intact.
        """
        decode_N = 5
        prefill_lens = [9, 15, 33]
        pattern = [True] * decode_N + [False] * len(prefill_lens)
        self._test_fused_gdn(decode_N=decode_N,
                             mixed_seqlens=prefill_lens,
                             H_qk=2,
                             H_v=8,
                             K=128,
                             V=128,
                             has_initial_state_active=pattern)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
