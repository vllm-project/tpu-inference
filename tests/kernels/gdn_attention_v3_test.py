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

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from tpu_inference.kernels.gdn.v3 import wrapper


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x_f32 = x.astype(jnp.float32)
    norm = jnp.sqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 / norm).astype(x.dtype)


def gdn_attention_ref(
    qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: jnp.ndarray | None,
    a_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs reference GDN attention sequence-by-sequence in eager mode.

    Bypasses XLA ragged masking complexity required for jax.jit by slicing
    sequence chunks directly in Python loop using query_start_loc boundaries.
    """

    num_tokens = qkv.shape[0]
    num_valid_seqs = int(distribution[2])

    out_mixed_qkv = jnp.zeros_like(qkv)
    new_conv_state = jnp.array(conv_state)
    new_recurrent_state = jnp.array(recurrent_state)
    output = jnp.zeros((num_tokens, n_v * d_v), dtype=qkv.dtype)

    for req_idx in range(num_valid_seqs):
        s = int(state_indices[req_idx])
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        has_init = bool((seq_lens[req_idx] - query_len) > 0)
        if has_init:
            c_state = new_conv_state[s]
        else:
            c_state = jnp.zeros_like(new_conv_state[s])
            new_recurrent_state = new_recurrent_state.at[s].set(
                jnp.zeros_like(new_recurrent_state[s]))

        # Part 1: Conv1D
        X = qkv[start:end]
        x_full = jnp.concatenate([c_state, X], axis=0)
        acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
        for k in range(kernel_size):
            acc += (x_full[k:k + query_len].astype(jnp.float32) *
                    conv_weight[:, 0, k].astype(jnp.float32)[None, :])
        if conv_bias is not None:
            acc += conv_bias.astype(jnp.float32)[None, :]
        conv_out = acc.astype(qkv.dtype)
        new_conv_state = new_conv_state.at[s].set(x_full[-(kernel_size - 1):])
        out_mixed_qkv = out_mixed_qkv.at[start:end].set(conv_out)

    out_mixed_qkv = jax.nn.silu(out_mixed_qkv)

    for req_idx in range(num_valid_seqs):
        s = int(state_indices[req_idx])
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        r_state = new_recurrent_state[s]
        qkv_seq = out_mixed_qkv[start:end]
        key_dim = n_kq * d_k
        q_seq = qkv_seq[:, :key_dim].reshape(query_len, n_kq, d_k)
        k_seq = qkv_seq[:, key_dim:key_dim * 2].reshape(query_len, n_kq, d_k)
        v_seq = qkv_seq[:, key_dim * 2:].reshape(query_len, n_v, d_v)

        repeat_factor = n_v // n_kq
        if repeat_factor > 1:
            q_seq = jnp.repeat(q_seq, repeat_factor, axis=1)
            k_seq = jnp.repeat(k_seq, repeat_factor, axis=1)

        q_seq = _l2_normalize(q_seq)
        k_seq = _l2_normalize(k_seq)
        v_seq = v_seq.astype(jnp.float32)

        b_seq = b[start:end]
        a_seq = a[start:end]

        beta_seq = jax.nn.sigmoid(b_seq.astype(jnp.float32))
        g_seq = -jnp.exp(a_log.astype(jnp.float32))[None, :] * jax.nn.softplus(
            a_seq.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :])

        def step_fn(carry_state, xs):
            q_t, k_t, v_t, beta_t, g_t = xs
            q_t = q_t * (d_k**-0.5)
            exp_g = jnp.exp(g_t)

            k_state = jnp.einsum("hd, hdm -> hm", k_t, carry_state)
            v_diff = v_t - exp_g[:, None] * k_state
            v_new = beta_t[:, None] * v_diff

            q_state = jnp.einsum("hd, hdm -> hm", q_t, carry_state)
            q_k = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
            out_step = exp_g[:, None] * q_state + q_k * v_new

            k_v_new = jnp.einsum("hd, hm -> hdm", k_t, v_new)
            new_state = carry_state * exp_g[:, None, None] + k_v_new

            return new_state.astype(recurrent_state.dtype), out_step.astype(
                qkv.dtype)

        final_r_state, out_seq = jax.lax.scan(
            step_fn, r_state, (q_seq, k_seq, v_seq, beta_seq, g_seq))
        new_recurrent_state = new_recurrent_state.at[s].set(final_r_state)
        output = output.at[start:end].set(out_seq.reshape(
            query_len, n_v * d_v))

    return (new_conv_state, new_recurrent_state), output


class GDNAttentionTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name="prefill",
            max_reqs=1,
            lengths=[8192],
            q_loc=[0, 8192],
            distribution=[0, 0, 3],
        ),
        dict(
            testcase_name="mixed",
            max_reqs=3,
            lengths=[256, 128, 128],
            q_loc=[0, 256, 384, 512],
            distribution=[0, 3, 3],
        ),
        dict(
            testcase_name="decode_only",
            max_reqs=64,
            lengths=[1] * 64,
            q_loc=list(range(65)),
            distribution=[64, 64, 64],
        ),
        dict(
            testcase_name="mixed_prefill_decode",
            max_reqs=11,
            lengths=[1] * 8 + [128, 128, 256],
            q_loc=[0, 1, 2, 3, 4, 5, 6, 7, 8, 136, 264, 520],
            distribution=[8, 11, 11],
        ),
        dict(
            testcase_name="padded_mixed_prefill",
            max_reqs=16,
            lengths=[128, 64, 32, 16, 8],
            q_loc=[0, 128, 192, 224, 240, 248] + [1] * 11,
            distribution=[0, 5, 5],
        ),
        dict(
            testcase_name="padded_decode_only",
            max_reqs=512,
            lengths=[1] * 64,
            q_loc=list(range(65)) + [1] * 448,
            distribution=[64, 64, 64],
        ),
        dict(
            testcase_name="prefill_fused",
            max_reqs=1,
            lengths=[8192],
            q_loc=[0, 8192],
            distribution=[0, 0, 3],
        ),
        dict(
            testcase_name="mixed_fused",
            max_reqs=3,
            lengths=[256, 128, 128],
            q_loc=[0, 256, 384, 512],
            distribution=[0, 3, 3],
        ),
    )
    def test_run_jax_gdn_attention_local(self, max_reqs, lengths, q_loc,
                                         distribution):
        kq_head_dim = 128
        v_head_dim = 128
        n_kq = 2
        n_v = 8
        kernel_size = 4

        num_tokens = sum(lengths)

        q_loc = jnp.array(q_loc)
        distribution = jnp.array(distribution, dtype=jnp.int32)

        # recurrent_state[0] and conv_state[0] are reserved for null blocks
        # (invalid / padded tokens). so start with index 1
        state_indices = jnp.arange(1, max_reqs + 1)
        num_blocks = max_reqs + 1

        rngs = iter(jax.random.split(jax.random.key(0), 12))

        query = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
        key = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
        value = jax.random.normal(next(rngs), (num_tokens, n_v * v_head_dim))
        b = jax.random.normal(next(rngs), (num_tokens, n_v))
        a = jax.random.normal(next(rngs), (num_tokens, n_v))

        conv_state_q = jnp.zeros(
            (num_blocks, kernel_size - 1, n_kq * kq_head_dim))
        conv_state_k = jnp.zeros(
            (num_blocks, kernel_size - 1, n_kq * kq_head_dim))
        conv_state_v = jnp.zeros(
            (num_blocks, kernel_size - 1, n_v * v_head_dim))
        recurrent_state = jnp.zeros((num_blocks, n_v, kq_head_dim, v_head_dim))

        conv_weight_q = jax.random.normal(next(rngs),
                                          (n_kq * kq_head_dim, 1, kernel_size))
        conv_weight_k = jax.random.normal(next(rngs),
                                          (n_kq * kq_head_dim, 1, kernel_size))
        conv_weight_v = jax.random.normal(next(rngs),
                                          (n_v * v_head_dim, 1, kernel_size))

        conv_bias_q = jax.random.normal(next(rngs), (n_kq * kq_head_dim, ))
        conv_bias_k = jax.random.normal(next(rngs), (n_kq * kq_head_dim, ))
        conv_bias_v = jax.random.normal(next(rngs), (n_v * v_head_dim, ))

        A_log = jax.random.normal(next(rngs), (n_v, ))
        dt_bias = jax.random.normal(jax.random.key(0), (n_v, ))

        mixed_qkv = jnp.concatenate([query, key, value], axis=-1)
        conv_state = jnp.concatenate(
            [conv_state_q, conv_state_k, conv_state_v], axis=-1)
        conv_weight = jnp.concatenate(
            [conv_weight_q, conv_weight_k, conv_weight_v], axis=0)
        conv_bias = jnp.concatenate([conv_bias_q, conv_bias_k, conv_bias_v],
                                    axis=-1)

        gdn_attention_jitted = jax.jit(
            wrapper.fused_conv1d_gdn,
            static_argnames=["n_kq", "n_v", "d_k", "d_v", "kernel_size"],
        )

        # All sequences in this test start from a fresh slot; the existing
        # parametrizations don't exercise prefix-cache-hit / chunked-prefill
        # continuation. ``seq_lens == query_lens`` (context_len = 0)
        # reproduces the prior behavior (zero initial state regardless of
        # slot contents).
        seq_lens = jnp.asarray(q_loc[1:max_reqs + 1] - q_loc[:max_reqs],
                               dtype=jnp.int32)

        common_kwargs = dict(
            qkv=mixed_qkv,
            b=b,
            a=a,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
        )

        # Run ref
        new_states_ref, output_ref = gdn_attention_ref(**common_kwargs)

        # Run chunked
        new_states_chunked, output_chunked = gdn_attention_jitted(
            **common_kwargs)

        # Compare results
        np.testing.assert_allclose(output_chunked,
                                   output_ref,
                                   rtol=2e-2,
                                   atol=2e-2)
        np.testing.assert_allclose(new_states_chunked[0],
                                   new_states_ref[0],
                                   rtol=2e-2,
                                   atol=2e-2)
        np.testing.assert_allclose(new_states_chunked[1],
                                   new_states_ref[1],
                                   rtol=2e-2,
                                   atol=2e-2)

    def test_has_initial_state_zeros_stale_slot(self):
        """Ensure stale states are ignore by new request.

        A new prefill landing on a slot whose previous tenant left
        non-zero state must produce the same output and final state as it
        would on a fresh-zero slot. This exercises the production bug fixed
        by the `has_initial_state` plumbing: vLLM's mamba pool reuses
        freed slots without clearing them, and previously the TPU GDN
        kernel consumed the stale state, silently corrupting the new
        request's recurrent trajectory.
        """
        kq_head_dim = 128
        v_head_dim = 128
        n_kq = 2
        n_v = 8
        kernel_size = 4

        # Two requests, one prefill of 64 tokens each.
        max_reqs = 2
        lengths = [64, 64]
        q_loc = jnp.array([0, 64, 128])
        distribution = jnp.array([0, 2, 2], dtype=jnp.int32)
        num_tokens = sum(lengths)

        state_indices = jnp.arange(1, max_reqs + 1)
        num_blocks = max_reqs + 1

        rngs = iter(jax.random.split(jax.random.key(7), 12))
        query = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
        key = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
        value = jax.random.normal(next(rngs), (num_tokens, n_v * v_head_dim))
        b = jax.random.normal(next(rngs), (num_tokens, n_v))
        a = jax.random.normal(next(rngs), (num_tokens, n_v))

        conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim
        conv_state_fresh = jnp.zeros((num_blocks, kernel_size - 1, conv_dim))
        recurrent_state_fresh = jnp.zeros(
            (num_blocks, n_v, kq_head_dim, v_head_dim))

        # Build a "stale" pair where the slots that the two new requests
        # land on are filled with arbitrary nonzero values (simulating a
        # prior request that finished without the pool clearing the slot).
        stale_conv = jax.random.normal(next(rngs),
                                       (num_blocks, kernel_size - 1, conv_dim))
        stale_recurrent = jax.random.normal(
            next(rngs), (num_blocks, n_v, kq_head_dim, v_head_dim))
        # Slot 0 is the null block; leave it zero.
        conv_state_stale = conv_state_fresh.at[1:].set(stale_conv[1:])
        recurrent_state_stale = recurrent_state_fresh.at[1:].set(
            stale_recurrent[1:])

        conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
        conv_bias = jax.random.normal(next(rngs), (conv_dim, ))
        A_log = jax.random.normal(next(rngs), (n_v, ))
        dt_bias = jax.random.normal(next(rngs), (n_v, ))

        mixed_qkv = jnp.concatenate([query, key, value], axis=-1)

        run_jitted = jax.jit(
            wrapper.fused_conv1d_gdn,
            static_argnames=["n_kq", "n_v", "d_k", "d_v", "kernel_size"],
        )

        # Both requests are brand new — no prior context. seq_lens equals
        # query_lens so context_len = 0 → has_initial_state = False.
        seq_lens_new = jnp.asarray(lengths, dtype=jnp.int32)

        common_kwargs = dict(
            qkv=mixed_qkv,
            b=b,
            a=a,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens_new,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
        )

        # Reference run: fresh-zero slots.
        (new_conv_fresh, new_rec_fresh), output_fresh = run_jitted(
            conv_state=conv_state_fresh,
            recurrent_state=recurrent_state_fresh,
            **common_kwargs,
        )
        # Stale-slot run: same inputs, but the slots already contain a
        # prior request's state. With the fix, has_initial_state=False
        # masks that out — outputs and the writeback at the active slots
        # must match the fresh-zero run.
        (new_conv_stale, new_rec_stale), output_stale = run_jitted(
            conv_state=conv_state_stale,
            recurrent_state=recurrent_state_stale,
            **common_kwargs,
        )

        np.testing.assert_allclose(output_fresh,
                                   output_stale,
                                   rtol=1e-5,
                                   atol=1e-5)
        # Compare the active slots (1..max_reqs+1); slot 0 (null) is
        # untouched in both runs and inactive slots are unused.
        np.testing.assert_allclose(
            new_conv_fresh[1:max_reqs + 1],
            new_conv_stale[1:max_reqs + 1],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            new_rec_fresh[1:max_reqs + 1],
            new_rec_stale[1:max_reqs + 1],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_has_initial_state_preserves_continuation(self):
        """Ensure existing slots are read during multi-step execution.
        
        When `has_initial_state[i]` is True, the kernel must use the
        slot's existing state as the prefill's initial state. This is the
        chunked-prefill / prefix-cache continuation path: a prior step
        wrote a recurrent/conv state to the slot, and the next prefill
        chunk for the same request must continue from that state — not
        from zero. Compares against running the equivalent single-shot
        prefill on the concatenated token stream from a zero state.
        """

        kq_head_dim = 128
        v_head_dim = 128
        n_kq = 2
        n_v = 8
        kernel_size = 4

        # One request, prefill split into two halves of 32 tokens each.
        # Step A: tokens [0, 32) starting from zero state.
        # Step B: tokens [32, 64) starting from Step A's final state.
        # Reference: tokens [0, 64) in a single shot from zero state.
        half = 32
        full = 64
        state_indices = jnp.array([1])
        num_blocks = 2

        rngs = iter(jax.random.split(jax.random.key(11), 12))
        query = jax.random.normal(next(rngs), (full, n_kq * kq_head_dim))
        key = jax.random.normal(next(rngs), (full, n_kq * kq_head_dim))
        value = jax.random.normal(next(rngs), (full, n_v * v_head_dim))
        b = jax.random.normal(next(rngs), (full, n_v))
        a = jax.random.normal(next(rngs), (full, n_v))

        conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim
        conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
        conv_bias = jax.random.normal(next(rngs), (conv_dim, ))
        A_log = jax.random.normal(next(rngs), (n_v, ))
        dt_bias = jax.random.normal(next(rngs), (n_v, ))

        mixed_qkv_full = jnp.concatenate([query, key, value], axis=-1)
        mixed_qkv_a = mixed_qkv_full[:half]
        mixed_qkv_b = mixed_qkv_full[half:]

        conv_state_zero = jnp.zeros((num_blocks, kernel_size - 1, conv_dim))
        recurrent_state_zero = jnp.zeros(
            (num_blocks, n_v, kq_head_dim, v_head_dim))

        run_jitted = jax.jit(
            wrapper.fused_conv1d_gdn,
            static_argnames=["n_kq", "n_v", "d_k", "d_v", "kernel_size"],
        )
        common_static = dict(
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=A_log,
            dt_bias=dt_bias,
            state_indices=state_indices,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
        )

        # Single-shot reference (all 64 tokens, zero state, has_initial=False
        # encoded as seq_lens == query_lens == [full]).
        (_, _), output_ref = run_jitted(
            qkv=mixed_qkv_full,
            b=b,
            a=a,
            conv_state=conv_state_zero,
            recurrent_state=recurrent_state_zero,
            query_start_loc=jnp.array([0, full]),
            distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
            seq_lens=jnp.array([full], dtype=jnp.int32),
            **common_static,
        )

        # Step A: first 32 tokens, zero state, has_initial=False.
        (conv_after_a, rec_after_a), output_a = run_jitted(
            qkv=mixed_qkv_a,
            b=b[:half],
            a=a[:half],
            conv_state=conv_state_zero,
            recurrent_state=recurrent_state_zero,
            query_start_loc=jnp.array([0, half]),
            distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
            seq_lens=jnp.array([half], dtype=jnp.int32),
            **common_static,
        )

        # Step B: next 32 tokens, slot now holds Step A's state.
        # seq_lens=[full] with query_lens=[half] gives context_len=half>0,
        # i.e., has_initial=True so the kernel continues from that state.
        (_, _), output_b = run_jitted(
            qkv=mixed_qkv_b,
            b=b[half:],
            a=a[half:],
            conv_state=conv_after_a,
            recurrent_state=rec_after_a,
            query_start_loc=jnp.array([0, half]),
            distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
            seq_lens=jnp.array([full], dtype=jnp.int32),
            **common_static,
        )

        # Step A's output must match the first half of the single-shot
        # reference; Step B (continuation) must match the second half.
        np.testing.assert_allclose(output_a,
                                   output_ref[:half],
                                   rtol=2e-2,
                                   atol=2e-2)
        np.testing.assert_allclose(output_b,
                                   output_ref[half:],
                                   rtol=2e-2,
                                   atol=2e-2)
