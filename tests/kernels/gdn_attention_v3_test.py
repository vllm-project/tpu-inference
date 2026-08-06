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

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax.experimental.layout import Layout

from tpu_inference.kernels.gdn.v3 import (cache_layout, config, metadata,
                                          wrapper)


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


def gdn_attention_spec_ref(
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
    read_offsets: jnp.ndarray,
    num_spec_seqs: int,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Reference for SPEC mode: per-token state checkpoints with read offsets.

    Sequence s reads its initial state from slot
    `state_indices[s] + read_offsets[s]` and writes the state after window
    position t to slot `state_indices[s] + t`.
    """
    num_tokens = qkv.shape[0]
    new_conv_state = jnp.array(conv_state)
    new_recurrent_state = jnp.array(recurrent_state)
    output = jnp.zeros((num_tokens, n_v * d_v), dtype=qkv.dtype)

    for req_idx in range(num_spec_seqs):
        base = int(state_indices[req_idx])
        read_slot = base + int(read_offsets[req_idx])
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        has_init = bool((seq_lens[req_idx] - query_len) > 0)
        c_state = (conv_state[read_slot]
                   if has_init else jnp.zeros_like(conv_state[read_slot]))
        r_state = (recurrent_state[read_slot]
                   if has_init else jnp.zeros_like(recurrent_state[read_slot]))

        # Conv1D with per-token windows.
        X = qkv[start:end]
        x_full = jnp.concatenate([c_state, X], axis=0)
        acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
        for k in range(kernel_size):
            acc += (x_full[k:k + query_len].astype(jnp.float32) *
                    conv_weight[:, 0, k].astype(jnp.float32)[None, :])
        if conv_bias is not None:
            acc += conv_bias.astype(jnp.float32)[None, :]
        conv_out = jax.nn.silu(acc).astype(qkv.dtype)
        for t in range(query_len):
            new_conv_state = new_conv_state.at[base + t].set(
                x_full[t + 1:t + kernel_size])

        # Recurrent GDN with per-token states.
        key_dim = n_kq * d_k
        q_seq = conv_out[:, :key_dim].reshape(query_len, n_kq, d_k)
        k_seq = conv_out[:, key_dim:key_dim * 2].reshape(query_len, n_kq, d_k)
        v_seq = conv_out[:, key_dim * 2:].reshape(query_len, n_v, d_v)

        repeat_factor = n_v // n_kq
        if repeat_factor > 1:
            q_seq = jnp.repeat(q_seq, repeat_factor, axis=1)
            k_seq = jnp.repeat(k_seq, repeat_factor, axis=1)

        q_seq = _l2_normalize(q_seq)
        k_seq = _l2_normalize(k_seq)
        v_seq = v_seq.astype(jnp.float32)

        beta_seq = jax.nn.sigmoid(b[start:end].astype(jnp.float32))
        g_seq = -jnp.exp(a_log.astype(jnp.float32))[None, :] * jax.nn.softplus(
            a[start:end].astype(jnp.float32) +
            dt_bias.astype(jnp.float32)[None, :])

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
            new_state = new_state.astype(recurrent_state.dtype)

            return new_state, (out_step.astype(qkv.dtype), new_state)

        _, (out_seq,
            states_seq) = jax.lax.scan(step_fn, r_state,
                                       (q_seq, k_seq, v_seq, beta_seq, g_seq))
        for t in range(query_len):
            new_recurrent_state = new_recurrent_state.at[base + t].set(
                states_seq[t])
        output = output.at[start:end].set(out_seq.reshape(
            query_len, n_v * d_v))

    return (new_conv_state, new_recurrent_state), output


def gdn_local_operands(max_reqs,
                       lengths,
                       q_loc,
                       distribution,
                       context_lens=None,
                       conv_dtype=jnp.float32,
                       recurrent_dtype=jnp.float32,
                       stale_states=False,
                       seed=0):
    """Operands for one fused_conv1d_gdn call over `lengths`.

    Shared by the comparison against the reference implementation and by
    the comparison of the two convolution state operand forms against
    each other, so both see the same inputs.

    `context_lens` gives each sequence the tokens already behind it,
    which is what makes the kernel read the slot it is about to write.
    The default of none leaves `seq_lens == query_lens` (context_len =
    0), so every sequence starts from a zero state regardless of what
    its slot holds. `stale_states` fills the slots with values instead
    of zeros, which only matters alongside a context length.
    """
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

    rngs = iter(jax.random.split(jax.random.key(seed), 12))

    query = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
    key = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
    value = jax.random.normal(next(rngs), (num_tokens, n_v * v_head_dim))
    b = jax.random.normal(next(rngs), (num_tokens, n_v))
    a = jax.random.normal(next(rngs), (num_tokens, n_v))

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

    conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim
    conv_shape = (num_blocks, kernel_size - 1, conv_dim)
    recurrent_shape = (num_blocks, n_v, kq_head_dim, v_head_dim)
    if stale_states:
        # Drawn off their own key, so the sequence above stays the one
        # the reference comparison has always run on.
        state_rngs = jax.random.split(jax.random.key(seed + 100), 2)
        conv_state = jax.random.normal(state_rngs[0], conv_shape)
        recurrent_state = jax.random.normal(state_rngs[1], recurrent_shape)
    else:
        conv_state = jnp.zeros(conv_shape)
        recurrent_state = jnp.zeros(recurrent_shape)

    query_lens = q_loc[1:max_reqs + 1] - q_loc[:max_reqs]
    if context_lens is None:
        seq_lens = jnp.asarray(query_lens, dtype=jnp.int32)
    else:
        seq_lens = jnp.asarray(query_lens + jnp.asarray(context_lens),
                               dtype=jnp.int32)

    return dict(
        qkv=jnp.concatenate([query, key, value], axis=-1),
        b=b,
        a=a,
        conv_state=conv_state.astype(conv_dtype),
        recurrent_state=recurrent_state.astype(recurrent_dtype),
        conv_weight=jnp.concatenate(
            [conv_weight_q, conv_weight_k, conv_weight_v], axis=0),
        conv_bias=jnp.concatenate([conv_bias_q, conv_bias_k, conv_bias_v],
                                  axis=-1),
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
        common_kwargs = gdn_local_operands(max_reqs, lengths, q_loc,
                                           distribution)

        gdn_attention_jitted = jax.jit(
            wrapper.fused_conv1d_gdn,
            static_argnames=[
                "n_kq", "n_v", "d_k", "d_v", "kernel_size", "conv_cache_native"
            ],
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

    @parameterized.named_parameters(
        dict(testcase_name="decode",
             max_reqs=8,
             lengths=[1] * 8,
             q_loc=list(range(9)),
             distribution=[8, 8, 8],
             context_lens=[3] * 8),
        dict(testcase_name="prefill",
             max_reqs=2,
             lengths=[64, 64],
             q_loc=[0, 64, 128],
             distribution=[0, 2, 2],
             context_lens=[0, 0]),
        dict(testcase_name="conv_window_boundary",
             max_reqs=3,
             lengths=[2, 3, 4],
             q_loc=[0, 2, 5, 9],
             distribution=[0, 3, 3],
             context_lens=[5, 5, 5]),
    )
    def test_conv_cache_native_matches_converted_operand(
            self, max_reqs, lengths, q_loc, distribution, context_lens):
        """The two convolution state operand forms give the same answer.

        The cache here is bfloat16, which is what the serving path
        allocates. The converted form widens the whole cache outside the
        kernel and narrows it again on the way out; the cache-native
        form does both inside the kernel, on the slot being read. It is
        the same rounding either way, so this is an equality and not a
        tolerance.

        The window-boundary case runs sequences of two, three and four
        tokens against a convolution window of kernel_size - 1 = 3
        rows, which is where how much of the stored window survives
        into the new one changes.
        """

        def operands():
            # Built twice: the wrapper donates the two state caches, so
            # the second call cannot be given the first call's arrays.
            return gdn_local_operands(max_reqs,
                                      lengths,
                                      q_loc,
                                      distribution,
                                      context_lens=context_lens,
                                      conv_dtype=jnp.bfloat16,
                                      stale_states=True,
                                      seed=5)

        (conv_native, rec_native
         ), out_native = wrapper.fused_conv1d_gdn(**operands(),
                                                  conv_cache_native=True)
        (conv_plain, rec_plain
         ), out_plain = wrapper.fused_conv1d_gdn(**operands(),
                                                 conv_cache_native=False)

        np.testing.assert_array_equal(np.asarray(conv_native),
                                      np.asarray(conv_plain))
        np.testing.assert_array_equal(np.asarray(rec_native),
                                      np.asarray(rec_plain))
        np.testing.assert_array_equal(np.asarray(out_native),
                                      np.asarray(out_plain))

    def test_bf16_recurrent_state_tracks_float32_reference(self):
        """A bfloat16 recurrent state cache against the float32 one.

        The recurrence still runs in float32 and only the stored
        checkpoint is narrower, so the difference is one rounding per step
        and it compounds along the trajectory. The bound comes from a
        2048-step decode trajectory: over those steps the largest relative
        L2 difference against the float32 reference is 0.0020121, while a
        deliberately broken widening used as a negative control diverges
        to 0.9191. The bound below sits between the two, five times the
        observed figure and about ninety times under the control, so it
        answers a real regression rather than the step count.
        """
        steps = 32
        bound = 0.01
        max_reqs = 4

        def final_recurrent_state(recurrent_dtype):
            operands = gdn_local_operands(max_reqs, [1] * max_reqs,
                                          list(range(max_reqs + 1)),
                                          [max_reqs] * 3,
                                          context_lens=[1] * max_reqs,
                                          conv_dtype=jnp.bfloat16,
                                          recurrent_dtype=recurrent_dtype,
                                          stale_states=True,
                                          seed=3)
            conv_state = operands.pop("conv_state")
            recurrent_state = operands.pop("recurrent_state")
            qkv_shape = operands.pop("qkv").shape
            b_shape = operands.pop("b").shape
            a_shape = operands.pop("a").shape
            for step in range(steps):
                # Both arms draw the same tokens, so the only difference
                # between them is the width the checkpoint is kept at.
                rngs = jax.random.split(jax.random.key(900 + step), 3)
                (conv_state, recurrent_state), _ = wrapper.fused_conv1d_gdn(
                    qkv=jax.random.normal(rngs[0], qkv_shape),
                    b=jax.random.normal(rngs[1], b_shape),
                    a=jax.random.normal(rngs[2], a_shape),
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                    conv_cache_native=True,
                    **operands)
            return np.asarray(recurrent_state.astype(jnp.float32))

        reference = final_recurrent_state(jnp.float32)
        narrow = final_recurrent_state(jnp.bfloat16)
        rel_l2 = (np.linalg.norm(narrow - reference) /
                  np.linalg.norm(reference))
        self.assertLess(rel_l2, bound)

    @parameterized.named_parameters(
        dict(
            testcase_name="spec_windows",
            # 3 verify windows: full (5), partial drafts (3), no drafts (1).
            spec_lengths=[5, 3, 1],
            read_offsets=[2, 0, 4],
            prefill_lengths=[],
        ),
        dict(
            testcase_name="spec_windows_padded",
            # More windows than one tile (decode_tile_size=4).
            spec_lengths=[5, 1, 3, 2, 4],
            read_offsets=[0, 1, 2, 3, 4],
            prefill_lengths=[],
        ),
        dict(
            testcase_name="spec_and_prefill",
            spec_lengths=[5, 2],
            read_offsets=[3, 1],
            prefill_lengths=[64],
        ),
    )
    def test_spec_mode_checkpoints(self, spec_lengths, read_offsets,
                                   prefill_lengths):
        """SPEC mode: initial state comes from `base + read_offset` and one
        state checkpoint per window position lands at `base + t`, matching a
        token-by-token eager reference. Prefill sequences in the same batch
        keep the original PER_SEQ behavior (read/write the base slot)."""
        kq_head_dim = 128
        v_head_dim = 128
        n_kq = 2
        n_v = 8
        kernel_size = 4
        num_spec_tokens = 4
        window = num_spec_tokens + 1

        lengths = list(spec_lengths) + list(prefill_lengths)
        num_seqs = len(lengths)
        num_spec_seqs = len(spec_lengths)
        num_tokens = sum(lengths)
        q_loc = jnp.array(np.concatenate([[0], np.cumsum(lengths)]),
                          dtype=jnp.int32)
        distribution = jnp.array([num_spec_seqs, num_spec_seqs, num_seqs],
                                 dtype=jnp.int32)

        # Slot groups of `window` consecutive slots per sequence; slot 0 is
        # the null block.
        state_indices = jnp.array([1 + i * window for i in range(num_seqs)],
                                  dtype=jnp.int32)
        num_blocks = 1 + num_seqs * window
        read_offsets_arr = jnp.array(list(read_offsets) +
                                     [0] * len(prefill_lengths),
                                     dtype=jnp.int32)

        rngs = iter(jax.random.split(jax.random.key(3), 12))
        conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim

        mixed_qkv = jax.random.normal(next(rngs), (num_tokens, conv_dim))
        b = jax.random.normal(next(rngs), (num_tokens, n_v))
        a = jax.random.normal(next(rngs), (num_tokens, n_v))

        # Every slot holds a distinct random state so a read from the wrong
        # slot (or a write to the wrong slot) is caught.
        conv_state = jax.random.normal(next(rngs),
                                       (num_blocks, kernel_size - 1, conv_dim))
        recurrent_state = jax.random.normal(
            next(rngs), (num_blocks, n_v, kq_head_dim, v_head_dim))

        conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
        conv_bias = jax.random.normal(next(rngs), (conv_dim, ))
        A_log = jax.random.normal(next(rngs), (n_v, ))
        dt_bias = jax.random.normal(next(rngs), (n_v, ))

        # All sequences continue an existing context (has_initial=True) for
        # the spec windows; prefills start fresh (context_len = 0).
        seq_lens = jnp.array([16 + length for length in spec_lengths] +
                             list(prefill_lengths),
                             dtype=jnp.int32)

        run_jitted = jax.jit(
            wrapper.fused_conv1d_gdn,
            static_argnames=[
                "n_kq", "n_v", "d_k", "d_v", "kernel_size", "num_spec_tokens"
            ],
        )

        (new_conv, new_rec), output = run_jitted(
            mixed_qkv,
            b,
            a,
            conv_state,
            recurrent_state,
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            q_loc,
            state_indices,
            distribution,
            seq_lens,
            read_offsets_arr,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
            num_spec_tokens=num_spec_tokens,
        )

        # Reference: spec windows token-by-token with checkpoints.
        (ref_conv, ref_rec), ref_output = gdn_attention_spec_ref(
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
            read_offsets=read_offsets_arr,
            num_spec_seqs=num_spec_seqs,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
        )
        # Reference for the prefill tail: original eager reference over the
        # same buffers (spec sequences excluded via the distribution).
        if prefill_lengths:
            spec_token_end = int(q_loc[num_spec_seqs])
            (ref_conv, ref_rec), prefill_output = gdn_attention_ref(
                qkv=mixed_qkv[spec_token_end:],
                b=b[spec_token_end:],
                a=a[spec_token_end:],
                conv_state=ref_conv,
                recurrent_state=ref_rec,
                conv_weight=conv_weight,
                conv_bias=conv_bias,
                a_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=q_loc[num_spec_seqs:] - q_loc[num_spec_seqs],
                state_indices=state_indices[num_spec_seqs:],
                distribution=jnp.array([0, 0, num_seqs - num_spec_seqs],
                                       dtype=jnp.int32),
                seq_lens=seq_lens[num_spec_seqs:],
                n_kq=n_kq,
                n_v=n_v,
                d_k=kq_head_dim,
                d_v=v_head_dim,
                kernel_size=kernel_size,
            )
            ref_output = ref_output.at[spec_token_end:].set(prefill_output)

        np.testing.assert_allclose(output, ref_output, rtol=2e-2, atol=2e-2)

        # Verify every written checkpoint slot; untouched slots must keep
        # their original contents.
        touched = set()
        for i, length in enumerate(spec_lengths):
            base = int(state_indices[i])
            touched.update(range(base, base + length))
        for i in range(len(prefill_lengths)):
            touched.add(int(state_indices[num_spec_seqs + i]))
        for slot in range(num_blocks):
            expected_conv = ref_conv[slot] if slot in touched else conv_state[
                slot]
            expected_rec = ref_rec[slot] if slot in touched else (
                recurrent_state[slot])
            np.testing.assert_allclose(
                new_conv[slot],
                expected_conv,
                rtol=2e-2,
                atol=2e-2,
                err_msg=f"conv checkpoint mismatch at slot {slot}")
            np.testing.assert_allclose(
                new_rec[slot],
                expected_rec,
                rtol=2e-2,
                atol=2e-2,
                err_msg=f"recurrent checkpoint mismatch at slot {slot}")

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


# The smallest operand set the wrapper's argument checks will look at.
# Two key/query projections and one value projection make up DIM.
N_KQ = N_V = 1
D_K = D_V = 128
KERNEL_SIZE = 4
DIM = N_KQ * D_K * 2 + N_V * D_V
BATCH = 2
NUM_SEQS = 2
NUM_SLOTS = NUM_SEQS + 1
NATIVE_CACHE_SHAPE = (NUM_SLOTS, KERNEL_SIZE - 1, DIM)
CACHE_LAYOUT_LOGGER = "vllm.tpu_inference.kernels.gdn.v3.cache_layout"


def boundary_operands(conv_state_shape=NATIVE_CACHE_SHAPE):
    """Abstract operands for one smallest-possible call.

    Shared by both suites below, which ask the wrapper's argument checks
    questions and never reach any arithmetic.
    """
    sds = jax.ShapeDtypeStruct
    return dict(
        qkv=sds((BATCH, DIM), jnp.bfloat16),
        b=sds((BATCH, N_V), jnp.bfloat16),
        a=sds((BATCH, N_V), jnp.bfloat16),
        conv_state=sds(conv_state_shape, jnp.bfloat16),
        recurrent_state=sds((NUM_SLOTS, N_V, D_K, D_V), jnp.float32),
        conv_weight=sds((DIM, 1, KERNEL_SIZE), jnp.bfloat16),
        conv_bias=sds((DIM, ), jnp.bfloat16),
        a_log=sds((N_V, ), jnp.float32),
        dt_bias=sds((N_V, ), jnp.float32),
        query_start_loc=sds((NUM_SEQS + 1, ), jnp.int32),
        state_indices=sds((NUM_SEQS, ), jnp.int32),
        distribution=sds((3, ), jnp.int32),
        seq_lens=sds((NUM_SEQS, ), jnp.int32),
    )


def gdn_config(mode=config.GDNMode.PER_SEQ,
               window_size=1,
               conv_cache_native=False):
    """The smallest config the two suites at the end read fields off."""
    dtypes = config.Dtypes(act_in=jnp.bfloat16,
                           act_out=jnp.bfloat16,
                           compute=jnp.float32,
                           recurrent_state=jnp.float32,
                           conv_state=jnp.bfloat16)
    return config.GDNConfig(mode=mode,
                            dtypes=dtypes,
                            batch_size=BATCH,
                            dim_size=DIM,
                            kernel_size=KERNEL_SIZE,
                            tile_size=1,
                            num_kq_heads=N_KQ,
                            num_v_heads=N_V,
                            kq_head_dim=D_K,
                            v_head_dim=D_V,
                            window_size=window_size,
                            conv_cache_native=conv_cache_native)


def trace_boundary(operands, **kwargs):
    """Abstract evaluation only: no program is built and none is run."""
    return jax.eval_shape(
        lambda **kw: wrapper.fused_conv1d_gdn(**kw,
                                              n_kq=N_KQ,
                                              n_v=N_V,
                                              d_k=D_K,
                                              d_v=D_V,
                                              kernel_size=KERNEL_SIZE,
                                              **kwargs), **operands)


class BoundaryTestCase(parameterized.TestCase):

    def assert_accepted(self, operands, **kwargs):
        """The argument checks let this call through.

        A machine with no chip cannot answer the lane-count question the
        wrapper asks a few lines past those checks, so that one refusal is
        recognised by its own words and the case then makes the same
        assertion wherever it runs. If jax rewords it, this fails rather
        than passing on a refusal it did not mean to allow.
        """
        try:
            trace_boundary(operands, **kwargs)
        except ValueError as refusal:
            self.assertIn("Unsupported TPU device kind", str(refusal))


class ReadOffsetGateTest(BoundaryTestCase):
    """Where a state read offset is accepted, and where it is refused.

    These cases stop at the wrapper's argument handling:

        pytest tests/kernels/gdn_attention_v3_test.py -k ReadOffsetGateTest
    """

    def test_read_offset_without_spec_decoding_refused(self):
        """It used to be accepted and ignored, which a caller cannot see."""
        offsets = jax.ShapeDtypeStruct((NUM_SEQS, ), jnp.int32)
        with self.assertRaises(ValueError) as refusal:
            trace_boundary(boundary_operands(),
                           read_offsets=offsets,
                           num_spec_tokens=0)
        message = str(refusal.exception)
        self.assertIn("read_offsets", message)
        self.assertIn("num_spec_tokens", message)
        self.assertIn(str(offsets.shape), message)

    def test_missing_read_offset_with_spec_decoding_refused(self):
        with self.assertRaises(ValueError) as refusal:
            trace_boundary(boundary_operands(), num_spec_tokens=4)
        self.assertIn("read_offsets is required", str(refusal.exception))

    def test_read_offset_of_the_wrong_length_refused(self):
        with self.assertRaises(ValueError) as refusal:
            trace_boundary(boundary_operands(),
                           read_offsets=jax.ShapeDtypeStruct((NUM_SEQS + 1, ),
                                                             jnp.int32),
                           num_spec_tokens=4)
        message = str(refusal.exception)
        self.assertIn("one offset per sequence", message)
        self.assertIn(f"({NUM_SEQS},)", message)

    @parameterized.named_parameters(
        ("with_spec_decoding", True, 4),
        ("with_one_speculative_token", True, 1),
        ("without_spec_decoding", False, 0),
    )
    def test_read_offset_accepted(self, pass_offsets, num_spec_tokens):
        """The serving path passes none; speculative decoding passes one.

        A single speculative token already gives a sequence two state
        checkpoints and an offset that says which of them it resumes
        from, so the gate is on there being any window at all and not on
        the window being wide.
        """
        offsets = (jax.ShapeDtypeStruct(
            (NUM_SEQS, ), jnp.int32) if pass_offsets else None)
        self.assert_accepted(boundary_operands(),
                             read_offsets=offsets,
                             num_spec_tokens=num_spec_tokens)


class NativeConvStateBoundaryTest(BoundaryTestCase):
    """What the wrapper does with a cache the kernel cannot read."""

    @parameterized.named_parameters(
        ("rank_two", (NUM_SLOTS, DIM)),
        ("rank_four", (NUM_SLOTS, KERNEL_SIZE - 1, DIM, 1)),
        ("whole_window_instead_of_the_carry_over",
         (NUM_SLOTS, KERNEL_SIZE, DIM)),
        ("another_models_channel_count",
         (NUM_SLOTS, KERNEL_SIZE - 1, 2 * DIM)),
    )
    def test_unreadable_conv_cache_refused(self, shape):
        with self.assertRaises(ValueError) as refusal:
            trace_boundary(boundary_operands(shape), conv_cache_native=True)
        message = str(refusal.exception)
        self.assertIn("is not the cache-native", message)
        self.assertIn(str(shape), message)
        self.assertIn(f"[slots, {KERNEL_SIZE - 1}, {DIM}]", message)

    def test_served_conv_cache_accepted(self):
        self.assert_accepted(boundary_operands(), conv_cache_native=True)


class ConvStateLayoutTest(parameterized.TestCase):
    """The layout a rank-3 convolution state cache is allocated in."""

    CACHE_SHAPE = (17, 3, 512)

    def setUp(self):
        super().setUp()
        # The derivation warns once per message, because it is asked once
        # per mamba layer and every layer of a model carries the same
        # cache. That record is process-global and lives in vllm, so a case
        # expecting a warning starts from an empty one rather than
        # depending on which case ran first. Named rather than caught: if
        # vllm moves this, the cases expecting a warning fail loudly.
        from vllm.logger import _print_warning_once
        _print_warning_once.cache_clear()

    @parameterized.named_parameters(
        ("bfloat16", (17, 3, 512), jnp.bfloat16, ((4, 128), (2, 1))),
        ("float16", (17, 3, 512), jnp.float16, ((4, 128), (2, 1))),
        ("float32", (17, 3, 512), jnp.float32, ((8, 128), )),
        ("int8", (17, 3, 512), jnp.int8, ((2, 128), (4, 1))),
        ("float8_e4m3fn", (17, 3, 512), jnp.float8_e4m3fn, ((2, 128), (4, 1))),
        ("wider_channels", (129, 3, 4096), jnp.bfloat16, ((4, 128), (2, 1))),
        ("single_row_window", (5, 1, 128), jnp.bfloat16, ((4, 128), (2, 1))),
    )
    def test_tiling_follows_element_width(self, shape, dtype, tiling):
        layout = cache_layout.conv_state_layout(shape, dtype)
        self.assertIsInstance(layout, Layout)
        self.assertEqual(layout.major_to_minor, (0, 1, 2))
        self.assertEqual(layout.tiling, tiling)

    @parameterized.named_parameters(
        ("rank_two", (17, 512)),
        ("rank_four", (17, 3, 512, 1)),
        ("rank_one", (512, )),
    )
    def test_cache_that_is_not_rank_three_declines(self, shape):
        """The pin is an optimization, so an odd cache still allocates."""
        with self.assertLogs(CACHE_LAYOUT_LOGGER, level="WARNING") as logs:
            self.assertIsNone(
                cache_layout.conv_state_layout(shape, jnp.bfloat16))
        printed = "\n".join(logs.output)
        self.assertIn("is not the rank-3", printed)
        self.assertIn(str(shape), printed)

    @parameterized.named_parameters(
        ("float64", jnp.float64),
        ("complex64", jnp.complex64),
    )
    def test_element_width_with_no_defined_layout_declines(self, dtype):
        with self.assertLogs(CACHE_LAYOUT_LOGGER, level="WARNING") as logs:
            self.assertIsNone(
                cache_layout.conv_state_layout(self.CACHE_SHAPE, dtype))
        self.assertIn("bytes per element", "\n".join(logs.output))

    def test_channels_first_cache_declines(self):
        """vLLM can be asked to store [blocks, channels, window] instead.

        The layout here describes the other order, and pinning it onto this
        one pads a three-wide minor dimension out to a full tile of 128,
        which is the whole cache over again more than forty times.
        """
        transposed = (17, 512, 3)
        with mock.patch.object(cache_layout,
                               "_stored_channels_first",
                               return_value=True):
            with self.assertLogs(CACHE_LAYOUT_LOGGER, level="WARNING") as logs:
                self.assertIsNone(
                    cache_layout.conv_state_layout(transposed, jnp.bfloat16))
        printed = "\n".join(logs.output)
        self.assertIn("DS", printed)
        self.assertIn(str(transposed), printed)

    @parameterized.named_parameters(
        ("channels_first", (17, 512, 3), True),
        ("channels_last", (17, 3, 512), False),
    )
    def test_order_read_off_the_shape_where_vllm_cannot_be_asked(
            self, shape, declines):
        """A vLLM predating the setting has no answer to give."""
        with mock.patch.object(cache_layout,
                               "_stored_channels_first",
                               return_value=None):
            layout = cache_layout.conv_state_layout(shape, jnp.bfloat16)
        if declines:
            self.assertIsNone(layout)
        else:
            self.assertIsInstance(layout, Layout)

    def test_decline_warns_once_per_model(self):
        """An unfamiliar cache is one fact about the model, and this is asked
        once per mamba layer, so it must not print per layer."""
        with self.assertLogs(CACHE_LAYOUT_LOGGER, level="WARNING") as logs:
            for _ in range(8):
                self.assertIsNone(
                    cache_layout.conv_state_layout((17, 512), jnp.bfloat16))
        self.assertEqual(len(logs.output), 1)


class KernelNameTest(parameterized.TestCase):
    """The profile name tells the two convolution operand forms apart."""

    def test_kernel_name_carries_the_operand_form(self):
        self.assertIn("_convnative",
                      gdn_config(conv_cache_native=True).get_kernel_name())
        self.assertNotIn("_convnative", gdn_config().get_kernel_name())


class MetadataOperandCountTest(parameterized.TestCase):
    """How many kernel operands the metadata flattens to.

    The wrapper keys `input_output_aliases` by position in the flattened
    operand list, counted off this number, so a single-checkpoint call
    has to lose the read-offset leaf from it.
    """

    def test_per_seq_metadata_carries_one_operand_fewer(self):
        arrays = dict(seq_lens=jnp.array([1, 1], dtype=jnp.int32),
                      query_start_loc=jnp.array([0, 1, 2], dtype=jnp.int32),
                      state_indices=jnp.array([1, 2], dtype=jnp.int32))
        per_seq = metadata.compute_per_seq_metadata(cfg=gdn_config(),
                                                    start_seq=jnp.int32(0),
                                                    end_seq=jnp.int32(2),
                                                    **arrays)
        windowed = metadata.compute_batched_seq_metadata(
            cfg=gdn_config(mode=config.GDNMode.BATCHED, window_size=5),
            read_offsets=jnp.zeros((2, ), dtype=jnp.int32),
            end_seq=jnp.int32(2),
            **arrays)
        self.assertIsNone(per_seq.s_idx_to_read_offset)
        self.assertEqual(len(per_seq), len(windowed) - 1)
