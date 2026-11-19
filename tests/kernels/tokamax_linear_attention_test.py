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
"""gdn_attention_v3_test's GDNAttentionTest on tokamax's GDN kernel."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from tokamax._src.ops.causal_conv1d_gated_delta_rule import (config, tiling,
                                                             wrapper)

from tests.kernels import gdn_attention_v3_test
from tpu_inference.layers.common import gdn_attention
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES


class TokamaxGDNAttentionTest(gdn_attention_v3_test.GDNAttentionTest):
    """Every GDN v3 case, reference and tolerance, with the kernel swapped."""

    def setUp(self):
        super().setUp()
        self.enter_context(
            mock.patch.object(gdn_attention_v3_test.wrapper,
                              "fused_conv1d_gdn", wrapper.fused_conv1d_gdn))

    def test_run_jax_gdn_attention_bf16_default_backend(self):
        """Runs run_jax_gdn_attention -> GDNLinearAttentionBackend in bf16."""
        n_kq, n_v, d_k, d_v, kernel_size = 2, 8, 128, 128, 4
        conv_dim = 2 * n_kq * d_k + n_v * d_v
        lengths = [1, 1, 1, 1, 128, 160]
        num_seqs = len(lengths)
        num_tokens = sum(lengths)
        q_loc = jnp.array(np.concatenate([[0], np.cumsum(lengths)]),
                          dtype=jnp.int32)
        distribution = jnp.array([4, 4, num_seqs], dtype=jnp.int32)
        seq_lens = jnp.array([32, 17, 9, 64, 128, 224], dtype=jnp.int32)
        read_state_indices = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)
        state_indices = jnp.arange(num_seqs + 1,
                                   2 * num_seqs + 1,
                                   dtype=jnp.int32)
        num_blocks = 2 * num_seqs + 2

        rngs = iter(jax.random.split(jax.random.key(42), 10))
        qkv = jax.random.normal(next(rngs), (num_tokens, conv_dim),
                                dtype=jnp.bfloat16)
        b = jax.random.normal(next(rngs), (num_tokens, n_v),
                              dtype=jnp.bfloat16)
        a = jax.random.normal(next(rngs), (num_tokens, n_v),
                              dtype=jnp.bfloat16)
        conv_state = jax.random.normal(next(rngs),
                                       (num_blocks, kernel_size - 1, conv_dim),
                                       dtype=jnp.bfloat16)
        recurrent_state = jax.random.normal(next(rngs),
                                            (num_blocks, n_v, d_k, d_v),
                                            dtype=jnp.float32) * 0.1
        conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size),
                                        dtype=jnp.bfloat16)
        conv_bias = jax.random.normal(next(rngs), (conv_dim, ),
                                      dtype=jnp.bfloat16)
        a_log = jax.random.normal(next(rngs), (n_v, ), dtype=jnp.float32)
        dt_bias = jax.random.normal(next(rngs), (n_v, ), dtype=jnp.float32)

        devices = np.array([jax.devices()[0]]).reshape(
            (1, ) * len(MESH_AXIS_NAMES))
        mesh = Mesh(devices, MESH_AXIS_NAMES)

        (new_conv, new_rec), out = gdn_attention.run_jax_gdn_attention(
            j_mixed_qkv=qkv,
            j_b=b,
            j_a=a,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            j_conv_weight=conv_weight,
            j_conv_bias=conv_bias,
            j_A_log=a_log,
            j_dt_bias=dt_bias,
            state_indices=state_indices,
            query_start_loc=q_loc,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            mesh=mesh,
            read_state_indices=read_state_indices,
        )

        (ref_conv, ref_rec), ref_out = gdn_attention_v3_test.gdn_attention_ref(
            qkv=qkv,
            b=b,
            a=a,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            read_state_indices=read_state_indices,
        )

        self.assertEqual(out.dtype, jnp.bfloat16)
        self.assertEqual(new_conv.dtype, jnp.bfloat16)
        self.assertEqual(new_rec.dtype, jnp.float32)
        np.testing.assert_allclose(out.astype(jnp.float32),
                                   ref_out.astype(jnp.float32),
                                   rtol=2e-2,
                                   atol=2e-2)
        np.testing.assert_allclose(new_conv[state_indices].astype(jnp.float32),
                                   ref_conv[state_indices].astype(jnp.float32),
                                   rtol=2e-2,
                                   atol=2e-2)
        np.testing.assert_allclose(new_rec[state_indices],
                                   ref_rec[state_indices],
                                   rtol=2e-2,
                                   atol=2e-2)
        # Cached read slots remain untouched when writing into distinct slots.
        np.testing.assert_array_equal(new_conv[read_state_indices],
                                      conv_state[read_state_indices])
        np.testing.assert_array_equal(new_rec[read_state_indices],
                                      recurrent_state[read_state_indices])

    def test_spec_mode_checkpoints_many_windows(self):
        """Guards the >= 3-tile speculative-verify state DMA hazard.

        The 2-buffer StateBufferedRef race only triggers when num_tiles >= 3
        (tile i's HBM state write overlaps tile i + 2's HBM state prefetch).
        tokamax's GDNConfig.DECODE_TILE_SIZES caps decode_tile_size at 16
        (picking <= 8 on v6e/v7x here), so 48 verify windows guarantee at least
        ceil(48 / 16) = 3 tiles under dynamic tiling and 12 tiles under
        decode_tile_size=4; both are asserted explicitly via get_tile_sizes.
        """
        n_kq, n_v, d_k, d_v, kernel_size = 2, 4, 128, 128, 4
        num_spec_tokens = 3
        window = num_spec_tokens + 1
        spec_lengths = [4, 3, 2, 1] * 12
        num_seqs = len(spec_lengths)
        num_tokens = sum(spec_lengths)
        q_loc = jnp.array(np.concatenate([[0], np.cumsum(spec_lengths)]),
                          dtype=jnp.int32)
        distribution = jnp.array([num_seqs, num_seqs, num_seqs],
                                 dtype=jnp.int32)
        state_indices = jnp.array([1 + i * window for i in range(num_seqs)],
                                  dtype=jnp.int32)
        num_blocks = 1 + num_seqs * window
        read_offsets = jnp.array([0, 1, 2, 3] * 12, dtype=jnp.int32)
        seq_lens = jnp.array([16 + ln for ln in spec_lengths], dtype=jnp.int32)

        conv_dim = 2 * n_kq * d_k + n_v * d_v
        rngs = iter(jax.random.split(jax.random.key(19), 9))
        qkv = jax.random.normal(next(rngs), (num_tokens, conv_dim),
                                dtype=jnp.bfloat16)
        b = jax.random.normal(next(rngs), (num_tokens, n_v),
                              dtype=jnp.bfloat16)
        a = jax.random.normal(next(rngs), (num_tokens, n_v),
                              dtype=jnp.bfloat16)
        conv_state = jax.random.normal(next(rngs),
                                       (num_blocks, kernel_size - 1, conv_dim),
                                       dtype=jnp.bfloat16) * 0.1
        recurrent_state = jax.random.normal(next(rngs),
                                            (num_blocks, n_v, d_k, d_v),
                                            dtype=jnp.float32) * 0.1
        conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size),
                                        dtype=jnp.bfloat16) * 0.1
        conv_bias = jax.random.normal(next(rngs), (conv_dim, ),
                                      dtype=jnp.bfloat16) * 0.01
        a_log = jax.random.normal(next(rngs), (n_v, ), dtype=jnp.float32)
        dt_bias = jax.random.normal(next(rngs), (n_v, ), dtype=jnp.float32)

        (ref_conv,
         ref_rec), ref_out = gdn_attention_v3_test.gdn_attention_spec_ref(
             qkv=qkv,
             b=b,
             a=a,
             conv_state=conv_state,
             recurrent_state=recurrent_state,
             conv_weight=conv_weight,
             conv_bias=conv_bias,
             a_log=a_log,
             dt_bias=dt_bias,
             query_start_loc=q_loc,
             state_indices=state_indices,
             read_offsets=read_offsets,
             num_spec_seqs=num_seqs,
             seq_lens=seq_lens,
             n_kq=n_kq,
             n_v=n_v,
             d_k=d_k,
             d_v=d_v,
             kernel_size=kernel_size,
         )

        packing = 4 // qkv.dtype.itemsize
        padded_batch_size = ((num_tokens + packing - 1) // packing) * packing
        num_lanes = pltpu.get_tpu_info().num_lanes
        decode_vmem_limit = config.GDNConfig.get_vmem_limit_bytes(window)
        mixed_vmem_limit = config.GDNConfig.get_vmem_limit_bytes()

        for decode_tile_override in (None, 4):
            with self.subTest(decode_tile_size=decode_tile_override):
                chosen_tile_size, _ = tiling.get_tile_sizes(
                    batch_size=num_tokens,
                    num_seqs=num_seqs,
                    padded_batch_size=padded_batch_size,
                    n_kq=n_kq,
                    n_v=n_v,
                    d_k=d_k,
                    d_v=d_v,
                    kernel_size=kernel_size,
                    conv_state_dim_size=conv_dim,
                    act_out_dtype=qkv.dtype,
                    recurrent_state_dtype=recurrent_state.dtype,
                    num_lanes=num_lanes,
                    decode_vmem_limit_bytes=decode_vmem_limit,
                    mixed_vmem_limit_bytes=mixed_vmem_limit,
                    window_size=window,
                    decode_tile_size=decode_tile_override,
                )
                num_tiles = (num_seqs + chosen_tile_size -
                             1) // chosen_tile_size
                self.assertGreaterEqual(num_tiles, 3)

                (new_conv, new_rec), out = wrapper.fused_conv1d_gdn(
                    qkv,
                    b,
                    a,
                    conv_state.copy(),
                    recurrent_state.copy(),
                    conv_weight,
                    conv_bias,
                    a_log,
                    dt_bias,
                    q_loc,
                    state_indices,
                    distribution,
                    seq_lens,
                    state_indices,
                    read_offsets,
                    n_kq=n_kq,
                    n_v=n_v,
                    d_k=d_k,
                    d_v=d_v,
                    kernel_size=kernel_size,
                    num_spec_tokens=num_spec_tokens,
                    decode_tile_size=decode_tile_override,
                )

                np.testing.assert_allclose(out.astype(jnp.float32),
                                           ref_out.astype(jnp.float32),
                                           rtol=2e-2,
                                           atol=2e-2)
                np.testing.assert_allclose(new_conv[1:].astype(jnp.float32),
                                           ref_conv[1:].astype(jnp.float32),
                                           rtol=2e-2,
                                           atol=2e-2)
                np.testing.assert_allclose(new_rec[1:],
                                           ref_rec[1:],
                                           rtol=2e-2,
                                           atol=2e-2)


if __name__ == "__main__":
    absltest.main()
