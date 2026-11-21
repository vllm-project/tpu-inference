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
import pytest

from tpu_inference.kernels.gdn.v3 import fast_decode
from tpu_inference.kernels.gdn.v3.rms_norm_gated import (_jax_rms_norm_gated,
                                                         pallas_rms_norm_gated)


def test_can_use_fast_decode_guards():
    assert fast_decode.can_use_fast_decode(
        num_spec_tokens=0,
        n_kq=2,
        n_v=6,
        d_k=128,
        d_v=128,
        dim=1280,
        kernel_size=4,
        batch_size=128,
        num_seqs=128,
    )
    assert not fast_decode.can_use_fast_decode(
        num_spec_tokens=1,
        n_kq=2,
        n_v=6,
        d_k=128,
        d_v=128,
        dim=1280,
        kernel_size=4,
        batch_size=128,
        num_seqs=128,
    )
    assert not fast_decode.can_use_fast_decode(
        num_spec_tokens=0,
        n_kq=2,
        n_v=6,
        d_k=64,
        d_v=128,
        dim=1280,
        kernel_size=4,
        batch_size=128,
        num_seqs=128,
    )


def test_pallas_rms_norm_gated_fallback_cpu_and_tpu():
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k1, (16, 256), dtype=jnp.bfloat16)
    z = jax.random.normal(k2, (16, 256), dtype=jnp.bfloat16)
    w = jax.random.normal(k3, (128, ), dtype=jnp.bfloat16)
    out = pallas_rms_norm_gated(x, z, w, eps=1e-6, head_dim=128)
    ref = _jax_rms_norm_gated(x, z, w, eps=1e-6, head_dim=128)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.parametrize("rows", [256, 1024])
def test_pallas_rms_norm_gated_tpu_parity(rows):
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel requires TPU backend")
    k1, k2, k3 = jax.random.split(jax.random.key(1), 3)
    x = jax.random.normal(k1, (rows, 768), dtype=jnp.bfloat16)
    z = jax.random.normal(k2, (rows, 768), dtype=jnp.bfloat16)
    w = jax.random.normal(k3, (128, ), dtype=jnp.bfloat16)
    out = pallas_rms_norm_gated(x, z, w, eps=1e-6, head_dim=128)
    ref = _jax_rms_norm_gated(x, z, w, eps=1e-6, head_dim=128)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.parametrize("fuse_norm", [False, True])
def test_fast_decode_conv1d_gdn_tpu_parity(fuse_norm):
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel requires TPU backend")
    from tests.kernels import gdn_attention_v3_test

    n_kq, n_v, d_k, d_v, kernel_size = 2, 6, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    num_seqs = 16
    num_blocks = num_seqs + 2

    rngs = iter(jax.random.split(jax.random.key(42), 11))
    qkv = jax.random.normal(next(rngs),
                            (num_seqs, dim), dtype=jnp.bfloat16) * 0.2
    b = jax.random.normal(next(rngs),
                          (num_seqs, n_v), dtype=jnp.bfloat16) * 0.2
    a = jax.random.normal(next(rngs),
                          (num_seqs, n_v), dtype=jnp.bfloat16) * 0.2
    conv_state = jax.random.normal(next(rngs),
                                   (num_blocks, kernel_size - 1, dim),
                                   dtype=jnp.bfloat16) * 0.2
    rec_state = jax.random.normal(next(rngs), (num_blocks, n_v, d_k, d_v),
                                  dtype=jnp.float32) * 0.2
    conv_weight = jax.random.normal(next(rngs), (dim, 1, kernel_size),
                                    dtype=jnp.bfloat16) * 0.2
    conv_bias = jax.random.normal(next(rngs),
                                  (dim, ), dtype=jnp.bfloat16) * 0.1
    a_log = jax.random.normal(next(rngs), (n_v, ), dtype=jnp.float32) * 0.2
    dt_bias = jax.random.normal(next(rngs), (n_v, ), dtype=jnp.bfloat16) * 0.1
    z = jax.random.normal(next(rngs), (num_seqs, n_v * d_v),
                          dtype=jnp.bfloat16) * 0.2
    nw = jax.random.normal(next(rngs), (d_v, ), dtype=jnp.bfloat16) * 0.2

    state_indices = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)
    q_loc = jnp.arange(0, num_seqs + 1, dtype=jnp.int32)
    dist = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
    seq_lens = jnp.full((num_seqs, ), 32, dtype=jnp.int32)

    out, new_cs, new_rs = fast_decode.fast_decode_conv1d_gdn(
        qkv,
        b,
        a,
        conv_state,
        rec_state,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        state_indices,
        state_indices,
        dist,
        seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        tile_s=8,
        zero_initialize_out=True,
        act_out_dtype=jnp.bfloat16,
        z=z if fuse_norm else None,
        norm_weight=nw if fuse_norm else None,
        norm_eps=1e-6,
        norm_bf16_roundtrip=True,
    )
    (ref_cs, ref_rs), ref_out = gdn_attention_v3_test.gdn_attention_ref(
        qkv=qkv,
        b=b,
        a=a,
        conv_state=conv_state,
        recurrent_state=rec_state,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=q_loc,
        state_indices=state_indices,
        distribution=dist,
        seq_lens=seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )
    if fuse_norm:
        ref_out = _jax_rms_norm_gated(ref_out, z, nw, eps=1e-6, head_dim=d_v)
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32),
        np.asarray(ref_out, dtype=np.float32),
        atol=2e-2,
        rtol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(new_cs, dtype=np.float32),
        np.asarray(ref_cs, dtype=np.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        np.asarray(new_rs, dtype=np.float32),
        np.asarray(ref_rs, dtype=np.float32),
        atol=2e-2,
        rtol=2e-2,
    )
