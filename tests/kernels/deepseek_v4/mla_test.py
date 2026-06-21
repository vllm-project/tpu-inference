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
"""Correctness test for MLA kernels."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

# Import the kernels
from tpu_inference.kernels.experimental.deepseek_v4 import mla


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_packing(dtype):
    bits = jax.dtypes.itemsize_bits(dtype)
    return 32 // bits


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    kv_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        align_to(kv_dim, 128),
    )


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def ref_implementation(
    q: jax.Array,  # [num_tokens, actual_num_q_heads, actual_lkv_dim]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    kv_lens_to_attend: jax.Array,  # i32[max_num_tokens]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    swa_accumution: jax.
    Array,  # float32[num_tokens, actual_num_q_heads, actual_lkv_dim]
    swa_l: jax.Array,  # float32[num_tokens, actual_num_q_heads]
    swa_m: jax.Array,  # float32[num_tokens, actual_num_q_heads]
    topk_indices: jax.Array | None = None,
    *,
    sm_scale: float = 1.0,
    mask_value: float | None = DEFAULT_MASK_VALUE,
):

    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    actual_lkv_dim = q.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if lkv_dim != actual_lkv_dim:
        q = jnp.pad(
            q,
            ((0, 0), (0, 0), (0, lkv_dim - actual_lkv_dim)),
            constant_values=0,
        )

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    total_num_pages, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    assert lkv_dim == q.shape[-1]

    kv_c_cache = cache_kv[..., :lkv_dim].reshape(total_num_pages, page_size,
                                                 lkv_dim)

    # Quantize and dequantize kv_c_cache to simulate the loss of quantization
    fp8_part = kv_c_cache[..., :448]
    bf16_part = kv_c_cache[..., 448:512]

    fp8_blocked = fp8_part.reshape(total_num_pages, page_size, 7, 64)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    x_amax = jnp.max(jnp.abs(fp8_blocked), axis=-1, keepdims=True)
    x_amax = jnp.clip(x_amax, 1e-4, None)
    sf = jnp.power(2.0, jnp.ceil(jnp.log2(x_amax / fp8_max)))

    fp8_quant = (fp8_blocked * (1.0 / sf)).astype(jnp.float8_e4m3fn)
    scales_quant = sf.reshape(total_num_pages, page_size,
                              7).astype(jnp.float8_e8m0fnu)

    fp8_dequant = (fp8_quant.astype(jnp.bfloat16) *
                   scales_quant[..., None].astype(jnp.bfloat16)).reshape(
                       total_num_pages, page_size, 448)
    kv_c_cache = jnp.concatenate([fp8_dequant, bf16_part], axis=-1)

    outputs = []
    for i in range(distribution[-1]):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        kv_len = kv_lens[i]

        q_i = q[q_start:q_end]  # [q_len, actual_num_q_heads, lkv_dim+r_dim]
        swa_accumution_i = swa_accumution[q_start:q_end]
        swa_l_i = swa_l[q_start:q_end]
        swa_m_i = swa_m[q_start:q_end]

        indices_start = i * pages_per_seq
        num_pages_i = cdiv(kv_len, page_size)
        indices_end = indices_start + num_pages_i
        indices = page_indices[indices_start:indices_end]

        # Gather paged kv_c and k_pe
        gathered_kv_c = kv_c_cache[
            indices]  # [num_pages_i, page_size, lkv_dim]

        # Flatten pages to sequence
        flat_kv_c = gathered_kv_c.reshape(
            -1, lkv_dim)  # [num_pages_i * page_size, lkv_dim]

        # Prepare k and v for attention
        k_i = flat_kv_c[:kv_len]  # [kv_len, lkv_dim]
        v_i = flat_kv_c[:kv_len]  # [kv_len, lkv_dim]

        # MQA attention:
        # q:[q_len, actual_num_q_heads, lkv_dim+r_dim]
        # k:[kv_len, lkv_dim+r_dim]
        # v:[kv_len, lkv_dim]
        # attn: [actual_num_q_heads, q_len, kv_len]
        attn = jnp.einsum("qnh,kh->nqk",
                          q_i,
                          k_i,
                          preferred_element_type=jnp.float32)
        attn *= sm_scale

        # Causal/CSA mask
        if topk_indices is not None:
            topk_indices_i = topk_indices[q_start:q_end]
            csa_mask = (
                topk_indices_i[:, :, None] == jnp.arange(kv_len)[None,
                                                                 None, :]).any(
                                                                     axis=1)
            mask = ~csa_mask[None, :, :]
        else:
            kv_lens_to_attend_i = kv_lens_to_attend[q_start:q_end]
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = kv_lens_to_attend_i[None, :, None] <= kv_span
        attn = jnp.where(mask, mask_value, attn)

        m_2 = jnp.max(attn, axis=-1, keepdims=True)
        m_1 = jnp.transpose(swa_m_i, (1, 0))[:, :, None]
        m = jnp.maximum(m_1, m_2)

        l_1 = jnp.transpose(swa_l_i, (1, 0))[:, :, None]
        l_1_scaled = l_1 * jnp.exp(m_1 - m)
        l_2 = jnp.sum(jnp.exp(attn - m), axis=-1, keepdims=True)
        l_sinks = jnp.exp(attention_sinks[..., None, None] - m)
        L = l_1_scaled + l_2 + l_sinks

        p_2 = jnp.exp(attn - m)
        acc_2 = jnp.einsum("nqk,kl->qnl", p_2, v_i)
        exp_m1_diff = jnp.transpose(jnp.exp(m_1 - m), (1, 0, 2))
        acc_1_scaled = swa_accumution_i * exp_m1_diff
        acc = acc_1_scaled + acc_2

        out_i = (acc / jnp.transpose(L, (1, 0, 2))).astype(q_i.dtype)
        outputs.append(out_i)

    return jnp.concatenate(outputs, axis=0)


def generate_attention_sinks(rng, num_heads):
    high = 500
    low = 200
    return jnp.array(
        rng.random(size=(num_heads, ), dtype=np.float32) * (high - low) + low)


def gen_random(rng, shape, dtype):
    return jnp.array(rng.random(size=shape, dtype=np.float32)).astype(dtype)


def gen_random_int(rng, shape, low, high):
    return jnp.array(rng.integers(low=low, high=high, size=shape))


class CorrectnessTest(parameterized.TestCase):

    @parameterized.parameters(True, False)
    def test_correctness(self, is_csa: bool = False):
        if is_csa:
            topk = 1024
        else:
            topk = None
        rng = np.random.default_rng(1234)

        print(f"JAX Backend: {jax.default_backend()}")

        # Configuration (Smaller for correctness test)
        batch_size = 12
        num_heads = 64
        head_dim = 512
        page_size = 16

        num_decode_seqs = batch_size // 2
        new_kv_lens = gen_random_int(
            rng,
            (batch_size - num_decode_seqs, ),
            4,
            60,
        )
        new_kv_lens = jnp.concatenate([
            jnp.ones((num_decode_seqs, ), dtype=jnp.int32),
            new_kv_lens,
        ])
        cu_q_lens = jnp.concatenate(
            [jnp.array([0]),
             jnp.cumulative_sum(new_kv_lens, dtype=jnp.int32)])
        kv_lens = new_kv_lens + gen_random_int(rng, (batch_size, ), 30, 200)
        total_tokens = jnp.sum(new_kv_lens)

        # Deterministic Inputs for debugging
        kv_dtype = jnp.bfloat16
        q_dtype = jnp.bfloat16

        # q: [total_tokens, num_heads, head_dim]
        q = gen_random(rng, (total_tokens, num_heads, head_dim), q_dtype)
        # q_pe: [total_tokens, num_heads, pe_dim]

        # Metadata
        kv_lens_to_attend = []
        topk_indices = []

        if is_csa:
            for i in range(batch_size):
                kv_len_i = int(kv_lens[i])
                for _ in range(new_kv_lens[i]):
                    perm = rng.permutation(kv_len_i)
                    indices = list(perm[:topk])
                    if len(indices) < topk:
                        indices.extend([-1] * (topk - len(indices)))
                    topk_indices.append(indices)
            topk_indices = jnp.array(topk_indices, dtype=jnp.int32)
        else:
            for i in range(batch_size):
                for _ in range(new_kv_lens[i]):
                    kv_lens_to_attend.append(
                        rng.integers(low=0, high=kv_lens[i]))
            kv_lens_to_attend = jnp.array(kv_lens_to_attend, dtype=jnp.int32)

        pages_per_seq = cdiv(500, page_size)
        page_indices = jnp.arange(batch_size * pages_per_seq, dtype=jnp.int32)

        # Cache setup
        total_pages = batch_size * pages_per_seq

        cache_kv_base = (gen_random(
            rng,
            get_kv_cache_shape(total_pages, page_size, head_dim, kv_dtype),
            jnp.float32,
        ) * 40.0 - 20.0).astype(kv_dtype)

        # Quantize cache_kv_base to create cache_kv_agent in DSV4 FP8 format.
        kv_c_flat = cache_kv_base.reshape(total_pages, page_size, head_dim)
        fp8_part = kv_c_flat[..., :448]
        bf16_part = kv_c_flat[..., 448:512]

        fp8_blocked = fp8_part.reshape(total_pages, page_size, 7, 64)
        fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
        x_amax = jnp.max(jnp.abs(fp8_blocked), axis=-1, keepdims=True)
        x_amax = jnp.clip(x_amax, 1e-4, None)
        sf = jnp.power(2.0, jnp.ceil(jnp.log2(x_amax / fp8_max)))

        fp8_quant = (fp8_blocked * (1.0 / sf)).astype(jnp.float8_e4m3fn)
        scales_quant = sf.reshape(total_pages, page_size,
                                  7).astype(jnp.float8_e8m0fnu)

        fp8_uint8 = jax.lax.bitcast_convert_type(
            fp8_quant.reshape(total_pages, page_size, 448), jnp.uint8)
        bf16_uint8 = jax.lax.bitcast_convert_type(
            bf16_part, jnp.uint8).reshape(total_pages, page_size, 128)
        scales_uint8 = jax.lax.bitcast_convert_type(scales_quant, jnp.uint8)
        pad_uint8 = jnp.zeros((total_pages, page_size, 57), dtype=jnp.uint8)

        flat_cache_agent = jnp.concatenate(
            [fp8_uint8, bf16_uint8, scales_uint8, pad_uint8], axis=-1)
        kernel_cache = flat_cache_agent.reshape(total_pages, page_size // 4, 4,
                                                640)

        distribution = jnp.array(
            [num_decode_seqs, num_decode_seqs, batch_size], dtype=jnp.int32)

        attention_sinks = generate_attention_sinks(rng, num_heads)

        swa_accumution = jnp.ones_like(q) * 5000
        swa_l = jnp.ones((total_tokens, num_heads), dtype=jnp.float32) * 200
        swa_m = jnp.ones((total_tokens, num_heads), dtype=jnp.float32) * 500

        print("Running Baseline Reference...")
        if is_csa:
            out_base = ref_implementation(
                q,
                cache_kv_base,
                kv_lens,
                None,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                swa_accumution,
                swa_l,
                swa_m,
                topk_indices=topk_indices,
                sm_scale=1.0,
            )
            out_agent = mla.mla_ragged_paged_attention(
                q,
                kernel_cache,
                kv_lens,
                None,
                topk_indices,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                swa_accumution,
                swa_l,
                swa_m,
                sm_scale=1.0,
                num_kv_pages_per_block=1,
                num_queries_per_block=16,
            )
        else:
            out_base = ref_implementation(
                q,
                cache_kv_base,
                kv_lens,
                kv_lens_to_attend,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                swa_accumution,
                swa_l,
                swa_m,
                sm_scale=1.0,
            )
            out_agent = mla.mla_ragged_paged_attention(
                q,
                kernel_cache,
                kv_lens,
                kv_lens_to_attend,
                None,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                swa_accumution,
                swa_l,
                swa_m,
                sm_scale=1.0,
                num_kv_pages_per_block=1,
                num_queries_per_block=16,
            )
        out_base.block_until_ready()
        out_agent.block_until_ready()

        # Compare
        print("Comparing output attention...")
        diff_out = np.abs(out_base - out_agent)
        print(f"Max Diff Out: {np.max(diff_out)}")
        np.testing.assert_allclose(out_base, out_agent, rtol=0.1, atol=0.1)


if __name__ == "__main__":
    absltest.main()
