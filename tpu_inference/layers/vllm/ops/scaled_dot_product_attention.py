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
"""Register function to overwrite torch functions used by vllm via torchax."""

import jax
import jax.numpy as jnp

from tpu_inference.layers.common.attention_interface import \
    sharded_flash_attention


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    *,
    mesh: jax.sharding.Mesh,
):
    """The same args as torch.nn.functional.scaled_dot_product_attention to use flash attention."""
    if dropout_p != 0.0:
        raise NotImplementedError("patched_sdpa does not support dropout_p")
    if enable_gqa is not False:
        raise NotImplementedError("patched_sdpa does not support enable_gqa")

    # Q, K, V shapes: (batch, num_heads, seq_len, head_dim)
    batch = query.shape[0]
    num_heads = query.shape[1]
    q_seq_len = query.shape[2]
    kv_seq_len = key.shape[2]

    # padding due to the requirement of sharded_flash_attention
    q_pad = (128 - (q_seq_len % 128)) % 128
    kv_pad = (128 - (kv_seq_len % 128)) % 128

    if q_pad > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, q_pad), (0, 0)))
    if kv_pad > 0:
        key = jnp.pad(key, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))

    # Prevent nan while using -inf
    mask_value = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
    attention_bias = jnp.zeros((batch, num_heads, q_seq_len, kv_seq_len),
                               dtype=jnp.float32)
    if attn_mask is not None:
        # attn_mask shape: (batch, num_heads, q_len, kv_len)
        if attn_mask.dtype == jnp.bool_:
            attention_bias = jnp.where(attn_mask, attention_bias, mask_value)
        else:
            attention_bias += attn_mask

    if q_pad > 0 or kv_pad > 0:
        attention_bias = jnp.pad(
            attention_bias,
            ((0, 0), (0, 0), (0, q_pad), (0, kv_pad)),
            mode="constant",
            constant_values=mask_value,
        )

    attn_fn = sharded_flash_attention(mesh,
                                      causal=is_causal,
                                      sm_scale=scale,
                                      use_attention_bias=True)
    out = attn_fn(query, key, value, attention_bias, None)

    if q_pad > 0:
        out = out[:, :, :q_seq_len, :]

    return out

def vllm_vit_sdpa(
    query,
    key,
    value,
    scale=None,
    cu_seqlens=None,
    enable_gqa=False,
    *,
    mesh,
):
    """JAX implementation for ViT SDPA used by vLLM."""
    query = jnp.swapaxes(query, 1, 2)
    key = jnp.swapaxes(key, 1, 2)
    value = jnp.swapaxes(value, 1, 2)

    batch = query.shape[0]
    num_heads = query.shape[1]
    q_seq_len = query.shape[2]
    kv_seq_len = key.shape[2]

    q_pad = (128 - (q_seq_len % 128)) % 128
    kv_pad = (128 - (kv_seq_len % 128)) % 128

    if q_pad > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, q_pad), (0, 0)))
    if kv_pad > 0:
        key = jnp.pad(key, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))

    if cu_seqlens is not None:
        cu_seqlens_arr = jnp.array(cu_seqlens)
        lens = cu_seqlens_arr[1:] - cu_seqlens_arr[:-1]
        num_segs = lens.shape[0]

        q_real_seg = jnp.repeat(jnp.arange(num_segs), lens, total_repeat_length=q_seq_len)
        kv_real_seg = q_real_seg

        if q_pad > 0:
            q_pad_seg = jnp.full((q_pad,), num_segs)
            q_seg = jnp.concatenate([q_real_seg, q_pad_seg])
        else:
            q_seg = q_real_seg

        if kv_pad > 0:
            kv_pad_seg = jnp.full((kv_pad,), num_segs)
            kv_seg = jnp.concatenate([kv_real_seg, kv_pad_seg])
        else:
            kv_seg = kv_real_seg

        q_seg = jnp.broadcast_to(q_seg, (batch, q_seg.shape[0]))
        kv_seg = jnp.broadcast_to(kv_seg, (batch, kv_seg.shape[0]))

        from tpu_inference.kernels.flash_attention.kernel import SegmentIds
        seg_ids = SegmentIds(q=q_seg, kv=kv_seg)
    else:
        seg_ids = None

    from tpu_inference.layers.common.attention_interface import \
        sharded_flash_attention
    attn_fn = sharded_flash_attention(
        mesh,
        causal=False,
        sm_scale=scale,
        use_attention_bias=False
    )

    out = attn_fn(query, key, value, seg_ids)

    if q_pad > 0:
        out = out[:, :, :q_seq_len, :]

    out = jnp.swapaxes(out, 1, 2)

    return out

