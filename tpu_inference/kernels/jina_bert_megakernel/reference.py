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
"""Dense pure-JAX reference of the JinaBert encoder, for kernel tests.

Takes the same per-layer weight mappings as `pack_jina_bert_weights` and
implements the semantics of `JinaBertEncoder.__call__` (segment-masked
bidirectional attention with symmetric ALiBi, post-LN, GeGLU) with dense
[heads, T, T] attention. Runs on any backend (no Pallas).

With `mxu_dtype=jnp.bfloat16` it rounds every matmul operand to bf16 at the
same points as the megakernel's ``"default"`` precision (and normalizes the
softmax after P.V like the kernel), so the two can be compared tightly on
CPU. With `mxu_dtype=None` all matmuls run at `lax.Precision.HIGHEST`.
"""

from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import lax


def segment_ids(seq_lens: jax.Array, num_tokens: int) -> jax.Array:
    """Segment id per token, as in `encoder_only_flash_attention`."""
    num_seqs = seq_lens.shape[0]
    return jnp.repeat(jnp.arange(num_seqs + 1, dtype=jnp.int32),
                      jnp.concatenate([
                          seq_lens.astype(jnp.int32),
                          jnp.zeros((1, ), jnp.int32)
                      ]),
                      total_repeat_length=num_tokens)


def reference_jina_bert_encoder(x: jax.Array,
                                seq_lens: jax.Array,
                                layers: Sequence[Mapping[str, jax.Array]],
                                *,
                                alibi_slopes: Sequence[float],
                                eps: float = 1e-12,
                                mxu_dtype=None) -> jax.Array:
    num_tokens, d = x.shape
    num_heads = len(alibi_slopes)
    head_dim = d // num_heads
    f32 = jnp.float32
    seg = segment_ids(seq_lens, num_tokens)
    same_seq = seg[:, None] == seg[None, :]
    pos = jnp.arange(num_tokens)
    dist = jnp.abs(pos[:, None] - pos[None, :]).astype(f32)
    slopes = jnp.asarray(alibi_slopes, f32)

    def mm(eq, a, b):
        if mxu_dtype is not None:
            a, b = a.astype(mxu_dtype), b.astype(mxu_dtype)
        return jnp.einsum(eq,
                          a,
                          b,
                          precision=lax.Precision.HIGHEST,
                          preferred_element_type=f32)

    def layer_norm(v, g, b):
        mean = jnp.mean(v, axis=-1, keepdims=True)
        vc = v - mean
        var = jnp.mean(vc * vc, axis=-1, keepdims=True)
        return vc * lax.rsqrt(var + eps) * row(g) + row(b)

    def row(b):  # [D] -> [1, D] (explicit broadcasting).
        return jnp.reshape(b, (1, d)).astype(f32)

    x = x.astype(f32)
    for p in layers:

        def heads(w):
            return jnp.reshape(w, (d, num_heads, head_dim)).astype(f32)

        def head_bias(b):
            return jnp.reshape(b, (1, num_heads, head_dim)).astype(f32)

        q = mm("td,dnh->tnh", x, heads(p["q_w"])) + head_bias(p["q_b"])
        k = mm("td,dnh->tnh", x, heads(p["k_w"])) + head_bias(p["k_b"])
        v = mm("td,dnh->tnh", x, heads(p["v_w"])) + head_bias(p["v_b"])
        s = mm("tnh,snh->nts", q * head_dim**-0.5, k)
        s = s - slopes[:, None, None] * dist[None]
        s = jnp.where(same_seq[None], s, -jnp.inf)
        probs = jnp.exp(s - jnp.max(s, axis=-1, keepdims=True))
        denom = jnp.sum(probs, axis=-1)  # [N, T]
        ctx = mm("nts,snh->tnh", probs, v) / denom.T[:, :, None]
        o_w = jnp.reshape(p["o_w"], (num_heads, head_dim, d)).astype(f32)
        attn = mm("tnh,nhd->td", ctx, o_w) + row(p["o_b"])
        h1 = layer_norm(attn + x, p["ln1_g"], p["ln1_b"])
        ffn = p["down_w"].shape[0]
        g = mm("td,df->tf", h1, p["gate_w"].astype(f32))
        act = jax.nn.gelu(g[:, :ffn], approximate=False) * g[:, ffn:]
        y = mm("tf,fd->td", act, p["down_w"].astype(f32))
        x = layer_norm(y + row(p["down_b"]) + h1, p["ln2_g"], p["ln2_b"])
    return x
