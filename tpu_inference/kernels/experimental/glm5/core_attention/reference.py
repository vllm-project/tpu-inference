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
"""Pure-JAX reference for GLM-5.2's Dynamic Sparse Attention main attention
step: gather the top-``index_topk`` raw (uncompressed) KV positions selected
by the lightning indexer, then run standard causal-masked dense softmax
attention over just that gathered subset.

This is a correctness-focused, non-online-softmax reference (dense
``softmax`` over the ``topk`` gathered axis, not FlashAttention-style
streaming), intentionally mirroring the *math* of upstream vLLM's
self-contained Triton reference kernel
``vllm/v1/attention/ops/xpu_mla_sparse.py::triton_bf16_mla_sparse_interface``
(single shared/MQA KV head, ``-1``-padded index masking, plain scaled
dot-product attention) rather than its DeepGEMM/FlashMLA CUDA-kernel
counterparts, which have no Python reference. Tolerances used in this
module's tests mirror ``tests/v1/attention/test_sparse_mla_backends.py``:
``rtol=0.01, atol=0.01`` for bf16, ``rtol=0.065, atol=0.05`` for fp8.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sparse_causal_attention(
    q: jax.Array,  # [T, H, Dqk]
    k: jax.Array,  # [S, Dqk], single shared (MQA) K vector per raw KV token
    v: jax.Array,  # [S, Dv], single shared (MQA) V vector per raw KV token
    topk_indices: jax.Array,  # [T, topk] int32, -1 = invalid/padding
    q_positions: jax.Array,  # [T] int32 absolute positions
    k_positions: jax.Array,  # [S] int32 absolute positions
    *,
    softmax_scale: float | None = None,
) -> jax.Array:
    """Gathers the ``topk`` selected raw KV positions per query token and
    computes causal-masked dense softmax attention over just that subset.

    Args:
      q: ``[T, num_heads, qk_head_dim]``.
      k, v: ``[S, qk_head_dim]`` / ``[S, v_head_dim]``, a single KV vector per
        raw cache token, shared (broadcast) across all of ``q``'s heads --
        matching GLM-5.2's MLA-style attention (one KV "head").
      topk_indices: ``[T, topk]`` int32 indices into the ``S`` axis of
        ``k``/``v``, as produced by
        ``indexer.reference.lightning_indexer_layer``/``select_topk_indices``.
        ``-1`` marks an unused/padding slot and is excluded from the softmax.
      q_positions, k_positions: absolute token positions. Only used for a
        defensive causal re-check (the indexer's own causal masking should
        already guarantee every selected index is causally valid); a
        selected index that is somehow non-causal is masked out here too.
      softmax_scale: defaults to ``qk_head_dim ** -0.5``.

    Returns:
      ``[T, num_heads, v_head_dim]``, same dtype as ``v``.
    """
    num_tokens, num_heads, qk_head_dim = q.shape
    num_kv, kv_qk_head_dim = k.shape
    _, v_head_dim = v.shape
    if kv_qk_head_dim != qk_head_dim:
        raise ValueError(f"q/k head_dim mismatch: {qk_head_dim} vs "
                         f"{kv_qk_head_dim}")
    if topk_indices.shape[0] != num_tokens:
        raise ValueError(f"topk_indices leading dim {topk_indices.shape[0]} "
                         f"!= num query tokens {num_tokens}")

    if softmax_scale is None:
        softmax_scale = qk_head_dim**-0.5

    valid = topk_indices >= 0  # [T, topk]
    safe_indices = jnp.clip(topk_indices, 0, num_kv - 1)

    k_sel = k[safe_indices]  # [T, topk, Dqk]
    v_sel = v[safe_indices]  # [T, topk, Dv]

    # Defensive causal re-check: a selected index must not be in the future
    # relative to its query token.
    causal = k_positions[safe_indices] <= q_positions[:, None]  # [T, topk]
    valid = valid & causal

    # This is a correctness reference (used to bound production kernel
    # error), so force full float32 matmul precision -- TPU's default matmul
    # precision silently truncates float32 inputs to bf16-ish passes.
    logits = jnp.einsum(
        "thd,tkd->thk",
        q.astype(jnp.float32),
        k_sel.astype(jnp.float32),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    ) * softmax_scale  # [T, H, topk]
    logits = jnp.where(valid[:, None, :], logits, -jnp.inf)

    # Rows with zero valid entries (shouldn't happen once the indexer always
    # keeps at least the token's own causal self-position, but guard against
    # it for robustness) would otherwise produce an all -inf row -> NaN
    # softmax; fall back to a uniform distribution over an all-invalid row.
    row_has_valid = jnp.any(valid, axis=-1)  # [T]
    safe_logits = jnp.where(row_has_valid[:, None, None], logits,
                            jnp.zeros_like(logits))

    probs = jax.nn.softmax(safe_logits, axis=-1)
    probs = jnp.where(row_has_valid[:, None, None], probs,
                      0.0).astype(jnp.float32)

    out = jnp.einsum(
        "thk,tkd->thd",
        probs,
        v_sel.astype(jnp.float32),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    return out.astype(v.dtype)


def dense_causal_attention_reference(
    q: jax.Array,  # [T, H, Dqk]
    k: jax.Array,  # [S, Dqk]
    v: jax.Array,  # [S, Dv]
    q_positions: jax.Array,  # [T] int32
    k_positions: jax.Array,  # [S] int32
    *,
    softmax_scale: float | None = None,
) -> jax.Array:
    """Standard (non-sparse) causal dense attention over the *entire* KV
    sequence. Used purely as an independent cross-check reference in tests:
    when ``topk`` is large enough to cover every causally-valid key,
    ``sparse_causal_attention`` must reduce to this.
    """
    _, _, qk_head_dim = q.shape
    if softmax_scale is None:
        softmax_scale = qk_head_dim**-0.5

    logits = jnp.einsum(
        "thd,sd->ths",
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    ) * softmax_scale
    causal = k_positions[None, :] <= q_positions[:, None]  # [T, S]
    logits = jnp.where(causal[:, None, :], logits, -jnp.inf)
    probs = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum(
        "ths,sd->thd",
        probs,
        v.astype(jnp.float32),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    return out.astype(v.dtype)
