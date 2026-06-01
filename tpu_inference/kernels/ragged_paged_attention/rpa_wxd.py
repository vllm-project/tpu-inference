"""A from-scratch, simplest-possible attention kernel in Pallas.

Goal: learn Pallas by building up to ragged paged attention (RPA) one step at a
time. This file is STEP 1 -- the absolute minimum:

  * plain `pl.pallas_call` (no `emit_pipeline`, no flash attention)
  * grid over query blocks, batch, and heads
  * inside the kernel: scores = q @ k^T, causal mask, softmax, out = p @ v
  * PREFILL only: q, k, v are the *new* tokens. We do NOT read or write a KV
    cache, and we do NOT do paging yet.

We also start WITHOUT raggedness: every request has the same length, so it's
really just batched causal self-attention. Raggedness (variable-length requests
packed together) and paging are added in later steps -- see the roadmap at the
bottom of this file.

------------------------------------------------------------------------------
PALLAS IN 60 SECONDS
------------------------------------------------------------------------------
A Pallas kernel runs on the TPU over a `grid` -- think nested for-loops. For
each grid point, Pallas copies a *block* of each input array from HBM (the big
slow memory) into VMEM (small fast memory), runs your Python kernel function on
those blocks, then copies the output block back to HBM.

  * `grid`        : a tuple of loop bounds, e.g. (num_q_blocks, batch, heads).
  * `BlockSpec`   : for each array, says (a) the block shape to bring into VMEM,
                    and (b) an `index_map` from grid indices -> which block.
  * kernel(refs)  : your function. It receives `Ref`s (pointers into VMEM). You
                    read with `ref[...]` and write with `ref[...] = value`.
  * `pl.program_id(axis)` : the current index along a grid axis (the loop var).

------------------------------------------------------------------------------
THE (8, 128) TILING RULE  (the #1 beginner gotcha)
------------------------------------------------------------------------------
TPU VMEM is physically tiled: the LAST TWO axes of every block map onto
(sublanes=8, lanes=128). So Pallas requires each block's last two axes to be
either divisible by (8, 128), or equal to the full array dimension. Practically:
put `head_dim` (=128) last so it fills lanes, and a token axis (multiple of 8,
e.g. 128) second-to-last so it fills sublanes.

That's why we use array layout [batch, heads, seq, head_dim] below: the last two
axes are (seq, head_dim) -> a clean tile. (An earlier draft used
[batch, seq, heads, d]; the block's last two axes became (heads=1, d) which
violates the rule and Pallas rejects it.)
"""

import functools
import math

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu  # noqa: F401  (used in later steps)


def create_inputs(batch, num_heads, seq_len, head_dim):
    """Prefill inputs: q, k, v are the new tokens for `batch` equal-length seqs.

    Shape convention: [batch, num_heads, seq_len, head_dim] -- last two axes
    (seq_len, head_dim) form the TPU (sublane, lane) tile (see the tiling note).
    """
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(0), 3)
    shape = (batch, num_heads, seq_len, head_dim)
    q = jax.random.normal(k0, shape, jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(k1, shape, jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(k2, shape, jnp.float32).astype(jnp.bfloat16)
    return q, k, v


def attention_kernel(q_ref, k_ref, v_ref, o_ref, m_scratch, l_scratch, acc_scratch, *, scale, bq, bkv, num_kv_blocks):
    """Compute attention for ONE query block of ONE (batch, head).

    With array layout [batch, heads, seq, d] and the BlockSpecs below, Pallas
    hands us these block shapes:
      q_ref : (1, 1, bq,      d)   one q block, one batch, one head
      k_ref : (1, 1, seq_len, d)   ALL keys for that (batch, head)
      v_ref : (1, 1, seq_len, d)   ALL values for that (batch, head)
      o_ref : (1, 1, bq,      d)   where we write this block's output

    Because we load the *entire* K and V for the sequence, we never need an
    online/streaming softmax (the "flash" trick we're deliberately skipping). We
    materialize the full score row for the q block, softmax it in one shot, then
    do the weighted sum. Simple, at the cost of VMEM.
    """
    q_block = pl.program_id(2)
    kv_block = pl.program_id(3)

    @pl.when(kv_block == 0)
    def _():
        m_scratch[...] = jnp.full_like(m_scratch, -jnp.inf)
        l_scratch[...] = jnp.full_like(l_scratch, 0.0)
        acc_scratch[...] = jnp.full_like(acc_scratch, 0.0)

    # Squeeze the singleton batch/head axes -> 2D matrices for clean matmuls.
    # Cast to f32: softmax (exp / sum) wants more precision than bf16.
    q = q_ref[0, 0, :, :].astype(jnp.float32)   # (bq, d)
    k = k_ref[0, 0, :, :].astype(jnp.float32)   # (bkv, d)
    v = v_ref[0, 0, :, :].astype(jnp.float32)   # (bkv, d)

    scores = jax.lax.dot_general(
        q, k, (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    scores = scores * scale

    row_ids = q_block * bq + jax.lax.broadcasted_iota(jnp.int32, scores.shape, 0)
    col_ids = kv_block * bkv + jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    causal = col_ids <= row_ids
    scores = jnp.where(causal, scores, -jnp.inf)
    
    prev_max = m_scratch[...]
    running_max = jnp.maximum(prev_max, jnp.max(scores, axis=1, keepdims=True))
    correction = jnp.exp(prev_max - running_max)  
    m_scratch[...] = running_max
    p = jnp.exp(scores - running_max)
    l_scratch[...] = correction * l_scratch[...] + jnp.sum(p, axis=1, keepdims=True)             # (bq, 1)

    # Weighted sum of values: out[i] = sum_j p[i,j] * v[j]   -> (bq, d)
    pv = jax.lax.dot_general(
        p, v, (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_scratch[...] = correction * acc_scratch[...] + pv

    @pl.when(kv_block == num_kv_blocks - 1)
    def _():
        # Write the block back. o_ref expects (1, 1, bq, d).
        o_ref[0, 0, :, :] = (acc_scratch[...] / l_scratch[...]).astype(jnp.bfloat16)

@jax.jit
def attention(q, k, v):
    batch, num_heads, seq_len, head_dim = q.shape
    bq = 128                             
    bkv = 128
    num_q_blocks = seq_len // bq
    num_kv_blocks = seq_len // bkv
    scale = 1.0 / math.sqrt(head_dim)

    grid = (batch, num_heads, num_q_blocks, num_kv_blocks)

    q_spec = pl.BlockSpec((1, 1, bq, head_dim),
                          lambda b, h, qb, kvb: (b, h, qb, 0))
    kv_spec = pl.BlockSpec((1, 1, bkv, head_dim),
                           lambda b, h, qb, kvb: (b, h, kvb, 0))
    o_spec = pl.BlockSpec((1, 1, bq, head_dim),
                          lambda b, h, qb, kvb: (b, h, qb, 0))
    return pl.pallas_call(
        functools.partial(attention_kernel, scale=scale, bq=bq, bkv=bkv, num_kv_blocks=num_kv_blocks),
        grid=grid,
        in_specs=[q_spec, kv_spec, kv_spec],
        out_specs=o_spec,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        scratch_shapes=[
            pltpu.VMEM((bq, 1), jnp.float32),          # m  (running max)
            pltpu.VMEM((bq, 1), jnp.float32),          # l  (running denominator)
            pltpu.VMEM((bq, head_dim), jnp.float32),   # acc (running numerator)
        ],

    )(q, k, v)


def reference_attention(q, k, v):
    """Plain JAX causal attention, used to validate the kernel."""
    b, h, s, d = q.shape
    scale = 1.0 / math.sqrt(d)
    qf, kf, vf = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    # scores: [b, h, sq, sk]
    scores = jnp.einsum("bhqd,bhkd->bhqk", qf, kf) * scale
    causal = jnp.tril(jnp.ones((s, s), bool))
    scores = jnp.where(causal[None, None], scores, -jnp.inf)
    p = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", p, vf)
    return out.astype(q.dtype)


if __name__ == "__main__":
    q, k, v = create_inputs(batch=2, num_heads=8, seq_len=512, head_dim=128)

    out = attention(q, k, v)
    ref = reference_attention(q, k, v)

    diff = jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))
    print("output shape :", out.shape)
    print("max abs diff :", float(diff.max()))
    print("mean abs diff:", float(diff.mean()))

