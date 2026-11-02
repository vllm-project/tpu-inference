# Kernel-side symmetric ALiBi for encoder-only attention

How `encoder_only_flash_attention` supports symmetric (encoder) ALiBi — used
by JinaBert / `jina-embeddings-v2` — without materializing an O(heads × T²)
bias tensor. Validated on TPU v6e at the full 8192-token context.

## Motivation

Symmetric ALiBi replaces position embeddings with an additive attention bias
`bias[h, i, j] = -slope_h * |i - j|`. The first implementation built this as
a dense `[num_heads, T, T]` float32 tensor and passed it to the flash kernel
via its `ab` operand. At `T = 8192` with 8 heads that is ~2.1 GB, rebuilt
inside every layer's attention call; XLA did not CSE the copies across the
`shard_map` boundaries, so warmup compilation of the 8192-token bucket needed
~16.9 GB of HLO temporaries and failed with `RESOURCE_EXHAUSTED`. That capped
`max_model_len` at ~2048.

## Design

Only the `[num_heads]` slope vector crosses the kernel boundary. The bias is
computed inside the Pallas kernel, per `[block_q, block_k]` tile, from block
indices — the same way the kernel already builds its causal mask:

```
row_ids = broadcasted_iota(rows) + q_seq_index * block_q
col_ids = broadcasted_iota(cols) + kv_seq_index * block_k_major + start_k
s -= slope_h * |row_ids - col_ids|          # after s *= sm_scale
```

Key properties:

- **No O(T²) memory.** Peak extra HBM is the slope operand:
  `num_heads × 8 × 128` floats (~4 KB). Warmup temporaries at 8192 drop from
  ~16.9 GB to well under 1 GB.
- **Correct for ragged batches.** The runner concatenates sequences into one
  token stream; cross-sequence pairs are masked by the kernel's segment IDs.
  For pairs within a sequence, global `|i - j|` equals within-sequence
  distance, so no per-sequence position offsets are needed.
- **Bias is added to the already-scaled logits** (`s *= sm_scale` first),
  matching the JinaBert reference formulation. This differs from the `ab`
  operand, which the kernel adds before scaling.
- **Slope operand layout.** Each head's slope is broadcast across a full
  `[NUM_SUBLANES, NUM_LANES]` (8 × 128) tile and block-fetched per head with
  block shape `(1, 8, 128)`. Mosaic requires the last two block dims to be
  multiples of 8/128 or equal to the array dims; a `[num_heads, 128]` operand
  with `(1, 128)` blocks violates the sublane rule and fails lowering. This
  mirrors the kv segment-ID operand layout.
- **TP-compatible.** `sharded_encoder_only_attention` shards the slope vector
  with the heads (`P("model")`), so each shard sees its local heads' slopes.
- **Both kernel paths covered:** the multi-step kv loop
  (`_flash_attention_kernel_single_batch`, col offset
  `kv_seq_index * block_k_major + start_k`) and the single-step kernel
  (`_flash_attention_kernel_single_batch_single_step`, col offset 0 since the
  kv block spans the whole sequence).

The dense `ab` path is unchanged and still used for sliding windows; ALiBi
and `ab` compose (window mask pre-scale, ALiBi post-scale).

## API

```python
encoder_only_flash_attention(q, k, v, seq_lens, alibi_slopes, ...)
# alibi_slopes: [num_heads] float32, or None (behavior unchanged).
```

`flash_attention(..., alibi_slopes=None)` accepts the same optional argument;
`attention_interface.encoder_only_attention(..., alibi_slopes=...)` is the
model-facing entry point (see `models/jax/jina_bert.py` for usage, including
the standard `2^(-8(h+1)/n)` slope schedule with the non-power-of-two head
extension).

## Validation

- **Exactness of tile indexing:** Pallas interpret mode (CPU, exact f32
  arithmetic) matches a dense reference to ~1e-6 across multi-step,
  single-step, ragged multi-sequence, and non-power-of-two-head cases.
- **On TPU:** `tests/kernels/encoder_alibi_kernel_test.py` compares against a
  `Precision.HIGHEST` dense reference. The kernel's f32 dots run at default
  MXU precision (bf16 multiply passes), which alone yields ~1e-3..1e-2
  elementwise noise after softmax — the no-ALiBi control case shows the same
  deltas. Tests therefore gate on cosine similarity > 0.9999 and
  mean |diff| < 2e-3, with a 2e-2 elementwise outlier cap. A real indexing
  bug shifts logits by O(slope × block_size) and fails these immediately.
- **End to end:** JinaBert model parity vs the official ONNX export
  (`tests/models/jax/test_jina_bert.py`, cosine > 0.999) and serving at
  `--max-model-len 8192` (see `docs/models/jina_embeddings_v2.md`).

## Serving note: `--max-num-batched-tokens`

Encoder/pooling models cannot chunk prefill (no KV cache; every token attends
to the whole sequence in one pass), so a prompt must fit within one
scheduling step's token budget, and the runner's precompiled shape buckets
are derived from the same value. Serve with
`--max-num-batched-tokens >= --max-model-len` (equal is the natural choice).
The budget also lets multiple short requests batch into one forward pass.

## Follow-ups

- Fold `sm_scale` and slope negation into a precomputed per-head constant to
  save one VPU multiply per tile (micro-optimization).
- The sliding-window path could reuse the same tile-side construction and
  drop its dense `ab` tensor too.
