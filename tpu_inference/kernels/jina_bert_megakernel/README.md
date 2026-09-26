# JinaBert v2 encoder megakernel

All encoder layers of `jinaai/jina-embeddings-v2-small-en` (4 layers,
hidden 512, 8 heads of 64, GeGLU MLP with intermediate 2048, post-LN,
symmetric ALiBi) run as one Pallas TPU kernel: one launch per step for the
whole encoder. `JinaBertEncoder` in `tpu_inference/models/jax/jina_bert.py`
uses it by default; embeddings and mean pooling stay outside the kernel.

| Environment variable             | Default   | Meaning                        |
| -------------------------------- | --------- | ------------------------------ |
| `USE_JINA_BERT_MEGAKERNEL`       | `1`       | `0` selects the per-layer path |
| `JINA_BERT_MEGAKERNEL_PRECISION` | `default` | matmul precision, see below    |

The per-layer path (XLA matmuls and the encoder flash-attention kernel) is
kept unchanged. The model also falls back to it, with a warning, where the
megakernel does not apply: see [Known limits](#known-limits).

## Design

A megakernel in the sense of <https://inferact.ai/blog/tpu-megakernels>.
The design is inspired by that post and by
[Inferact/tpu-megakernels](https://github.com/Inferact/tpu-megakernels)
(Apache-2.0); no code is taken from either.

- **One grid-less `pallas_call`.** The residual stream `[T, 512]` is
  copied into VMEM once and stays there across all layers; apart from small
  per-step metadata, the only HBM traffic is the input, the output and the
  weights.
- **Weight streaming.** Each layer's weights are copied HBM -> VMEM with
  async DMAs into a 2-slot buffer; layer `l + 1` is prefetched while layer
  `l` computes.
- **Two passes per layer** over token tiles (128 rows if `T <= 128`, else
  256):
  1. QKV projection with a fused `[512, 1536]` weight. The softmax scale
     `1/sqrt(64) = 2^-3` is folded into the Q columns and bias (exact).
  2. Per query tile: flash-style online softmax over only the KV tiles that
     can hold tokens of the same requests (per-tile `[kv_lo, kv_hi)` bounds
     in SMEM). The ALiBi bias and the same-request mask are computed in the
     kernel from iotas and segment ids, so there is no `O(T^2)` bias tensor
     in HBM. Then output projection + residual + LayerNorm, the GeGLU MLP in
     512-column chunks + residual + LayerNorm, written back in place.
- **Packed steps.** Segment ids come from `seq_lens` exactly as in the
  per-layer path (requests back to back from token 0, zero entries are
  unused slots, tokens past `sum(seq_lens)` are padding). Tiles after the
  last real token are skipped; rows at padding positions are unspecified.
- **Heads in 128-lane pairs.** Q is stored per head, zeroed outside that
  head's 64 lanes, so `Q K^T` contracts over an aligned 128-lane slice (the
  zero lanes cost nothing extra on the MXU) without unaligned lane slicing.

The packed weight layout (`pack_jina_bert_weights`) is built once after
checkpoint loading, in `JinaBertForMaskedLM.load_weights`. It is a derived
copy; the checkpoint params and their loading are unchanged. If
`load_weights` was bypassed (e.g. `--load-format dummy`), the weights are
packed inside the step instead.

## Matmul precision

Parameters and activations are float32. The matmul precision is explicit:

- **`default`**: MXU operands are rounded to bfloat16, one MXU pass, float32
  accumulation. This is what `lax.Precision.DEFAULT` does to float32
  matmuls on TPU, so it matches the per-layer path's XLA matmuls. It
  applies to every matmul: QKV, `Q K^T`, `P V`, output projection and both
  MLP matmuls. The packed matmul weights are stored in bfloat16.
- **`highest`**: float32 operands with `lax.Precision.HIGHEST`
  (`#tpu.contract_precision<fp32>`), i.e. full-fp32 matmuls.

In both modes the residual stream, biases, LayerNorm (eps 1e-12), softmax,
GELU and all accumulations are float32. GELU is the exact erf form, with
erf from Abramowitz & Stegun 7.1.26 (`|error| <= 1.5e-7`).

Measured in the Pallas interpreter against a dense float32 reference
(`reference.py`, random weights at realistic scales, up to 600 tokens):

| Precision | Max abs error | Min per-token cosine | Min pooled cosine |
| --------- | ------------- | -------------------- | ----------------- |
| `default` | 7.9e-2        | 0.99977              | 0.99989           |
| `highest` | 1.8e-5        | 1.0000000            | 1.0000000         |

## VMEM budget

From `estimate_vmem_bytes`, for the kernel's own buffers (MiB):

| Tokens | `default` | `highest` |
| ------ | --------- | --------- |
| 128    | 18.9      | 35.6      |
| 512    | 23.6      | 41.9      |
| 1024   | 26.9      | 47.2      |
| 2048   | 33.4      | 57.7      |

The double-buffered weights take 16 MiB (`default`) or 32 MiB (`highest`)
at every size. At 2048 tokens the residual stream, Q, K and V take 12 MiB
or 20 MiB. The kernel asks Mosaic for `min(need + 32 MiB, 85% of VMEM)`,
which is well within v6e's 128 MiB.

## Known limits

- **At most 2048 tokens per step.** Longer steps raise `ValueError` at trace
  time. Serve with `--max-num-batched-tokens 2048` (and
  `--max-model-len <= 2048`).
- **Where the model falls back to the per-layer path**, with a warning:
  - a mesh with more than one device (the kernel is single-chip, TP=1);
  - a model dtype other than float32;
  - head geometries other than an even number of 64-wide heads, or an
    intermediate size that is not a multiple of 512;
  - models whose buffers would not fit VMEM at 2048 tokens.
- Attention work is still quadratic within a request; only tile pairs from
  different requests are skipped.

## Tests

- `tests/kernels/jina_bert_megakernel_test.py`: kernel vs the dense
  reference, input validation, tile metadata, packing, VMEM. Small cases
  also run on CPU in the Pallas interpreter.
- `tests/models/jax/test_jina_bert_megakernel.py`: megakernel vs the
  per-layer path on the real checkpoint, with the same weights and inputs.
  It covers single sequences of 1 to 2048 tokens and packed steps, and
  reports max abs error and min per-token and pooled cosine (TPU only).
- `tests/models/jax/test_jina_bert.py`: ONNX parity. It now runs through the
  megakernel by default.
