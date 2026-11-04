# Serving jina-embeddings-v2 (JinaBert) on TPU

tpu-inference includes a JAX-native implementation of JinaBert, the encoder
of the [jina-embeddings-v2](https://huggingface.co/jinaai/jina-embeddings-v2-small-en)
models (a BERT variant with symmetric ALiBi instead of position embeddings,
GEGLU feed-forward and post-LayerNorm). It runs under vLLM's pooling runner
and serves two kinds of checkpoints:

| `architectures` in `config.json`       | Serve with                            | Endpoint         |
| -------------------------------------- | ------------------------------------- | ---------------- |
| `JinaBertForMaskedLM`, `JinaBertModel` | `--runner pooling --convert embed`    | `/v1/embeddings` |
| `JinaBertForSequenceClassification`    | `--runner pooling --convert classify` | `/classify`      |

Validated on TPU v6e with one chip (TP=1) and `--dtype float32`:
`jinaai/jina-embeddings-v2-small-en` over its full 8192-token context, and a
fine-tuned small-en prompt classifier.

## Install

Follow the [installation guide](../getting_started/installation.md); nothing
else is needed. tpu-inference registers the JinaBert architectures with
vLLM's model registry itself (in every vLLM process, including the API
server), and the checkpoints load as a standard `BertConfig`, so do **not**
pass `--trust-remote-code`: the remote Jina code is not used, and it imports
modules that transformers v5 no longer has.

## Embeddings

```bash
vllm serve jinaai/jina-embeddings-v2-small-en \
  --runner pooling --convert embed --dtype float32
```

`--convert embed` is required: vLLM classifies the `JinaBertForMaskedLM`
architecture string as masked-LM and gates the embeddings API otherwise. No
pooler flag is needed: vLLM picks up mean pooling from the repo's
sentence-transformers configuration.

```bash
curl localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model": "jinaai/jina-embeddings-v2-small-en", "input": "How is the weather today?"}'
```

The response holds a 512-dimensional, L2-normalized embedding.

## Sequence classification

A fine-tuned classifier (encoder, mean pooling, linear head) is a checkpoint
whose `config.json` sets `"architectures":
["JinaBertForSequenceClassification"]` and `num_labels` (`id2label` is
optional), with the head stored as `classifier.weight` `[num_labels, hidden]`
and `classifier.bias` `[num_labels]` next to the encoder weights.

```bash
vllm serve /path/to/jina-classifier \
  --runner pooling --convert classify --dtype float32 --max-model-len 1024
```

`--convert classify` is required: vLLM resolves `--convert auto` to `none`
for this registered architecture, and the server would then load the encoder
without its head (a warning says so at startup).

```bash
curl localhost:8000/classify -H 'Content-Type: application/json' \
  -d '{"model": "/path/to/jina-classifier", "input": ["How is the weather today?"]}'
```

Each result has `probs` (softmax over the labels) and `label`; requests with
`"use_activation": false` get the raw logits instead.

The head is kept in float32 and applied at `HIGHEST` matmul precision
whatever `--dtype` is. Mean pooling, the head and the softmax run on the TPU,
fused into the model step for steps with at most 32 requests, so only
`[num_requests, num_labels]` is copied to the host. Steps the device path
does not handle (pooler configs with `logit_mean` / `logit_sigma`
calibration, other pooling tasks) use vLLM's CPU classify pooler with the
same head.

For scores that match a PyTorch float32 reference to about `1e-6` (useful
when scores sit close to a decision threshold), set
`JINA_BERT_MEGAKERNEL_PRECISION=highest` and keep
`--max-num-batched-tokens 2048`, so that every step runs the megakernel at
full float32 precision (see the next section).

## Long context and the token budget

The encoder runs as one Pallas megakernel for every step of at most 2048
tokens, and layer by layer (XLA matmuls and the encoder flash-attention
kernel, which computes the ALiBi bias per tile, see
[the design notes](../developers_guide/encoder_alibi_kernel.md)) for longer
steps. The full 8192-token context therefore needs no extra flags.

`--max-num-batched-tokens` is the token budget of one step. Encoders cannot
split a prompt across steps, so it must be at least `max_model_len` (vLLM
sets it that way by default), and it is also the largest shape the runner
precompiles. A larger budget lets several prompts share one step, which is
the main throughput lever for long inputs; a budget of 2048 keeps every step
on the megakernel. The per-layer path uses XLA's default matmul precision
(bf16 MXU passes, float32 accumulation, like the megakernel's `default`
precision); `JINA_BERT_MEGAKERNEL_PRECISION=highest` only applies to
megakernel steps.

## Configuration

Every speed-up has a switch; unset means the default.

| Environment variable             | Default   | Other values                                                                                       |
| -------------------------------- | --------- | -------------------------------------------------------------------------------------------------- |
| `USE_JINA_BERT_MEGAKERNEL`       | `1`       | `0`: per-layer path for every step                                                                 |
| `JINA_BERT_MEGAKERNEL_PRECISION` | `default` | `highest`: full-fp32 matmuls (default: bf16 MXU operands, fp32 accumulation, as the per-layer path) |
| `JINA_BERT_MEGAKERNEL_VERSION`   | `v2`      | `v1`: the first megakernel                                                                         |
| `JINA_BERT_DEVICE_POOLING`       | `1`       | `0`: copy the step's hidden states to the host and pool (and classify) with vLLM's CPU pooler     |
| `JINA_BERT_FUSED_POOLING`        | `1`       | `0`: run device pooling as a separate dispatch after the forward pass                              |
| `TPU_POOLING_FAST_PATH`          | `1`       | `0`: build sampling metadata every step and run vLLM's CPU pooler under torchax's dispatch modes  |

With device pooling, per-request mean pooling (and the L2 normalization of
embeddings) runs on the TPU and only `[num_requests, hidden]` is copied to
the host instead of `[num_tokens, hidden]`. With fused pooling, that
reduction runs inside the model step's XLA program for steps with at most
32 requests, so a step needs a single dispatch. With
`JINA_BERT_DEVICE_POOLING=0`, `TPU_POOLING_FAST_PATH=1` still runs vLLM's
pooler as plain torch on the host copy (bitwise identical, about 1.4 ms less
per step).

## Validate

```bash
pytest tests/kernels/encoder_alibi_kernel_test.py -v          # ALiBi flash attention vs a dense reference
pytest tests/kernels/jina_bert_megakernel_test.py -v -rs       # megakernel vs a dense reference
pytest tests/models/jax/test_jina_bert.py -v -rs               # parity vs the official ONNX export
pytest tests/models/jax/test_jina_bert_megakernel.py -v -rs -s # megakernel vs the per-layer path
pytest tests/e2e/test_jina_embeddings.py -v -rs                # engine: embeddings and classification
pytest tests/models/common/test_pooling.py tests/models/common/test_pooling_classify.py \
  tests/models/vllm/test_jina_bert_compat.py tests/test_envs.py  # no TPU needed
```

The parity test compares per-token hidden states and mean-pooled embeddings
with the model repo's official ONNX export (same weights, float32) and
requires cosine similarity above 0.999. The megakernel tests compare both
kernel versions at both matmul precisions with the per-layer path and a
dense reference (see the kernel README).

## Measure

With vLLM stopped (the script needs the chip), from the repo root:

```bash
python scripts/benchmarking/jina/measure_jina_forward.py --output /tmp/after.json
```

It times the jitted forward pass (embeddings and encoder) on the real
checkpoint for 1x1024, 2x1024 packed and 1x2048 tokens: the megakernel (each
kernel version) and the per-layer path, at both matmul precisions, and for
the default megakernel the forward pass plus host copy and vLLM's embed
pooler for each pooling path. The JSON has the median, p90, mean and min in
ms. Useful flags: `--cases 1x2048`, `--precisions default`,
`--paths megakernel`, `--versions v2`, `--no-pooling`, `--runs 500` and
`--trace-dir /tmp/jina-forward-trace` (a `jax.profiler` trace).

For a device trace of a live server, start it with
`--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/tmp/jina-trace", "ignore_frontend": true}'`,
put it under load, and bracket a few seconds with
`curl -X POST localhost:8000/start_profile` and
`curl -X POST localhost:8000/stop_profile`. Open the trace with xprof or
TensorBoard; see [Profiling](../developers_guide/profiling.md).

## Known limitations

- The megakernel and device pooling run on a single chip (TP=1). Larger
  meshes use the per-layer path and vLLM's CPU pooler.
- The megakernel is float32-only; other dtypes use the per-layer path, and
  bf16 accuracy has not been validated against the float32 baseline.
- The runner still allocates a KV cache for this encoder-only model, which
  it never uses.
