# OLMo 2

The native JAX implementation supports OLMo 2 checkpoints, including
[allenai/OLMo-2-0425-1B](https://modelscope.cn/models/allenai/OLMo-2-0425-1B).
It uses joint Q/K projection normalization and normalization after the attention
and feed-forward sublayers. The standard checkpoints use bias-free attention and
128-dimensional heads. Configurations requiring padded attention heads or
attention biases are not supported.

## Serve

```bash
MODEL_IMPL_TYPE=flax_nnx vllm serve allenai/OLMo-2-0425-1B \
    --dtype bfloat16 --max-model-len 4096
```

To download the checkpoint from ModelScope instead of Hugging Face:

```bash
VLLM_USE_MODELSCOPE=True MODEL_IMPL_TYPE=flax_nnx \
    vllm serve allenai/OLMo-2-0425-1B --dtype bfloat16 --max-model-len 4096
```

The ModelScope download option requires the `modelscope` Python package. A local
checkpoint directory can also be passed as the model argument.

## Validate

The unit tests create small random Transformers models without downloading
weights. They compare prefill and incremental decode logits, exercise grouped
query attention and tied embeddings, and verify pipeline weight loading.
TPU tests also compare the real paged attention kernel against Transformers.

To include the pretrained checkpoint comparison on a single TPU:

```bash
OLMO2_TEST_MODEL=allenai/OLMo-2-0425-1B \
    python -m pytest -s tests/models/jax/test_olmo2.py
```

`OLMO2_TEST_MODEL` also accepts a local ModelScope checkpoint directory. The
checkpoint test compares BF16 JAX with FP32 Transformers and checks that greedy
prefill and decode predictions agree. It also compares output distributions,
using Transformers' own BF16 error as a rounding baseline, since a uniform logit
shift does not change sampling probabilities.
