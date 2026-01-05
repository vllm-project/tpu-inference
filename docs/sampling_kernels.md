# Sampling Performance Optimization

vLLM TPU includes optimized kernels for large vocabulary models with 15-75× speedup.

## Quick Start

```bash
export FLASH_SAMPLING_TOPK_THRESHOLD=128
vllm serve meta-llama/Llama-3.1-8B
```

## Performance

| Batch | Speedup |
|-------|---------|
| 16    | 15× avg |
| 128   | 75× avg |

## Configuration

### `FLASH_SAMPLING_TOPK_THRESHOLD`

**Default**: 0 (disabled)
**Recommended**: 64 or 128

Enables fast kernel when all batch top_k ≤ threshold.

**Use when**:
- ✅ Vocabulary > 100K tokens
- ✅ Consistent top_k across batch
- ✅ Batch size ≥ 16

**Avoid when**:
- ❌ Variable top_k per request
- ❌ Small vocabularies

```bash
# Enable for large vocab models
export FLASH_SAMPLING_TOPK_THRESHOLD=128
```

See [kernel README](../tpu_inference/kernels/sampling/README.md) for details.
