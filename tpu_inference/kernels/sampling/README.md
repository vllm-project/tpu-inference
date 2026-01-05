# TPU Sampling Kernels

High-performance sampling kernels extracted from [tallax](https://github.com/oliverdutton/tallax).

## Performance

| Scenario | Batch | Speedup |
|----------|-------|---------|
| Large vocab (262K, top-k=64) | 16  | 15× avg |
| Large vocab (262K, top-k=64) | 128 | 75× avg |
| Speculative (32K, top-5)     | 16  | 15× vs XLA |

## Usage

```python
from tpu_inference.kernels.sampling import topk_topp_and_sample

tokens = topk_topp_and_sample(
    rng_key, logits, sampling_metadata,
    max_k=128, sampling_eps=1e-5, replace_val=-1e12
)
```

Enable via environment variable:
```bash
export PALLAS_SAMPLING_TOPK_THRESHOLD=128
```

## Algorithms

**Divide-and-Filter Top-K**: Partitions vocabulary, computes top-m per partition, filters unconverged bins. With 256 bins, bins-top-4 has >95% probability of containing top-128.

**Bitonic Top-K**: Compressed transpose format reduces cross-lane ops from 128 to 4.

## Files

- `divide_and_filter_topk.py` - Main top-k
- `top_p_and_sample.py` - Top-p + sampling
- `sampling.py` - Combined kernel
- `cumsum.py`, `gather.py` - Utilities

## Testing

```bash
pytest tests/kernels/sampling/
```
