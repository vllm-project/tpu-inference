# Inference Microbenchmark Script

This script is designed as a vLLM analog to MaxText's own [inference microbenchmark](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/inference_microbenchmark.py).  Specifically, it attempts to isolate and benchmark (with as little overhead as possible) prefill and decode for a given number of benchmark iterations (10 by default) after running a brief warmup. # TODO (jacobplatin): add chunked prefill?

Note that our main entrypoint to vLLM is the [`EngineCore`](https://github.com/vllm-project/vllm/blob/799397e/vllm/v1/engine/core.py#L55) class, rather than something higher level like the [`LLM`](https://github.com/vllm-project/vllm/blob/799397ee4f57b90ee1b5f12f88b12f4de0de0d1d/vllm/entrypoints/llm.py#L60) class, which gives us a bit more finegrained control, especially with accessing the TPU model runner and scheduler.

## Example Command

This command runs the benchmark with default settings for the `Llama-3.1-8B-Instruct` model.

```bash
TPU_BACKEND_TYPE=jax python scripts/inference/inference_microbenchmark.py --max-model-len 2048 --max-num-seqs 1 --max-num-batched-tokens 4096 --profile --profile-dir inference-microbenchmark --prefill-lengths "128, 256"
```

## Command-Line Options

This script accepts arguments specific to benchmarking but can also take in any vLLM-specific argument as well (note the defaults of a few the relevant settings below).

### Benchmark-Specific Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--prompt` | `str` | `"I love to"` | The initial prompt to use for the benchmark. This will be padded to the nearest multiple of the `prefill_len_padding` setting. |
| `--prefill-lengths` | `str` | `"128"` | A comma-separated string of sequence lengths to benchmark for the prefill phase. Each length must be a multiple of the padding size (currently 128, but specifically set as `prefill_len_padding`). |
| `--profile` | `flag` | `False` | If set, enables profiling for the first iteration of each benchmark (prefill and decode). |
| `--profile-dir` | `str` | `None` | The directory where the profiles will be saved. **Required if `--profile` is enabled.** |

### vLLM Engine Options

The following arguments are passed to the vLLM directly, but can also be overriden on the command line in typical vLLM fashion (e.g. adding `--dtype` to your command).

| Argument | Default (in script) | Description |
|---|---|---|
| `--model` | `"meta-llama/Llama-3.3-70B-Instruct"` | The name or path of the Hugging Face model to use. |
| `--tensor-parallel-size`| `8` | The number of devices to use for tensor parallelism. |
| `--max-num-seqs` | `1` | The maximum number of sequences in a batch. |
| `--max-model-len` | `1024` | The maximum total sequence length (prompt + generated tokens) the model can handle. |
| `--max-num-batched-tokens`| `8192` | The maximum number of tokens that can be processed in a single batch. |
| `--block-size` | `32` | The size of a block in the KV cache. |
