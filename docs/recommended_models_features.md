# Recommended Model and Feature Matrices

Although vLLM TPU’s new unified backend makes out-of-the-box high performance serving possible with any model supported in vLLM, the reality is that we're still in the process of implementing a few core components.
For this reason, until we land more capabilities, we recommend starting from this list of stress tested models and features below.

We are still landing components in tpu-inference that will improve performance for larger scale, higher complexity models (XL MoE, +vision encoders, MLA, etc.).

If you’d like us to prioritize something specific, please submit a GitHub feature request [here](https://github.com/vllm-project/tpu-inference/issues/new/choose).

## Recommended Models

These tables show the models currently tested for accuracy and performance.

### Models

| Model | Type | UnitTest | Accuracy/Correctness | Benchmark |
| :--- | :--- | :---: | :---: | :---: |
| `google/gemma-3-27b-it` | Text | ✅ | ✅ | ✅ |
| `meta-llama/Llama-3.1-8B-Instruct` | Text | ✅ | ✅ | ✅ |
| `meta-llama/Llama-3.3-70B-Instruct` | Text | ✅ | ✅ | ✅ |
| `meta-llama/Llama-Guard-4-12B` | Text | ✅ | ✅ | ✅ |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Multimodal | ✅ | ✅ | ✅ |
| `Qwen/Qwen3-30B-A3B` | Text | ✅ | ✅ | ✅ |
| `Qwen/Qwen3-32B` | Text | ✅ | ✅ | ✅ |
| `Qwen/Qwen3-4B` | Text | ✅ | ✅ | ✅ |
| `deepseek-ai/DeepSeek-V3.1` | Text |  unverified |  unverified |  unverified |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | Multimodal |  unverified |  unverified |  unverified |
| `moonshotai/Kimi-K2-Thinking` | Text |  unverified |  unverified |  unverified |
| `openai/gpt-oss-120b` | Text |  unverified |  unverified |  unverified |
| `Qwen/Qwen3-30B-A3B-Instruct` | Multimodal |  unverified |  unverified |  unverified |
| `Qwen/Qwen3-Coder-480B-A35B-Instruct` | Text |  unverified |  unverified |  unverified |

## Recommended Features

This table shows the features currently tested for accuracy and performance.

| Feature | CorrectnessTest | PerformanceTest |
| :--- | :---: | :---: |
| `async scheduler` | ✅ | ✅ |
| `Chunked Prefill` | ✅ | ✅ |
| `data_parallelism` | ✅ |  unverified |
| `DCN-based P/D disaggregation` |  unverified | ✅ |
| `KV cache host offloading` |  unverified |  unverified |
| `LoRA_Torch` | ✅ | ✅ |
| `Multimodal Inputs` | ✅ | ✅ |
| `Out-of-tree model support` | ✅ | ✅ |
| `Prefix Caching` | ✅ | ✅ |
| `runai_model_streamer_loader` | ✅ | N/A |
| `sampling_params` | ✅ | N/A |
| `Single Program Multi Data` | ✅ | ✅ |
| `Single-Host-P-D-disaggregation` | N/A | N/A |
| `Speculative Decoding: Eagle3` | ✅ | ✅ |
| `Speculative Decoding: Ngram` | ✅ | ✅ |
| `structured_decoding` | ✅ | N/A |

## Kernel Support

This table shows the current kernel support status.

| Feature | CorrectnessTest | PerformanceTest |
| :--- | :---: | :---: |
| `Ragged Paged Attention V3` | ✅ | ✅ |
| `Collective Communication Matmul` | ✅ |  unverified |
| `MLA` (Multi-Head Latent Attention) |  unverified |  unverified |
| `MoE` (Mixture of Experts) |  unverified |  unverified |
| `Quantized Attention` |  unverified |  unverified |
| `Quantized KV Cache` |  unverified |  unverified |
| `Quantized Matmul` |  unverified |  unverified |

## Parallelism Support

This table shows the current parallelism support status.

| Feature | CorrectnessTest | PerformanceTest |
| :--- | :---: | :---: |
| `PP` (Pipeline Parallelism) | ✅ | ✅ |
| `DP` (Data Parallelism) | ✅ |  unverified |
| `EP` (Expert Parallelism) | ✅ |  unverified |
| `TP` (Tensor Parallelism) | ✅ |  unverified |
| `CP` (Context Parallelism) |  unverified |  unverified |
| `SP` (Sequence Parallelism) |  unverified |  unverified |

## Quantization Support

This table shows the current quantization support status.

| Feature | Recommended TPU Generations | CorrectnessTest | PerformanceTest |
| :--- | :--- | :---: | :---: |
| `FP8 W8A8` | v7 |  unverified |  unverified |
| `FP8 W8A16` | v7 |  unverified |  unverified |
| `FP4 W4A16` | v7 |  unverified |  unverified |
| `INT8 W8A8` | v5, v6 |  unverified |  unverified |
| `INT4 W4A16` | v5, v6 |  unverified |  unverified |
| `AWQ INT4` | v5, v6 |  unverified |  unverified |
