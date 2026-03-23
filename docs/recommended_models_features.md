# Models and Features

This table shows what hardware generations support which models and feature sets, allowing developers to pick the optimal platform for execution.

*Last Updated: 2026-03-23 07:57 PM UTC*

## Recommended Models
<!-- START: model_support -->
| Model | Type | Unit&nbsp;Test | Correctness&nbsp;Test | Benchmark |
| --- | --- | --- | --- | --- |
| [deepseek-ai/DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [meta-llama/Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) | Multimodal | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [meta-llama/Llama-Guard-4-12B/Multimodal](https://huggingface.co/meta-llama/Llama-Guard-4-12B/Multimodal) | Multimodal | <span title="✅ Passing">✅</span> | <span title="❌ Failing">❌</span> | <span title="❓ Untested">❓</span> |
| [meta-llama/Llama-Guard-4-12B/Text-Only](https://huggingface.co/meta-llama/Llama-Guard-4-12B/Text-Only) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Multimodal | <span title="✅ Passing">✅</span> | <span title="❌ Failing">❌</span> | <span title="❓ Untested">❓</span> |
| [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Multimodal | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |

<!-- END: model_support -->

## Recommended Features
<!-- START: core_features -->
<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Flax</th>
      <th>Torchax</th>
      <th>Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>async scheduler</td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
    </tr>
    <tr>
      <td>Chunked Prefill</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>DCN-based P/D disaggregation</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>hybrid kv cache</td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
    </tr>
    <tr>
      <td>KV cache host offloading</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>LoRA_Torch</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>multi-host</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>Multimodal Inputs</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>Out-of-tree model support</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>Prefix Caching</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>runai_model_streamer_loader</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>sampling_params</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>Single Program Multi Data</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>Single-Host-P-D-disaggregation</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>Speculative Decoding: Eagle3</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>Speculative Decoding: Ngram</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>structured_decoding</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>
<!-- END: core_features -->

## Kernel Support
<!-- START: kernel_support -->
<!-- END: kernel_support -->

## Parallelism Support
<!-- START: parallelism -->
<table>
  <thead>
    <tr>
      <th rowspan="2" width="150" style="text-align:left">Feature</th>
      <th colspan="2">Flax</th>
      <th colspan="2">Torchax</th>
    </tr>
    <tr>
      <th>Single-host</th>
      <th>Multi-host</th>
      <th>Single-host</th>
      <th>Multi-host</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CP</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>DP</td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>EP</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>PP</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>SP&nbsp;(<a href="https://github.com/vllm-project/tpu-inference/issues/1749">vote&nbsp;to&nbsp;prioritize</a>)</td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>TP</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>
<!-- END: parallelism -->

## Quantization Support
<!-- START: quantization -->
<table>
  <thead>
    <tr>
      <th>Checkpoint dtype</th>
      <th>Method</th>
      <th>Supported<br>Hardware Acceleration</th>
      <th>flax</th>
      <th>torchax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AWQ INT4</td>
      <td></td>
      <td>v5, v6</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>FP4 W4A16</td>
      <td>mxfp4</td>
      <td>v7</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>FP8 W8A8</td>
      <td>compressed-tensor</td>
      <td>v7</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>FP8 W8A16</td>
      <td>compressed-tensor</td>
      <td>v7</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>INT4 W4A16</td>
      <td>awq</td>
      <td>v5, v6</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>INT8 W8A8</td>
      <td>compressed-tensor</td>
      <td>v5, v6</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>

> **Note:**
> &bull;&nbsp;&nbsp;&nbsp;*This table only tests checkpoint loading compatibility.*
<!-- END: quantization -->
