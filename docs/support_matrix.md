## TPU Support Matrix Dashboard

Below is the live status of our supported models, features, and kernels. Click on any category to expand the detailed support table. It is automatically updated from our detailed [Support Matrices](https://github.com/vllm-project/tpu-inference/tree/main/support_matrices).

*Last Updated: 2026-06-18 03:34 PM UTC*

<details open markdown="1">
<summary> <b>🚦 <i>Status Legend</i> </b> </summary>

> - ✅ **Passing:** Tested and works as expected. Ready for use.
> - ❌ **Failing:** Known to be broken or not functional. Help is wanted to fix this!
> - 🧪 **Experimental:** Works, but unoptimized or pending community validation.
> - 📝 **Planned:** Not yet implemented, but on the official roadmap.
> - ⛔️ **Unplanned:** There is no benefit to adding this.
> - ❓ **Untested:** The functionality exists but has not been recently or thoroughly verified.
>
> <details>
> <summary> <b>📐 <i>View Matrix Aggregation Rules (v6e/v7x & C+P)</i></b> </summary>
>
> - **🛠️ Correctness + Performance (C + P)**
>   - ❌ **Failing**: If either check fails.
>   - ✅ **Passing**: If **BOTH** checks pass successfully.
>   - ❓ **Untested**: If any check is untested (and neither fails).
>
> - **🌐 Hardware Rollups (v6e + v7x)**
>   - ❌ **Failing**: If the feature fails on **either** v6e or v7x.
>   - ✅ **Passing**: If the feature passes on **BOTH** v6e and v7x.
>   - ❓ **Untested**: If either generation is untested (and neither fails).
> </details>

</details>

<br>

### Release Support Matrices

<details open markdown="1">
<summary><b>Click to expand support matrices</b></summary>

<blockquote>

<i>Stable support status for official releases and production deployments.</i><br><br>

<details open markdown="1">
<summary><b> ✅ Tested Models </b></summary>

<!-- START: release_model_support -->
| Model | Type | Unit&nbsp;Test | Correctness&nbsp;Test | Performance&nbsp;Test |
| --- | --- | --- | --- | --- |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Multimodal | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | Multimodal | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="❌ Failing">❌</span> |
| [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | Multimodal | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="❌ Failing">❌</span> |
| [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="❓ Untested">❓</span> |
| [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) | Text | <span title="✅ Passing">✅</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) | Text | <span title="✅ Passing">✅</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | Multimodal | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Multimodal | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Multimodal | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Multimodal | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [deepseek-ai/DeepSeek-Math-V2](https://huggingface.co/deepseek-ai/DeepSeek-Math-V2) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [deepseek-ai/DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [deepseek-ai/DeepSeek-V3.2-Speciale](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |

<!-- END: release_model_support -->

</details>

<details open markdown="1">
<summary><b> 🚀&nbsp; Advanced Capabilities </b></summary>
<blockquote>

<details open markdown="1">
<summary>Core Features</summary>

<!-- START: release_core_features -->
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
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
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
      <td>LoRA_Torch</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
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
      <td>Single Program Multi Data</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>Speculative Decoding: Ngram</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>KV Cache Offload</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>Multimodal Inputs</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>Speculative Decoding: Eagle3</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>hybrid kv cache</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>multi-host</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
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
      <td>Single-Host-P-D-disaggregation</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>structured_decoding</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>
<!-- END: release_core_features -->

</details>

<details open markdown="1">
<summary>Parallelism Techniques</summary>

<!-- START: release_parallelism -->
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
      <td>PP</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>DP</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
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
      <td>TP</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>CP</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>SP&nbsp;(<a href="https://github.com/vllm-project/tpu-inference/issues/1749">vote&nbsp;to&nbsp;prioritize</a>)</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>
<!-- END: release_parallelism -->

</details>

<details open markdown="1">
<summary>Quantization Methods</summary>

<!-- START: release_quantization -->
<table>
  <thead>
    <tr>
      <th>Checkpoint dtype</th>
      <th>Method</th>
      <th>Supported<br>Hardware Acceleration</th>
      <th>Flax</th>
      <th>Torchax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FP4 W4A16</td>
      <td>mxfp4</td>
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
      <td>FP8 W8A8</td>
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
> - *This table only tests checkpoint loading compatibility.*
<!-- END: release_quantization -->

</details>

</details>

<details open markdown="1">
<summary><b> 🔬 Microbenchmark Kernel Support </b></summary>
<blockquote>

<!-- START: release_microbenchmarks -->
<table>
  <thead>
    <tr>
      <th width="150" style="text-align:left">Category</th>
      <th width="300" style="text-align:left">Test</th>
      <th>W16A16</th>
      <th>W8A8</th>
      <th>W8A16</th>
      <th>W4A4</th>
      <th>W4A8</th>
      <th>W4A16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>Moe</b></td>
      <td>Fused&nbsp;MoE</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>gmm</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"><b>Dense</b></td>
      <td>All&#8209;gather&nbsp;matmul</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3"><b>Attention</b></td>
      <td>Generic&nbsp;Ragged&nbsp;Paged<br>Attention&nbsp;V3*</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>MLA</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>Ragged&nbsp;Paged<br>Attention&nbsp;V3&nbsp;Head_Dim<br>64*</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>

> **Note:**
> - *For attention kernels, W[x]A[y] denotes KV cache as W, A as compute, and x, y as bit precision.*
<!-- END: release_microbenchmarks -->

</details>

</blockquote>
</details>

<br>

### Nightly Support Matrices

<details markdown="1">
<summary><b>Click to expand support matrices</b></summary>

<blockquote>

<i>Support status for the latest nightly/main branch developments.</i><br><br>

<details open markdown="1">
<summary><b> ✅ Tested Models </b></summary>

<!-- START: nightly_model_support -->
| Model | Type | Unit&nbsp;Test | Correctness&nbsp;Test | Performance&nbsp;Test |
| --- | --- | --- | --- | --- |
| [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | Multimodal | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Multimodal | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="❌ Failing">❌</span> |
| [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | Embedding | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="❓ Untested">❓</span> |
| [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="❓ Untested">❓</span> |
| [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | Text | <span title="✅ Passing">✅</span> | <span title="✅ Passing">✅</span> | <span title="❓ Untested">❓</span> |
| [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | Multimodal | <span title="✅ Passing">✅</span> | <span title="❌ Failing">❌</span> | <span title="❓ Untested">❓</span> |
| [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | Multimodal | <span title="✅ Passing">✅</span> | <span title="❌ Failing">❌</span> | <span title="❓ Untested">❓</span> |
| [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) | Text | <span title="✅ Passing">✅</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | Multimodal | <span title="❌ Failing">❌</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | Multimodal | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Multimodal | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Multimodal | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Multimodal | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [deepseek-ai/DeepSeek-Math-V2](https://huggingface.co/deepseek-ai/DeepSeek-Math-V2) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [deepseek-ai/DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [deepseek-ai/DeepSeek-V3.2-Speciale](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |
| [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5) | Text | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> | <span title="❓ Untested">❓</span> |

<!-- END: nightly_model_support -->

</details>

<details open markdown="1">
<summary><b> 🚀&nbsp; Advanced Capabilities </b></summary>
<blockquote>

<details open markdown="1">
<summary>Core Features</summary>

<!-- START: nightly_core_features -->
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
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
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
      <td>KV Cache Offload</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>LoRA_Torch</td>
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
      <td>Single Program Multi Data</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>Speculative Decoding: Eagle3</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>Speculative Decoding: Ngram</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>Multimodal Inputs</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>Out-of-tree model support</td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
    </tr>
    <tr>
      <td>multi-host</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>hybrid kv cache</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
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
      <td>Step Pooling (Embedding)</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>structured_decoding</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>
<!-- END: nightly_core_features -->

</details>

<details open markdown="1">
<summary>Parallelism Techniques</summary>

<!-- START: nightly_parallelism -->
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
      <td>PP</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
    </tr>
    <tr>
      <td>DP</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>TP</td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>EP</td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="✅&nbsp;Passing">✅</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>SP&nbsp;(<a href="https://github.com/vllm-project/tpu-inference/issues/1749">vote&nbsp;to&nbsp;prioritize</a>)</td>
      <td><span title="❌&nbsp;Failing">❌</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>CP</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>
<!-- END: nightly_parallelism -->

</details>

<details open markdown="1">
<summary>Quantization Methods</summary>

<!-- START: nightly_quantization -->
<table>
  <thead>
    <tr>
      <th>Checkpoint dtype</th>
      <th>Method</th>
      <th>Supported<br>Hardware Acceleration</th>
      <th>Flax</th>
      <th>Torchax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FP4 W4A16</td>
      <td>mxfp4</td>
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
      <td>FP8 W8A8</td>
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
    <tr>
      <td>NVFP4 W4A16</td>
      <td>modelopt_fp4</td>
      <td>v7</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>

> **Note:**
> - *This table only tests checkpoint loading compatibility.*
<!-- END: nightly_quantization -->

</details>

</details>

<details open markdown="1">
<summary><b> 🔬 Microbenchmark Kernel Support </b></summary>
<blockquote>

<!-- START: nightly_microbenchmarks -->
<table>
  <thead>
    <tr>
      <th width="150" style="text-align:left">Category</th>
      <th width="300" style="text-align:left">Test</th>
      <th>W16A16</th>
      <th>W8A8</th>
      <th>W8A16</th>
      <th>W4A4</th>
      <th>W4A8</th>
      <th>W4A16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>Moe</b></td>
      <td>Fused&nbsp;MoE</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>gmm</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"><b>Dense</b></td>
      <td>All&#8209;gather&nbsp;matmul</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3"><b>Attention</b></td>
      <td>Generic&nbsp;Ragged&nbsp;Paged<br>Attention&nbsp;V3*</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>MLA</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>Ragged&nbsp;Paged<br>Attention&nbsp;V3&nbsp;Head_Dim<br>64*</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>

> **Note:**
> - *For attention kernels, W[x]A[y] denotes KV cache as W, A as compute, and x, y as bit precision.*
<!-- END: nightly_microbenchmarks -->

</details>

</blockquote>
</details>

<br>

