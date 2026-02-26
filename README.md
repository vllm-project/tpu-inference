<p align="center">
   <!-- This image will ONLY show up in GitHub's dark mode -->
  <img src="docs/assets/tpu_inference_dark_mode_short.png#gh-dark-mode-only" alt="vLLM TPU" style="width: 86%;">
    <!-- This image will ONLY show up in GitHub's light mode (and on other platforms) -->
  <img src="docs/assets/tpu_inference_light_mode_short.png#gh-light-mode-only" alt="vLLM TPU" style="width: 86%;">
</p>

<p align="center">
| <a href="https://docs.vllm.ai/projects/tpu/en/latest/"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a>  (#sig-tpu) |
</p>

---

<p>
  <b>🤝 Contribute to the Project</b><br>
  <sub><i>Looking to help? Click a badge below to find issues that need your attention.</i></sub>
</p>

<!-- START: issue_badges -->
[![bug](https://img.shields.io/badge/bug-12-d73a4a?style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22bug%22) [![good first issue](https://img.shields.io/badge/good%20first%20issue-8-7057ff?style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22good%20first%20issue%22) [![enhancement](https://img.shields.io/badge/enhancement-7-a2eeef?style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22enhancement%22) [![contribution-welcome](https://img.shields.io/badge/contribution--welcome-5-ededed?style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22contribution-welcome%22) [![auto-generated](https://img.shields.io/badge/auto--generated-5-ededed?style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22auto-generated%22) [![View All Issues](https://img.shields.io/badge/View%20All%20Issues-184-238636?style=flat-square)](https://github.com/vllm-project/tpu-inference/issues)
<!-- END: issue_badges -->






## 📰 Latest News

- [Pytorch Conference](https://pytorchconference.sched.com/event/27QCh/sponsored-session-everything-everywhere-all-at-once-vllm-hardware-optionality-with-spotify-and-google-brittany-rockwell-google-shireen-kheradpey-spotify) Learn how Spotify uses vLLM with both GPUs and TPUs to drive down costs and improve user experience.
- Check back soon for a recording of our session at [Ray Summit, November 3-5](https://www.anyscale.com/ray-summit/2025) in San Francisco!
- Check back soon for a recording of our session at [JAX DevLab on November 18th](https://rsvp.withgoogle.com/events/devlab-fall-2025) in Sunnyvale!

- [2025/10] [vLLM TPU: A New Unified Backend Supporting PyTorch and JAX on TPU](https://blog.vllm.ai/2025/10/16/vllm-tpu.html)

<details markdown="1">
<summary><i>Previous News</i> 🔥</summary>

</details>

<br>



## 👋&nbsp; About 

vLLM TPU is now powered by `tpu-inference`, an expressive and powerful new hardware plugin unifying JAX and PyTorch under a single lowering path within the vLLM project. The new backend now provides a framework for developers to:

- Push the limits of TPU hardware performance in open source.
- Provide more flexibility to JAX and PyTorch users by running PyTorch model definitions performantly on TPU without any additional code changes, while also extending native support to JAX.
- Retain vLLM standardization: keep the same user experience, telemetry, and interface.
<br>

## ⭐&nbsp; Recommended models and features

Although vLLM TPU’s new unified backend makes out-of-the-box high performance serving possible with any model supported in vLLM, the reality is that we're still in the process of implementing a few core components.

For this reason, we’ve provided a **[Recommended Models and Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/)** page detailing the models and features that are validated through unit, integration, and performance testing.

<br>

## 🚀&nbsp; Get started

Get started with vLLM on TPUs by following the [quickstart guide](https://docs.vllm.ai/projects/tpu/en/latest/getting_started/quickstart/).

Visit our [documentation](https://docs.vllm.ai/projects/tpu/en/latest/) to learn more.

**Compatible TPU Generations**
- Recommended: v7x, v5e, v6e
- Experimental: v3, v4, v5p

<br>

## 🍳&nbsp; Recipes

- [v7x (Ironwood) Recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM)
- [v6e (Trillium) Recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM)

<br>

## 📊&nbsp; TPU Support Matrix Dashboard

Below is the live status of our supported models, features, and kernels. Click on any category to expand the detailed support table. It is automatically updated from our detailed [Support Matrices](https://github.com/vllm-project/tpu-inference/tree/main/support_matrices).

*Last Updated: 2026-02-13 11:00 AM UTC*

<details open markdown="1">
<summary> <b>🚦 <i>Status Legend</i> </b> </summary>

> | Emoji | Status | Meaning |
> | :--- | :--- | :--- |
> | ✅ | **Passing** | Tested and works as expected. Ready for use. |
> | ⚠️ | **Beta** | Works, but may be unstable or have known issues. Use with caution. |
> | 🧪 | **Experimental** | Works, but unoptimized or pending community validation. [Vote to prioritize](https://github.com/vllm-project/tpu-inference/issues/1749) |
> | ❌ | **Failing** | Known to be broken or not functional. Help is wanted to fix this! |
> | 📝 | **Planned** | Not yet implemented, but on the official roadmap. |
> | ❓ | **Untested**| The functionality exists but has not been recently or thoroughly verified. |
> | ⚪️ | **N/A** | Not applicable for this feature. |

</details>

<br>

<details open markdown="1">
<summary><b> ✅ Model Support </b></summary>

<!-- START: model_support -->
| Model | Type | Load&nbsp;Test | Correctness&nbsp;Test | Benchmark |
| --- | --- | --- | --- | --- |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | Multimodal | ❓&nbsp;Untested | ❓&nbsp;Untested | ❓&nbsp;Untested |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Multimodal | ✅&nbsp;Passing | ✅&nbsp;Passing | ✅&nbsp;Passing |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Multimodal | ❓&nbsp;Untested | ❓&nbsp;Untested | ❓&nbsp;Untested |
| `moonshotai/Kimi-K2-Thinking` | Text | ❓&nbsp;Untested | ❓&nbsp;Untested | ❓&nbsp;Untested |
| `Qwen/Qwen3-Coder-480B-A35B-Instruct` | Text | ❓&nbsp;Untested | ❓&nbsp;Untested | ❓&nbsp;Untested |
| `meta-llama/Llama-3.3-70B-Instruct` | Text | ✅&nbsp;Passing | ✅&nbsp;Passing | ✅&nbsp;Passing |
| `Qwen/Qwen3-4B` | Text | ✅&nbsp;Passing | ✅&nbsp;Passing | ✅&nbsp;Passing |
| `google/gemma-3-27b-it` | Text | ✅&nbsp;Passing | ✅&nbsp;Passing | ✅&nbsp;Passing |
| `Qwen/Qwen3-32B` | Text | ✅&nbsp;Passing | ✅&nbsp;Passing | ✅&nbsp;Passing |
| `deepseek-ai/DeepSeek-V3.1` | Text | ❓&nbsp;Untested | ❓&nbsp;Untested | ❓&nbsp;Untested |
| `meta-llama/Llama-Guard-4-12B` | Text | ✅&nbsp;Passing | ✅&nbsp;Passing | ✅&nbsp;Passing |
| `openai/gpt-oss-120b` | Text | ❓&nbsp;Untested | ❓&nbsp;Untested | ❓&nbsp;Untested |
| `meta-llama/Llama-3.1-8B-Instruct` | Text | ✅&nbsp;Passing | ✅&nbsp;Passing | ✅&nbsp;Passing |
| `Qwen/Qwen3-30B-A3B` | Text | ✅&nbsp;Passing | ✅&nbsp;Passing | ✅&nbsp;Passing |

<!-- END: model_support -->

</details>

<details markdown="1">
<summary><b> 🚀&nbsp; Advanced Capabilities </b></summary>
<ul>
  <li>
<details markdown="1">
<summary>Core Features</summary>

<!-- START: core_features -->
<table>
  <thead>
    <tr>
      <th rowspan="2">Test / Feature</th>
      <th colspan="3">v6e</th>
      <th colspan="3">v7x</th>
    </tr>
    <tr>
      <th>flax</th>
      <th>pytorch</th>
      <th>default</th>
      <th>flax</th>
      <th>pytorch</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Chunked Prefill</td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
    </tr>
    <tr>
      <td>DCN-based P/D disaggregation</td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
    </tr>
    <tr>
      <td>KV cache host offloading</td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
    </tr>
    <tr>
      <td>LoRA_Torch</td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
    </tr>
    <tr>
      <td>Multimodal Inputs</td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
    </tr>
    <tr>
      <td>Out-of-tree model support</td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
    </tr>
    <tr>
      <td>Prefix Caching</td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
    </tr>
    <tr>
      <td>Single Program Multi Data</td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
    </tr>
    <tr>
      <td>Single-Host-P-D-disaggregation</td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
    </tr>
    <tr>
      <td>Speculative Decoding: Eagle3</td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
    </tr>
    <tr>
      <td>Speculative Decoding: Ngram</td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
    </tr>
    <tr>
      <td>async scheduler</td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
      <td></td>
      <td>✅&nbsp;Passing</td>
      <td></td>
    </tr>
    <tr>
      <td>data_parallelism</td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
    </tr>
    <tr>
      <td>runai_model_streamer_loader</td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
    </tr>
    <tr>
      <td>sampling_params</td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
    </tr>
    <tr>
      <td>structured_decoding</td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
      <td></td>
      <td>❓&nbsp;Untested</td>
      <td></td>
    </tr>
  </tbody>
</table>
<!-- END: core_features -->

</details>
  </li>
  <li>
<details markdown="1">
<summary>Parallelism Techniques</summary>

<!-- START: parallelism -->
| Feature | CorrectnessTest | PerformanceTest |
| --- | --- | --- |
| CP | ❓&nbsp;Untested | ❓&nbsp;Untested |
| DP | ✅&nbsp;Passing | ❓&nbsp;Untested |
| EP | ✅&nbsp;Passing | ❓&nbsp;Untested |
| PP | ✅&nbsp;Passing | ✅&nbsp;Passing |
| SP | ❓&nbsp;Untested | ❓&nbsp;Untested |
| TP | ✅&nbsp;Passing | ❓&nbsp;Untested |

<!-- END: parallelism -->

</details>
  </li>
  <li>
<details markdown="1">
<summary>Quantization Methods</summary>

<!-- START: quantization -->
<table>
  <thead>
    <tr>
      <th rowspan="2">Format</th>
      <th rowspan="2">Method</th>
      <th rowspan="2">Recommended<br>TPU Generations</th>
      <th colspan="3">v6e</th>
      <th colspan="3">v7x</th>
    </tr>
    <tr>
      <th>flax</th>
      <th>pytorch</th>
      <th>default</th>
      <th>flax</th>
      <th>pytorch</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AWQ INT4</td>
      <td></td>
      <td>v5, v6</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
    </tr>
    <tr>
      <td>FP4 W4A16</td>
      <td>mxfp4</td>
      <td>v7</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
    </tr>
    <tr>
      <td>FP8 W8A8</td>
      <td>compressed-tensor</td>
      <td>v7</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
    </tr>
    <tr>
      <td>FP8 W8A16</td>
      <td>compressed-tensor</td>
      <td>v7</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
    </tr>
    <tr>
      <td>INT4 W4A16</td>
      <td>awq</td>
      <td>v5, v6</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
    </tr>
    <tr>
      <td>INT8 W8A8</td>
      <td>compressed-tensor</td>
      <td>v5, v6</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
      <td>❓ Untested</td>
    </tr>
  </tbody>
</table>
<!-- END: quantization -->

</details>
  </li>
  <li>
<details markdown="1">
<summary>Hardware Acceleration & Dtype Support</summary>

| TPU Generations | Natively Supported FP4/FP8 |
| :--- | :--- |
| v5 | int8, int4, bf16 |
| v6 | int8, int4, bf16 |
| v7 | fp8_e4m3fn, fp8_e5m2, bf16 |

</details>
  </li>

</ul>
</details>

<details markdown="1">
<summary><b> 🔬 Microbenchmark Kernel Support </b></summary>

<!-- START: microbenchmarks -->
<table>
  <thead>
    <tr>
      <th rowspan="2" width="300">test</th>
      <th colspan="6">v6e</th>
      <th colspan="6">V7X</th>
    </tr>
    <tr>
      <th>W16A16</th>
      <th>W8A16</th>
      <th>W8 A8</th>
      <th>W4A4</th>
      <th>W4A8</th>
      <th>W4A16</th>
      <th>W16A16</th>
      <th>W8A16</th>
      <th>W8 A8</th>
      <th>W4A4</th>
      <th>W4A8</th>
      <th>W4A16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>all&#8209;gather&#8209;matmul</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
    </tr>
    <tr>
      <td>fused&nbsp;moe</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
    </tr>
    <tr>
      <td>generic&nbsp;ragged<br>paged&nbsp;attention<br>v3*</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
    </tr>
    <tr>
      <td>gmm</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
    </tr>
    <tr>
      <td>mla*</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
    </tr>
    <tr>
      <td>ragged&nbsp;paged<br>attention&nbsp;v3<br>head_dim&nbsp;64*</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
      <td>❓&nbsp;Untested</td>
    </tr>
  </tbody>
</table>

> **Note:**
> *   *Tested on TPU v7 (Nightly 20260225)*
> *   *For attention kernels, W[x]A[y] denotes KV cache as W, A as compute, and x, y as bit precision.*
<!-- END: microbenchmarks -->

</details>

<br>

## 🤝 Contribute

<!-- START: issue_badges -->
[![bug](https://img.shields.io/badge/bug-12-d73a4a?style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22bug%22) [![good first issue](https://img.shields.io/badge/good%20first%20issue-8-7057ff?style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22good%20first%20issue%22) [![enhancement](https://img.shields.io/badge/enhancement-7-a2eeef?style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22enhancement%22) [![contribution-welcome](https://img.shields.io/badge/contribution--welcome-5-ededed?style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22contribution-welcome%22) [![auto-generated](https://img.shields.io/badge/auto--generated-5-ededed?style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22auto-generated%22) [![View All Issues](https://img.shields.io/badge/View%20All%20Issues-184-238636?style=flat-square)](https://github.com/vllm-project/tpu-inference/issues)
<!-- END: issue_badges -->

We're thrilled you're interested in contributing to the vLLM TPU project! Your help is essential for making our tools better for everyone. There are many ways to get involved, even if you're not ready to write code.

**Ways to Contribute:**

*   **🐞 Submit Bugs & Suggest Features:** See an issue or have an idea? Open a [new issue](https://github.com/vllm-project/tpu-inference/issues/new/choose) to let us know.
*   **👀 Provide Feedback on Pull Requests:** Lend your expertise by reviewing [open pull requests](https://github.com/vllm-project/tpu-inference/pulls) and helping us improve the quality of our codebase.
*   **📚 Improve Our Documentation:** Help us make our guides clearer. Fix a typo, clarify a confusing section, or write a new recipe.

If you're ready to contribute code, our **[Contributing Guide](https://github.com/vllm-project/tpu-inference/blob/main/CONTRIBUTING.md)** is the best place to start. It covers everything you need to know, including:

*   **Tips for finding an issue to work on** (we recommend starting with our **[good-first issues](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)**!).

<br>

## 🌟 Contributors Wall

A huge thank you to everyone who has helped build and improve `vllm-project/tpu-inference`!

<details markdown="1">
<summary><b>🌟 <i>Contribution Type Legend & Ranking</i></b></summary>

> | Emoji | Contribution | Meaning |
> | :---: | :--- | :--- |
> | 💻 | **Code** | Submitted merged pull requests or code changes. |
> | 🐛 | **Issues** | Opened valid issues or bug reports. |
> | 👀 | **Reviews** | Reviewed pull requests and provided feedback. |

<br>

**🏆 Ranking:** Contributors are sorted from highest to lowest based on their total effort score (`Total Commits + Unique Issues Opened + PRs Reviewed`). If there is a tie, contributors are displayed alphabetically.

</details>

<br>

<!-- START: contributors -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/xiangxu-google"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/117880274?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="xiangxu-google"/><br /><sub><b>xiangxu-google</b></sub></a><br /><a href="https://github.com/xiangxu-google" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jrplatin"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/31421084?u=0cefbcd58973670cc5def2d7a26abcf80dcaa285&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="jrplatin"/><br /><sub><b>jrplatin</b></sub></a><br /><a href="https://github.com/jrplatin" title="Contributions">🐛 👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/buildkite-bot"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/103607375?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="buildkite-bot"/><br /><sub><b>buildkite-bot</b></sub></a><br /><a href="https://github.com/buildkite-bot" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kyuyeunk"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/62023335?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="kyuyeunk"/><br /><sub><b>kyuyeunk</b></sub></a><br /><a href="https://github.com/kyuyeunk" title="Contributions">🐛 👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/py4"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/747819?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="py4"/><br /><sub><b>py4</b></sub></a><br /><a href="https://github.com/py4" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/fenghuizhang"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/159459388?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="fenghuizhang"/><br /><sub><b>fenghuizhang</b></sub></a><br /><a href="https://github.com/fenghuizhang" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lk-chen"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/5988771?u=99794c6f49c741aa6fbce0ba8e6cd015cf2ffceb&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="lk-chen"/><br /><sub><b>lk-chen</b></sub></a><br /><a href="https://github.com/lk-chen" title="Contributions">🐛 👀 💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/wenxindongwork"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/161090399?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="wenxindongwork"/><br /><sub><b>wenxindongwork</b></sub></a><br /><a href="https://github.com/wenxindongwork" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/vanbasten23"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/5279639?u=ba4c44f0572212a277f42f3937218027a8e06666&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="vanbasten23"/><br /><sub><b>vanbasten23</b></sub></a><br /><a href="https://github.com/vanbasten23" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sixiang-google"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/169193309?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="sixiang-google"/><br /><sub><b>sixiang-google</b></sub></a><br /><a href="https://github.com/sixiang-google" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lsy323"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/6871543?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="lsy323"/><br /><sub><b>lsy323</b></sub></a><br /><a href="https://github.com/lsy323" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Lumosis"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/30372757?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="Lumosis"/><br /><sub><b>Lumosis</b></sub></a><br /><a href="https://github.com/Lumosis" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/QiliangCui"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/9204706?u=1bf5731b7c40471f3277bc7f9b7d9c95e26ae722&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="QiliangCui"/><br /><sub><b>QiliangCui</b></sub></a><br /><a href="https://github.com/QiliangCui" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Chenyaaang"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/42742451?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="Chenyaaang"/><br /><sub><b>Chenyaaang</b></sub></a><br /><a href="https://github.com/Chenyaaang" title="Contributions">👀 💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bzgoogle"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/198827084?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="bzgoogle"/><br /><sub><b>bzgoogle</b></sub></a><br /><a href="https://github.com/bzgoogle" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gpolovets1"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/21033602?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="gpolovets1"/><br /><sub><b>gpolovets1</b></sub></a><br /><a href="https://github.com/gpolovets1" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mrjunwan-lang"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/227443695?u=efdbb09594f01677d3c5bcde550c129e99bab45e&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="mrjunwan-lang"/><br /><sub><b>mrjunwan-lang</b></sub></a><br /><a href="https://github.com/mrjunwan-lang" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yarongmu-google"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/150371854?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="yarongmu-google"/><br /><sub><b>yarongmu-google</b></sub></a><br /><a href="https://github.com/yarongmu-google" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/wwl2755-google"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/214731710?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="wwl2755-google"/><br /><sub><b>wwl2755-google</b></sub></a><br /><a href="https://github.com/wwl2755-google" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yaochengji"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/8017489?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="yaochengji"/><br /><sub><b>yaochengji</b></sub></a><br /><a href="https://github.com/yaochengji" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/patemotter"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/587312?u=deea9c20e09f9e254128a3109c6ec41747637cc0&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="patemotter"/><br /><sub><b>patemotter</b></sub></a><br /><a href="https://github.com/patemotter" title="Contributions">👀 💻</a></td>
    </tr>
  </tbody>
</table>
<br/>
<details markdown="1">
<summary><b>...and more! Click to view all contributors.</b></summary>
<br/>
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/boe20211"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/120631815?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="boe20211"/><br /><sub><b>boe20211</b></sub></a><br /><a href="https://github.com/boe20211" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jcyang43"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/24908445?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="jcyang43"/><br /><sub><b>jcyang43</b></sub></a><br /><a href="https://github.com/jcyang43" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kwang3939"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/29532482?u=b4fcf489ef09f16340432c08501dd85e24c1a61d&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="kwang3939"/><br /><sub><b>kwang3939</b></sub></a><br /><a href="https://github.com/kwang3939" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bythew3i"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/21976464?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="bythew3i"/><br /><sub><b>bythew3i</b></sub></a><br /><a href="https://github.com/bythew3i" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/pv97"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/18700335?u=e4a98876d81c6091aaa62ecd722e3979804bf18e&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="pv97"/><br /><sub><b>pv97</b></sub></a><br /><a href="https://github.com/pv97" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/karan"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/3261985?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="karan"/><br /><sub><b>karan</b></sub></a><br /><a href="https://github.com/karan" title="Contributions">🐛 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dennisYehCienet"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/182058254?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="dennisYehCienet"/><br /><sub><b>dennisYehCienet</b></sub></a><br /><a href="https://github.com/dennisYehCienet" title="Contributions">👀 💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/syhuang22"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/92184759?u=7526c4825f18141a20727fb29689c4d63448bc34&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="syhuang22"/><br /><sub><b>syhuang22</b></sub></a><br /><a href="https://github.com/syhuang22" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/helloworld1"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/247316?u=c107bf04adacad31e301daeb87fb95b27e282859&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="helloworld1"/><br /><sub><b>helloworld1</b></sub></a><br /><a href="https://github.com/helloworld1" title="Contributions">🐛 👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ica-chao"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/217655063?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="ica-chao"/><br /><sub><b>ica-chao</b></sub></a><br /><a href="https://github.com/ica-chao" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/richardsliu"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/39319471?u=8af5be44ea820d267202639ca549a57e2ed69bd1&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="richardsliu"/><br /><sub><b>richardsliu</b></sub></a><br /><a href="https://github.com/richardsliu" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/catswe"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/212922539?u=682a6bf9b7f8df2094f4dd625f20715f429d2723&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="catswe"/><br /><sub><b>catswe</b></sub></a><br /><a href="https://github.com/catswe" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/RobMulla"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/6800879?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="RobMulla"/><br /><sub><b>RobMulla</b></sub></a><br /><a href="https://github.com/RobMulla" title="Contributions">🐛 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/xingliu14"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/93360308?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="xingliu14"/><br /><sub><b>xingliu14</b></sub></a><br /><a href="https://github.com/xingliu14" title="Contributions">🐛 💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/juncgu-google"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/218836653?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="juncgu-google"/><br /><sub><b>juncgu-google</b></sub></a><br /><a href="https://github.com/juncgu-google" title="Contributions">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/saltysoup"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/8356553?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="saltysoup"/><br /><sub><b>saltysoup</b></sub></a><br /><a href="https://github.com/saltysoup" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/weiyu0824"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/62784299?u=2a699a9e215eb088c728742875d7c1b2424360a8&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="weiyu0824"/><br /><sub><b>weiyu0824</b></sub></a><br /><a href="https://github.com/weiyu0824" title="Contributions">👀 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/andrewkvuong"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/32935673?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="andrewkvuong"/><br /><sub><b>andrewkvuong</b></sub></a><br /><a href="https://github.com/andrewkvuong" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rupengliu-meta"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/230299083?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="rupengliu-meta"/><br /><sub><b>rupengliu-meta</b></sub></a><br /><a href="https://github.com/rupengliu-meta" title="Contributions">🐛 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bvrockwell"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/24945384?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="bvrockwell"/><br /><sub><b>bvrockwell</b></sub></a><br /><a href="https://github.com/bvrockwell" title="Contributions">🐛 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sierraisland"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/133469784?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="sierraisland"/><br /><sub><b>sierraisland</b></sub></a><br /><a href="https://github.com/sierraisland" title="Contributions">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/wang2yn84"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/13134832?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="wang2yn84"/><br /><sub><b>wang2yn84</b></sub></a><br /><a href="https://github.com/wang2yn84" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/wdhongtw"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/16065489?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="wdhongtw"/><br /><sub><b>wdhongtw</b></sub></a><br /><a href="https://github.com/wdhongtw" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/JiriesKaileh"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/70413306?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="JiriesKaileh"/><br /><sub><b>JiriesKaileh</b></sub></a><br /><a href="https://github.com/JiriesKaileh" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ylangtsou"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/149562838?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="ylangtsou"/><br /><sub><b>ylangtsou</b></sub></a><br /><a href="https://github.com/ylangtsou" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/amacaskill"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/44151034?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="amacaskill"/><br /><sub><b>amacaskill</b></sub></a><br /><a href="https://github.com/amacaskill" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/BirdsOfAFthr"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/29437681?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="BirdsOfAFthr"/><br /><sub><b>BirdsOfAFthr</b></sub></a><br /><a href="https://github.com/BirdsOfAFthr" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/patrickji2014"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/110961369?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="patrickji2014"/><br /><sub><b>patrickji2014</b></sub></a><br /><a href="https://github.com/patrickji2014" title="Contributions">👀 💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/qihqi"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/1719482?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="qihqi"/><br /><sub><b>qihqi</b></sub></a><br /><a href="https://github.com/qihqi" title="Contributions">🐛 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yuanfz98"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/42092999?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="yuanfz98"/><br /><sub><b>yuanfz98</b></sub></a><br /><a href="https://github.com/yuanfz98" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cychiuak"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/68217955?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="cychiuak"/><br /><sub><b>cychiuak</b></sub></a><br /><a href="https://github.com/cychiuak" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hosseinsarshar"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/4457205?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="hosseinsarshar"/><br /><sub><b>hosseinsarshar</b></sub></a><br /><a href="https://github.com/hosseinsarshar" title="Contributions">🐛 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/samos123"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/388784?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="samos123"/><br /><sub><b>samos123</b></sub></a><br /><a href="https://github.com/samos123" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AlienKevin"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/22850071?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="AlienKevin"/><br /><sub><b>AlienKevin</b></sub></a><br /><a href="https://github.com/AlienKevin" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dgouju"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/16699383?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="dgouju"/><br /><sub><b>dgouju</b></sub></a><br /><a href="https://github.com/dgouju" title="Contributions">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/eitanporat"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/121024776?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="eitanporat"/><br /><sub><b>eitanporat</b></sub></a><br /><a href="https://github.com/eitanporat" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ernie-chang"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/198010465?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="ernie-chang"/><br /><sub><b>ernie-chang</b></sub></a><br /><a href="https://github.com/ernie-chang" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lepan-google"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/129339828?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="lepan-google"/><br /><sub><b>lepan-google</b></sub></a><br /><a href="https://github.com/lepan-google" title="Contributions">🐛 💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/muskansh-google"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/253866901?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="muskansh-google"/><br /><sub><b>muskansh-google</b></sub></a><br /><a href="https://github.com/muskansh-google" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/saikat-royc"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/63082967?u=e603c49527018a5bf25dcdcef148a5b0a683965a&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="saikat-royc"/><br /><sub><b>saikat-royc</b></sub></a><br /><a href="https://github.com/saikat-royc" title="Contributions">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/abhinavclemson"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/54861033?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="abhinavclemson"/><br /><sub><b>abhinavclemson</b></sub></a><br /><a href="https://github.com/abhinavclemson" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aman2930"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/4409685?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="aman2930"/><br /><sub><b>aman2930</b></sub></a><br /><a href="https://github.com/aman2930" title="Contributions">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/BabyChouSr"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/49086305?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="BabyChouSr"/><br /><sub><b>BabyChouSr</b></sub></a><br /><a href="https://github.com/BabyChouSr" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/CienetStingLin"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/126043951?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="CienetStingLin"/><br /><sub><b>CienetStingLin</b></sub></a><br /><a href="https://github.com/CienetStingLin" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/coolkp"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/22536797?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="coolkp"/><br /><sub><b>coolkp</b></sub></a><br /><a href="https://github.com/coolkp" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/functionstackx"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/47992694?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="functionstackx"/><br /><sub><b>functionstackx</b></sub></a><br /><a href="https://github.com/functionstackx" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/helloleah"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/6391870?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="helloleah"/><br /><sub><b>helloleah</b></sub></a><br /><a href="https://github.com/helloleah" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mailvijayasingh"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/14227112?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="mailvijayasingh"/><br /><sub><b>mailvijayasingh</b></sub></a><br /><a href="https://github.com/mailvijayasingh" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/QiliangCui2023"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/130511281?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="QiliangCui2023"/><br /><sub><b>QiliangCui2023</b></sub></a><br /><a href="https://github.com/QiliangCui2023" title="Contributions">👀</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/shireen-bean"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/18443759?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="shireen-bean"/><br /><sub><b>shireen-bean</b></sub></a><br /><a href="https://github.com/shireen-bean" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/utkarshsharma1"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/28705599?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="utkarshsharma1"/><br /><sub><b>utkarshsharma1</b></sub></a><br /><a href="https://github.com/utkarshsharma1" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/A9isha"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/55637700?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="A9isha"/><br /><sub><b>A9isha</b></sub></a><br /><a href="https://github.com/A9isha" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AahilA"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/44123487?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="AahilA"/><br /><sub><b>AahilA</b></sub></a><br /><a href="https://github.com/AahilA" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/amishacorns"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/13968559?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="amishacorns"/><br /><sub><b>amishacorns</b></sub></a><br /><a href="https://github.com/amishacorns" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/carlesoctav"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/106587439?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="carlesoctav"/><br /><sub><b>carlesoctav</b></sub></a><br /><a href="https://github.com/carlesoctav" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dannikay"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/48867745?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="dannikay"/><br /><sub><b>dannikay</b></sub></a><br /><a href="https://github.com/dannikay" title="Contributions">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/depksingh"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/217023309?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="depksingh"/><br /><sub><b>depksingh</b></sub></a><br /><a href="https://github.com/depksingh" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Dineshkumar-Anandan-ZS0367"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/105219055?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="Dineshkumar-Anandan-ZS0367"/><br /><sub><b>Dineshkumar-Anandan-ZS0367</b></sub></a><br /><a href="https://github.com/Dineshkumar-Anandan-ZS0367" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dtrifiro"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/36171005?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="dtrifiro"/><br /><sub><b>dtrifiro</b></sub></a><br /><a href="https://github.com/dtrifiro" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/erfanzar"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/59269023?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="erfanzar"/><br /><sub><b>erfanzar</b></sub></a><br /><a href="https://github.com/erfanzar" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/inho9606"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/29620436?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="inho9606"/><br /><sub><b>inho9606</b></sub></a><br /><a href="https://github.com/inho9606" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jk1333"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/17493839?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="jk1333"/><br /><sub><b>jk1333</b></sub></a><br /><a href="https://github.com/jk1333" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jyj0w0"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/27630668?u=9bd1c8c42d174a99cc37ae8eb36e2167b624e7e8&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="jyj0w0"/><br /><sub><b>jyj0w0</b></sub></a><br /><a href="https://github.com/jyj0w0" title="Contributions">👀</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kuafou"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/41641871?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="kuafou"/><br /><sub><b>kuafou</b></sub></a><br /><a href="https://github.com/kuafou" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kyle-google"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/111800332?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="kyle-google"/><br /><sub><b>kyle-google</b></sub></a><br /><a href="https://github.com/kyle-google" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Mhdaw"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/164439157?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="Mhdaw"/><br /><sub><b>Mhdaw</b></sub></a><br /><a href="https://github.com/Mhdaw" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mokeddembillel"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/25545242?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="mokeddembillel"/><br /><sub><b>mokeddembillel</b></sub></a><br /><a href="https://github.com/mokeddembillel" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/oindrila-b"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/53270901?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="oindrila-b"/><br /><sub><b>oindrila-b</b></sub></a><br /><a href="https://github.com/oindrila-b" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/oliverdutton"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/44170519?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="oliverdutton"/><br /><sub><b>oliverdutton</b></sub></a><br /><a href="https://github.com/oliverdutton" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/pathfinder-pf"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/230268798?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="pathfinder-pf"/><br /><sub><b>pathfinder-pf</b></sub></a><br /><a href="https://github.com/pathfinder-pf" title="Contributions">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/piotrfrankowski"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/17426499?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="piotrfrankowski"/><br /><sub><b>piotrfrankowski</b></sub></a><br /><a href="https://github.com/piotrfrankowski" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/reeaz27-droid"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/245602856?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="reeaz27-droid"/><br /><sub><b>reeaz27-droid</b></sub></a><br /><a href="https://github.com/reeaz27-droid" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rupeng-liu"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/242684140?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="rupeng-liu"/><br /><sub><b>rupeng-liu</b></sub></a><br /><a href="https://github.com/rupeng-liu" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/salmanmohammadi"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/25081738?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="salmanmohammadi"/><br /><sub><b>salmanmohammadi</b></sub></a><br /><a href="https://github.com/salmanmohammadi" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/vlad-karp"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/210436218?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="vlad-karp"/><br /><sub><b>vlad-karp</b></sub></a><br /><a href="https://github.com/vlad-karp" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/XMaster96"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/28674439?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="XMaster96"/><br /><sub><b>XMaster96</b></sub></a><br /><a href="https://github.com/XMaster96" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yixinshi"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/29932260?u=a7bc68ebf1bcb7ce766e239a2c4d9c263931322b&v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="yixinshi"/><br /><sub><b>yixinshi</b></sub></a><br /><a href="https://github.com/yixinshi" title="Contributions">👀</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yuyanpeng-google"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/193563974?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="yuyanpeng-google"/><br /><sub><b>yuyanpeng-google</b></sub></a><br /><a href="https://github.com/yuyanpeng-google" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zixi-qi"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/22851944?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="zixi-qi"/><br /><sub><b>zixi-qi</b></sub></a><br /><a href="https://github.com/zixi-qi" title="Contributions">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zongweiz"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/5266615?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="zongweiz"/><br /><sub><b>zongweiz</b></sub></a><br /><a href="https://github.com/zongweiz" title="Contributions">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zzzwen"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/1835075?v=4&s=100&w=90&h=90&fit=cover&mask=circle&maxage=7d" width="45" height="45" alt="zzzwen"/><br /><sub><b>zzzwen</b></sub></a><br /><a href="https://github.com/zzzwen" title="Contributions">💻</a></td>
    </tr>
  </tbody>
</table>
</details>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- END: contributors -->

<br>

## 💬&nbsp; Contact us  

- For technical questions and feature requests, open a GitHub [Issue](https://github.com/vllm-project/tpu-inference/issues)
- For feature requests, please open one on Github [here](https://github.com/vllm-project/tpu-inference/issues/new/choose)
- For discussing with fellow users, use the [TPU support topic in the vLLM Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27)
- For coordinating contributions and development, use the [Developer Slack](https://join.slack.com/share/enQtOTY2OTUxMDIyNjY1OS00M2MxYWQwZjAyMGZjM2MyZjRjNTA0ZjRkNjkzOTRhMzg0NDM2OTlkZDAxOTAzYmJmNzdkNDc4OGZjYTUwMmRh)
- For collaborations and partnerships, contact us at [vllm-tpu@google.com](mailto:vllm-tpu@google.com)



