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

[![good first issue](https://img.shields.io/github/issues/vllm-project/tpu-inference/good%20first%20issue.svg?label=good%20first%20issue&color=green&style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)  [![documentation bugs](https://img.shields.io/github/issues-search/vllm-project/tpu-inference?query=is%3Aopen%20is%3Aissue%20label%3Adocumentation%20label%3Abug&label=documentation%20bugs&color=orange&style=flat-square)](https://github.com/vllm-project/tpo-inference/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation+label%3Abug) [![blocked issues](https://img.shields.io/github/issues/vllm-project/tpu-inference/blocked.svg?label=🛑%20Blocked&color=brightred&style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3Ablocked)



_Latest News_ 🔥

- [Pytorch Conference](https://pytorchconference.sched.com/event/27QCh/sponsored-session-everything-everywhere-all-at-once-vllm-hardware-optionality-with-spotify-and-google-brittany-rockwell-google-shireen-kheradpey-spotify) Learn how Spotify uses vLLM with both GPUs and TPUs to drive down costs and improve user experience.
- Check back soon for a recording of our session at [Ray Summit, November 3-5](https://www.anyscale.com/ray-summit/2025) in San Francisco!
- Check back soon for a recording of our session at [JAX DevLab on November 18th](https://rsvp.withgoogle.com/events/devlab-fall-2025) in Sunnyvale!

- [2025/10] [vLLM TPU: A New Unified Backend Supporting PyTorch and JAX on TPU](https://blog.vllm.ai/2025/10/16/vllm-tpu.html)

<details>
<summary><i>Previous News</i> 🔥</summary>

</details>

<br>

---

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

*Last Updated: 2026-01-24 10:00 AM UTC*

<!-- <b> Universal Status Legend </b> -->
<!-- **Legend:** ✅ Passing | ⚠️ Beta | ❌ Failing | 📝 Planned | ❓ Untested  | N/A -->
<details >
<summary> <b>🚦 <i>Status Legend definition</i> </b> </summary>

| Emoji | Status | Meaning |
| :--- | :--- | :--- |
| ✅ | **Passing** | Tested and works as expected. Ready for use. |
| ⚠️ | **Beta** | Works, but may be unstable or have known issues. Use with caution. |
| ❌ | **Failing** | Known to be broken or not functional. Help is wanted to fix this! |
| 📝 | **Planned** | Not yet implemented, but on the official roadmap. |
| ❓ | **Untested**| The functionality exists but has not been recently or thoroughly verified. |
| ⚪️ | **N/A** | Not applicable for this feature. |

</details>
<br>

<details open>
<summary><b> ✅ Model Support </b></summary>

| Model | Type | Load Test | Correctness Test | Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Multimodal | ✅ Passing | ❌ Failing | ⚪️ N/A |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | Multimodal | ✅ Passing | ❌ Failing | ⚪️ N/A |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Multimodal | ✅ Passing | ❌ Failing | ⚪️ N/A |
| `meta-llama/Llama-3.1-8B-Instruct` | Text | ✅ Passing | ✅ Passing | ✅ Passing |
| `meta-llama/Llama-3.3-70B-Instruct` | Text | ✅ Passing | ✅ Passing | ✅ Passing |
| `meta-llama/Llama-Guard-4-12B` | Text | ✅ Passing | ❌ Failing | ⚪️ N/A |
| `Qwen/Qwen3-Coder-480B-A35B-Instruct`| Text | ❌ Failing | ⚪️ N/A | ⚪️ N/A |
| `moonshotai/Kimi-K2-Thinking` | Text | ❓ Untested | ❓ Untested | ❓ Untested |

</details>

<details>
<summary><b> 🚀&nbsp; Advanced Capabilities </b></summary>
<ul>
  <li>
    <details>
      <summary>Core Features</summary>

| Feature | Correctness Test | Performance Test |
| :--- | :--- | :--- |
| Chunked Prefill | ✅ Passing | ✅ Passing |
| LoRA_Torch | ✅ Passing | ✅ Passing |
| Prefix Caching | ✅ Passing | ✅ Passing |
| Multimodal Inputs | ❌ Failing | ⚪️ N/A |
| DCN-based P/D disaggregation | ❓ Untested | ❌ Failing |

  </li>
  <li>
    <details>
      <summary>Parallelism Techniques</summary>

| Technique (Abbreviation) | Correctness Test | Performance Test |
| :--- | :--- | :--- |
| Data Parallelism (DP) | ✅ Passing | ✅ Passing |
| Pipeline Parallelism (PP) | ✅ Passing | ✅ Passing |
| Expert Parallelism (EP) | ❌ Failing | ⚪️ N/A |
| Tensor Parallelism (TP) | ❌ Failing | ⚪️ N/A |
| Context Parallelism (CP) | ❓ Untested | ❓ Untested |

  </li>
  <li>
    <details>
      <summary>Quantization Methods</summary>

| Method / Precision | Technique | Recommended TPU | Correctness Test | Performance Test |
| :--- | :--- | :--- | :--- | :--- |
| INT8 W8A8 | compressed-tensor | v5, v6 | ❓ Untested | ❓ Untested |
| INT4 W4A16 | awq | v5, v6 | ❓ Untested | ❓ Untested |
| FP8 W8A8 | compressed-tensor | v7 | ❓ Untested | ❓ Untested |

  </li>
</ul>
</details>

<details>
<summary><b> 🔬 Key Kernel Support (For Experts) </b></summary>

| Kernel | Correctness Test | Performance Test |
| :--- | :--- | :--- |
| Ragged Paged Attention V3 | ✅ Passing | ✅ Passing |
| Collective Communication Matmul | ✅ Passing | ❓ Untested |
| Quantized Matmul | ❓ Untested | ❓ Untested |
| MoE | ❓ Untested | ❓ Untested |

</details>
<br>

## 🤝 Contribute

[![good first issue](https://img.shields.io/github/issues/vllm-project/tpu-inference/good%20first%20issue.svg?label=good%20first%20issue&color=green&style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)  [![documentation bugs](https://img.shields.io/github/issues-search/vllm-project/tpu-inference?query=is%3Aopen%20is%3Aissue%20label%3Adocumentation%20label%3Abug&label=documentation%20bugs&color=orange&style=flat-square)](https://github.com/vllm-project/tpo-inference/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation+label%3Abug) [![blocked issues](https://img.shields.io/github/issues/vllm-project/tpu-inference/blocked.svg?label=🛑%20Blocked&color=brightred&style=flat-square)](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+is%3Aopen+label%3Ablocked)

We're thrilled you're interested in contributing to the vLLM TPU project! Your help is essential for making our tools better for everyone. There are many ways to get involved, even if you're not ready to write code.

**Ways to Contribute:**

*   **🐞 Submit Bugs & Suggest Features:** See an issue or have an idea? Open a [new issue](https://github.com/vllm-project/tpu-inference/issues/new/choose) to let us know.
*   **👀 Provide Feedback on Pull Requests:** Lend your expertise by reviewing [open pull requests](https://github.com/vllm-project/tpu-inference/pulls) and helping us improve the quality of our codebase.
*   **📚 Improve Our Documentation:** Help us make our guides clearer. Fix a typo, clarify a confusing section, or write a new recipe.

If you're ready to contribute code, our **[Contributing Guide](https://github.com/vllm-project/tpu-inference/blob/main/CONTRIBUTING.md)** is the best place to start. It covers everything you need to know, including:

*   Setting up your development environment.
*   Our development workflow, from running tests to debugging.
*   Guidelines for writing clean and effective code.
*   How to submit a high-quality Pull Request.
*   Tips for finding an issue to work on (we recommend starting with our **[good-first issues](link-to-your-mission-board)**!).

<br>

## 💬&nbsp; Contact us  

- For technical questions and feature requests, open a GitHub [Issue](https://github.com/vllm-project/tpu-inference/issues)
- For feature requests, please open one on Github [here](https://github.com/vllm-project/tpu-inference/issues/new/choose)
- For discussing with fellow users, use the [TPU support topic in the vLLM Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27)
- For coordinating contributions and development, use the [Developer Slack](https://join.slack.com/share/enQtOTY2OTUxMDIyNjY1OS00M2MxYWQwZjAyMGZjM2MyZjRjNTA0ZjRkNjkzOTRhMzg0NDM2OTlkZDAxOTAzYmJmNzdkNDc4OGZjYTUwMmRh)
- For collaborations and partnerships, contact us at [vllm-tpu@google.com](mailto:vllm-tpu@google.com)



