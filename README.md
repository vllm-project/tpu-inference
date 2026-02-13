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

*Last Updated: 2026-02-13 11:00 AM UTC*

<details>
<summary> <b>🚦 <i>Status Legend</i> </b> </summary>

> | Emoji | Status | Meaning |
> | :--- | :--- | :--- |
> | ✅ | **Passing** | Tested and works as expected. Ready for use. |
> | ⚠️ | **Beta** | Works, but may be unstable or have known issues. Use with caution. |
> | ❌ | **Failing** | Known to be broken or not functional. Help is wanted to fix this! |
> | 📝 | **Planned** | Not yet implemented, but on the official roadmap. |
> | ❓ | **Untested**| The functionality exists but has not been recently or thoroughly verified. |
> | ⚪️ | **N/A** | Not applicable for this feature. |

</details>

<br>

<details open>
<summary><b> ✅ Model Support </b></summary>

<!-- START: model_support -->
| Model | UnitTest | Accuracy/Correctness | Benchmark |
| --- | --- | --- | --- |
| moonshotai/Kimi-K2-Thinking | unverified | unverified | unverified |
| Qwen/Qwen3-Coder-480B-A35B-Instruct | unverified | unverified | unverified |
| meta-llama/Llama-3.3-70B-Instruct | ✅ | ✅ | ✅ |
| Qwen/Qwen3-4B | ✅ | ✅ | ✅ |
| google/gemma-3-27b-it | ✅ | ✅ | ✅ |
| Qwen/Qwen3-32B | ✅ | ✅ | ✅ |
| deepseek-ai/DeepSeek-V3.1 | unverified | unverified | unverified |
| meta-llama/Llama-Guard-4-12B | ✅ | ✅ | ✅ |
| openai/gpt-oss-120b | unverified | unverified | unverified |
| meta-llama/Llama-3.1-8B-Instruct | ✅ | ✅ | ✅ |
| Qwen/Qwen3-30B-A3B | ✅ | ✅ | ✅ |
| meta-llama/Llama-4-Maverick-17B-128E-Instruct | unverified | unverified | unverified |
| Qwen/Qwen2.5-VL-7B-Instruct | ✅ | ✅ | ✅ |
| Qwen/Qwen3-Omni-30B-A3B-Instruct | unverified | unverified | unverified |

<!-- END: model_support -->

</details>

<details>
<summary><b> 🚀&nbsp; Advanced Capabilities </b></summary>
<ul>
  <li>
    <details>
      <summary>Core Features</summary>

<!-- START: core_features -->
| Feature | CorrectnessTest | PerformanceTest |
| --- | --- | --- |
| Chunked Prefill | ✅ | ✅ |
| DCN-based P/D disaggregation | unverified | ✅ |
| KV cache host offloading | unverified | unverified |
| LoRA_Torch | ✅ | ✅ |
| Multimodal Inputs | ✅ | ✅ |
| Out-of-tree model support | ✅ | ✅ |
| Prefix Caching | ✅ | ✅ |
| Single Program Multi Data | ✅ | ✅ |
| Single-Host-P-D-disaggregation | N/A | N/A |
| Speculative Decoding: Eagle3 | ✅ | ✅ |
| Speculative Decoding: Ngram | ✅ | ✅ |
| async scheduler | ✅ | ✅ |
| data_parallelism | ✅ | unverified |
| runai_model_streamer_loader | ✅ | N/A |
| sampling_params | ✅ | N/A |
| structured_decoding | ✅ | N/A |

<!-- END: core_features -->

    </details>
  </li>
  <li>
    <details>
      <summary>Parallelism Techniques</summary>

<!-- START: parallelism -->
| Feature | CorrectnessTest | PerformanceTest |
| --- | --- | --- |
| CP | unverified | unverified |
| DP | ✅ | unverified |
| EP | ✅ | unverified |
| PP | ✅ | ✅ |
| SP | unverified | unverified |
| TP | ✅ | unverified |

<!-- END: parallelism -->

    </details>
  </li>
  <li>
    <details>
      <summary>Quantization Methods</summary>

<!-- START: quantization -->
| Feature | Recommended TPU Generations | CorrectnessTest | PerformanceTest |
| --- | --- | --- | --- |
| AWQ INT4 | v5, v6 | unverified | unverified |
| FP4 W4A16 | v7 | unverified | unverified |
| FP8 W8A8 | v7 | unverified | unverified |
| FP8 W8A16 | v7 | unverified | unverified |
| INT4 W4A16 | v5, v6 | unverified | unverified |
| INT8 W8A8 | v5, v6 | unverified | unverified |

<!-- END: quantization -->

    </details>
  </li>
</ul>
</details>

<details>
<summary><b> 🔬 Key Kernel Support (For Experts) </b></summary>

<!-- START: kernel_support -->
| Feature | CorrectnessTest | PerformanceTest |
| --- | --- | --- |
| Collective Communication Matmul | ✅ | unverified |
| MLA | unverified | unverified |
| MoE | unverified | unverified |
| Quantized Attention | unverified | unverified |
| Quantized KV Cache | unverified | unverified |
| Quantized Matmul | unverified | unverified |
| Ragged Paged Attention V3 | ✅ | ✅ |

<!-- END: kernel_support -->

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

*   **Tips for finding an issue to work on** (we recommend starting with our **[good-first issues](link-to-your-mission-board)**!).

<br>

## 🌟 Contributors Wall

A huge thank you to everyone who has helped build and improve `vllm-project/tpu-inference`!

<!-- START: contributors -->
<div align="center" style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;">
<a href="https://github.com/xiangxu-google" title="xiangxu-google"><img src="https://avatars.githubusercontent.com/u/117880274?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/buildkite-bot" title="buildkite-bot"><img src="https://avatars.githubusercontent.com/u/103607375?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/kyuyeunk" title="kyuyeunk"><img src="https://avatars.githubusercontent.com/u/62023335?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/py4" title="py4"><img src="https://avatars.githubusercontent.com/u/747819?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/jrplatin" title="jrplatin"><img src="https://avatars.githubusercontent.com/u/31421084?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/fenghuizhang" title="fenghuizhang"><img src="https://avatars.githubusercontent.com/u/159459388?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/sixiang-google" title="sixiang-google"><img src="https://avatars.githubusercontent.com/u/169193309?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/lsy323" title="lsy323"><img src="https://avatars.githubusercontent.com/u/6871543?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/wenxindongwork" title="wenxindongwork"><img src="https://avatars.githubusercontent.com/u/161090399?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/Lumosis" title="Lumosis"><img src="https://avatars.githubusercontent.com/u/30372757?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/vanbasten23" title="vanbasten23"><img src="https://avatars.githubusercontent.com/u/5279639?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/QiliangCui" title="QiliangCui"><img src="https://avatars.githubusercontent.com/u/9204706?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/Chenyaaang" title="Chenyaaang"><img src="https://avatars.githubusercontent.com/u/42742451?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/bzgoogle" title="bzgoogle"><img src="https://avatars.githubusercontent.com/u/198827084?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/yarongmu-google" title="yarongmu-google"><img src="https://avatars.githubusercontent.com/u/150371854?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/yaochengji" title="yaochengji"><img src="https://avatars.githubusercontent.com/u/8017489?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/mrjunwan-lang" title="mrjunwan-lang"><img src="https://avatars.githubusercontent.com/u/227443695?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/wwl2755-google" title="wwl2755-google"><img src="https://avatars.githubusercontent.com/u/214731710?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/lk-chen" title="lk-chen"><img src="https://avatars.githubusercontent.com/u/5988771?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/boe20211" title="boe20211"><img src="https://avatars.githubusercontent.com/u/120631815?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/kwang3939" title="kwang3939"><img src="https://avatars.githubusercontent.com/u/29532482?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/bythew3i" title="bythew3i"><img src="https://avatars.githubusercontent.com/u/21976464?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/gpolovets1" title="gpolovets1"><img src="https://avatars.githubusercontent.com/u/21033602?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/jcyang43" title="jcyang43"><img src="https://avatars.githubusercontent.com/u/24908445?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/pv97" title="pv97"><img src="https://avatars.githubusercontent.com/u/18700335?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/karan" title="karan"><img src="https://avatars.githubusercontent.com/u/3261985?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/dennisYehCienet" title="dennisYehCienet"><img src="https://avatars.githubusercontent.com/u/182058254?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/ica-chao" title="ica-chao"><img src="https://avatars.githubusercontent.com/u/217655063?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/patemotter" title="patemotter"><img src="https://avatars.githubusercontent.com/u/587312?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/richardsliu" title="richardsliu"><img src="https://avatars.githubusercontent.com/u/39319471?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/xingliu14" title="xingliu14"><img src="https://avatars.githubusercontent.com/u/93360308?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/andrewkvuong" title="andrewkvuong"><img src="https://avatars.githubusercontent.com/u/32935673?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/wdhongtw" title="wdhongtw"><img src="https://avatars.githubusercontent.com/u/16065489?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/sierraisland" title="sierraisland"><img src="https://avatars.githubusercontent.com/u/133469784?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/syhuang22" title="syhuang22"><img src="https://avatars.githubusercontent.com/u/92184759?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/RobMulla" title="RobMulla"><img src="https://avatars.githubusercontent.com/u/6800879?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/wang2yn84" title="wang2yn84"><img src="https://avatars.githubusercontent.com/u/13134832?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/JiriesKaileh" title="JiriesKaileh"><img src="https://avatars.githubusercontent.com/u/70413306?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/weiyu0824" title="weiyu0824"><img src="https://avatars.githubusercontent.com/u/62784299?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/ylangtsou" title="ylangtsou"><img src="https://avatars.githubusercontent.com/u/149562838?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/catswe" title="catswe"><img src="https://avatars.githubusercontent.com/u/212922539?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/helloworld1" title="helloworld1"><img src="https://avatars.githubusercontent.com/u/247316?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/rupengliu-meta" title="rupengliu-meta"><img src="https://avatars.githubusercontent.com/u/230299083?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/bvrockwell" title="bvrockwell"><img src="https://avatars.githubusercontent.com/u/24945384?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/BirdsOfAFthr" title="BirdsOfAFthr"><img src="https://avatars.githubusercontent.com/u/29437681?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/amacaskill" title="amacaskill"><img src="https://avatars.githubusercontent.com/u/44151034?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/qihqi" title="qihqi"><img src="https://avatars.githubusercontent.com/u/1719482?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/cychiuak" title="cychiuak"><img src="https://avatars.githubusercontent.com/u/68217955?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/ernie-chang" title="ernie-chang"><img src="https://avatars.githubusercontent.com/u/198010465?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/abhinavclemson" title="abhinavclemson"><img src="https://avatars.githubusercontent.com/u/54861033?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/mailvijayasingh" title="mailvijayasingh"><img src="https://avatars.githubusercontent.com/u/14227112?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/utkarshsharma1" title="utkarshsharma1"><img src="https://avatars.githubusercontent.com/u/28705599?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/CienetStingLin" title="CienetStingLin"><img src="https://avatars.githubusercontent.com/u/126043951?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/helloleah" title="helloleah"><img src="https://avatars.githubusercontent.com/u/6391870?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/coolkp" title="coolkp"><img src="https://avatars.githubusercontent.com/u/22536797?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/hosseinsarshar" title="hosseinsarshar"><img src="https://avatars.githubusercontent.com/u/4457205?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/aman2930" title="aman2930"><img src="https://avatars.githubusercontent.com/u/4409685?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/yuyanpeng-google" title="yuyanpeng-google"><img src="https://avatars.githubusercontent.com/u/193563974?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/vlad-karp" title="vlad-karp"><img src="https://avatars.githubusercontent.com/u/210436218?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/rupeng-liu" title="rupeng-liu"><img src="https://avatars.githubusercontent.com/u/242684140?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/zixi-qi" title="zixi-qi"><img src="https://avatars.githubusercontent.com/u/22851944?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/lepan-google" title="lepan-google"><img src="https://avatars.githubusercontent.com/u/129339828?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/inho9606" title="inho9606"><img src="https://avatars.githubusercontent.com/u/29620436?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/dannikay" title="dannikay"><img src="https://avatars.githubusercontent.com/u/48867745?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/zzzwen" title="zzzwen"><img src="https://avatars.githubusercontent.com/u/1835075?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/kuafou" title="kuafou"><img src="https://avatars.githubusercontent.com/u/41641871?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/kyle-google" title="kyle-google"><img src="https://avatars.githubusercontent.com/u/111800332?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/amishacorns" title="amishacorns"><img src="https://avatars.githubusercontent.com/u/13968559?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/AahilA" title="AahilA"><img src="https://avatars.githubusercontent.com/u/44123487?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
<a href="https://github.com/A9isha" title="A9isha"><img src="https://avatars.githubusercontent.com/u/55637700?v=4" width="60" style="border-radius: 50%; margin: 5px;"/></a>
</div>
<!-- END: contributors -->

<br>

## 💬&nbsp; Contact us  

- For technical questions and feature requests, open a GitHub [Issue](https://github.com/vllm-project/tpu-inference/issues)
- For feature requests, please open one on Github [here](https://github.com/vllm-project/tpu-inference/issues/new/choose)
- For discussing with fellow users, use the [TPU support topic in the vLLM Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27)
- For coordinating contributions and development, use the [Developer Slack](https://join.slack.com/share/enQtOTY2OTUxMDIyNjY1OS00M2MxYWQwZjAyMGZjM2MyZjRjNTA0ZjRkNjkzOTRhMzg0NDM2OTlkZDAxOTAzYmJmNzdkNDc4OGZjYTUwMmRh)
- For collaborations and partnerships, contact us at [vllm-tpu@google.com](mailto:vllm-tpu@google.com)



