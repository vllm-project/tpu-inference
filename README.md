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

_Latest News_ 🔥

- [Pytorch Conference](https://pytorchconference.sched.com/event/27QCh/sponsored-session-everything-everywhere-all-at-once-vllm-hardware-optionality-with-spotify-and-google-brittany-rockwell-google-shireen-kheradpey-spotify) Learn how Spotify uses vLLM with both GPUs and TPUs to drive down costs and improve user experience.
- Check back soon for a recording of our session at [Ray Summit, November 3-5](https://www.anyscale.com/ray-summit/2025) in San Francisco!
- Check back soon for a recording of our session at [JAX DevLab on November 18th](https://rsvp.withgoogle.com/events/devlab-fall-2025) in Sunnyvale!

- [2025/10] [vLLM TPU: A New Unified Backend Supporting PyTorch and JAX on TPU](https://blog.vllm.ai/2025/10/16/vllm-tpu.html)

<details>
<summary><i>Previous News</i> 🔥</summary>

</details>

---
## About

vLLM TPU is now powered by `tpu-inference`, an expressive and powerful new hardware plugin unifying JAX and PyTorch under a single lowering path within the vLLM project. The new backend now provides a framework for developers to:

- Push the limits of TPU hardware performance in open source.
- Provide more flexibility to JAX and PyTorch users by running PyTorch model definitions performantly on TPU without any additional code changes, while also extending native support to JAX.
- Retain vLLM standardization: keep the same user experience, telemetry, and interface.

## Recommended models and features

Although vLLM TPU’s new unified backend makes out-of-the-box high performance serving possible with any model supported in vLLM, the reality is that we're still in the process of implementing a few core components.

For this reason, we’ve provided a **[Recommended Models and Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/)** page detailing the models and features that are validated through unit, integration, and performance testing.

## Get started

Get started with vLLM on TPUs by following the [quickstart guide](https://docs.vllm.ai/projects/tpu/en/latest/getting_started/quickstart/).

Visit our [documentation](https://docs.vllm.ai/projects/tpu/en/latest/) to learn more.

**Compatible TPU Generations**
- Recommended: v5e, v6e
- Experimental: v3, v4, v5p

*Check out a few v6e recipes [here](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM)!*

## Contribute

We're always looking for ways to partner with the community to accelerate vLLM TPU development. If you're interested in contributing to this effort, check out the [Contributing guide](https://github.com/vllm-project/tpu-inference/blob/main/CONTRIBUTING.md) and [Issues](https://github.com/vllm-project/tpu-inference/issues) to start. We recommend filtering Issues on the [**good first issue** tag](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+state%3Aopen+label%3A%22good+first+issue%22) if it's your first time contributing.

## Contact us

- For technical questions and feature requests, open a GitHub [Issue](https://github.com/vllm-project/tpu-inference/issues)
- For feature requests, please open one on Github [here](https://github.com/vllm-project/tpu-inference/issues/new/choose)
- For discussing with fellow users, use the [TPU support topic in the vLLM Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27)
- For coordinating contributions and development, use the [Developer Slack](https://join.slack.com/share/enQtOTY2OTUxMDIyNjY1OS00M2MxYWQwZjAyMGZjM2MyZjRjNTA0ZjRkNjkzOTRhMzg0NDM2OTlkZDAxOTAzYmJmNzdkNDc4OGZjYTUwMmRh)
- For collaborations and partnerships, contact us at [vllm-tpu@google.com](mailto:vllm-tpu@google.com)
