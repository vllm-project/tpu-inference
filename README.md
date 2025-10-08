<p align="center">
  <img src="docs/assets/tpu2.0_banner.png" alt="vLLM TPU 2.0">
</p>

<h3 align="center">
A new, high performance TPU backend unifying PyTorch and JAX in vLLM
</h3>

<p align="center">
| <a href="https://github.com/vllm-project/tpu-inference/tree/main/docs"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> (#sig-tpu) |
</p>

---

_Upcoming Events_ 🔥

- Join us at the [PyTorch Conference, October 22-23](https://events.linuxfoundation.org/pytorch-conference/) in San Francisco!
- Join us at [Ray Summit, November 3-5](https://www.anyscale.com/ray-summit/2025) in San Francisco!
- Join us at [JAX DevLab on November 18th](https://rsvp.withgoogle.com/events/devlab-fall-2025) in Sunnyvale!
  
_Latest News_ 🔥

- [2025/10] vLLM TPU 2.0: A New Unified-Backend Supporting PyTorch and JAX on TPU
<!--TODO: add link: Read Google Cloud's Blog Post about vLLM TPU 2.0!-->

<details>
<summary><i>Previous News</i> 🔥</summary>
  
</details>

---

## About

vLLM TPU is now powered by `tpu-inference`, an expressive and powerful new hardware plugin unifying JAX and PyTorch under a single lowering path within the vLLM project. The new backend now provides a framework for developers to:

- Push the limits of TPU hardware **performance** in open source.
- Provide more **flexibility** to JAX and PyTorch users by running PyTorch model definitions performantly on TPU without any additional code changes, while also extending native support to JAX.
- Retain vLLM **standardization** by keeping the same user experience, telemetry, and interface.

## Supported models and features

See the following links for a list of stress-tested and validated models and features:

- [Model support matrix](https://github.com/vllm-project/tpu-inference/blob/main/model_support_matrix.csv)
- [Feature support matrix](https://github.com/vllm-project/tpu-inference/blob/main/feature_support_matrix.csv)

## Get started

Get started with vLLM on TPUs by following the [quickstart guide](https://github.com/vllm-project/tpu-inference/tree/main/docs/getting_started/quickstart.md).

Visit our [documentation](https://github.com/vllm-project/tpu-inference/tree/main/docs) to learn more:

## Contribute

We're always looking for ways to partner with the community to accelerate vLLM TPU development. If you're interested in contributing to this effort, check the current [Issues](https://github.com/vllm-project/tpu-inference/issues) with the **First Timer** tag for recommend issues to get started with.

## Contact us

- For technical questions and feature requests, open a GitHub [Issue](https://github.com/vllm-project/tpu-inference/issues)
- For discussing with fellow users, use the [TPU support topic in the vLLM Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27)
- For coordinating contributions and development, use [Slack](https://slack.vllm.ai) (#sig-tpu)
- For collaborations and partnerships, contact us at [vllm-tpu@google.com](mailto:vllm-tpu@google.com)
