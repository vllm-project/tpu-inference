<p align="center">
  <img src="docs/assets/tpu_header_new_preview_v1.png" alt="vLLM TPU 2.0" style="width: 80%; margin: 85px 0;">
</p>

<h3 align="center">
A New High Performance TPU Backend Unifying PyTorch and JAX in vLLM
</h3>

<p align="center">
| <a href="https://github.com/vllm-project/tpu-inference/tree/main/docs"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27"><b>User Forum</b></a> | <a href="https://join.slack.com/share/enQtOTY2OTUxMDIyNjY1OS00M2MxYWQwZjAyMGZjM2MyZjRjNTA0ZjRkNjkzOTRhMzg0NDM2OTlkZDAxOTAzYmJmNzdkNDc4OGZjYTUwMmRh"><b>Developer Slack</b></a> |
</p>

---

_Upcoming Events_ 🔥

- Join us at the [PyTorch Conference, October 22-23](https://events.linuxfoundation.org/pytorch-conference/) in San Francisco!
- Join us at [Ray Summit, November 3-5](https://www.anyscale.com/ray-summit/2025) in San Francisco!
- Join us at [JAX DevLab on November 18th](https://rsvp.withgoogle.com/events/devlab-fall-2025) in Sunnyvale!
  
_Latest News_ 🔥

- [2025/10] vLLM TPU: A New Unified Backend Supporting PyTorch and JAX on TPU

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

For this reason, we’ve provided a list of recommended [models](https://github.com/vllm-project/tpu-inference/blob/main/model_support_matrix.csv) and [features](https://github.com/vllm-project/tpu-inference/blob/main/feature_support_matrix.csv) that are validated for accuracy and stress-tested for performance.

## Get started

Get started with vLLM on TPUs by following the [quickstart guide](https://github.com/vllm-project/tpu-inference/tree/main/docs/getting_started/quickstart.md).

Visit our [documentation](https://github.com/vllm-project/tpu-inference/tree/main/docs) to learn more.

## Contribute

We're always looking for ways to partner with the community to accelerate vLLM TPU development. If you're interested in contributing to this effort, check out the [Contributing guide](https://github.com/vllm-project/tpu-inference/blob/main/CONTRIBUTING.md) and [Issues](https://github.com/vllm-project/tpu-inference/issues) to start. We recommend filtering Issues on the [**good first issue** tag](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22) if it's your first time contributing.

## Contact us

- For technical questions and feature requests, open a GitHub [Issue](https://github.com/vllm-project/tpu-inference/issues)
- For feature requests, please open one on Github [here](https://github.com/vllm-project/tpu-inference/issues/new/choose)
- For discussing with fellow users, use the [TPU support topic in the vLLM Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27)
- For coordinating contributions and development, use the [Developer Slack](https://join.slack.com/share/enQtOTY2OTUxMDIyNjY1OS00M2MxYWQwZjAyMGZjM2MyZjRjNTA0ZjRkNjkzOTRhMzg0NDM2OTlkZDAxOTAzYmJmNzdkNDc4OGZjYTUwMmRh)
- For collaborations and partnerships, contact us at [vllm-tpu@google.com](mailto:vllm-tpu@google.com)
