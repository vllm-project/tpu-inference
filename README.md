<p align="center">
  <img src="docs/assets/tpu2.0_banner.png" alt="vLLM TPU 2.0">
</p>

<h3 align="center">
A high performance backend unifying PyTorch and JAX in vLLM on TPU
</h3>

<p align="center">
| <a href="https://github.com/vllm-project/tpu-inference/tree/main/docs"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> (#sig-tpu) |
</p>

---

_Upcoming Events_ 🔥

- Join us at the [PyTorch Conference, October 22-23](https://events.linuxfoundation.org/pytorch-conference/) in San Francisco!
- Join us at [Ray Summit, November 3-5](https://www.anyscale.com/ray-summit/2025) in San Francisco!
- Join us at [JAX DevDay on November 18th](https://rsvp.withgoogle.com/events/devlab-fall-2025) in Sunnyvale!
- vLLM Social Hour sponsored by Google at PyTorch Conference (link to RSVP will be added closer to the event date)
- Google is hosting the vLLM Meetup in Seoul, South Korea (link to RSVP will be added closer to the event date)
- Google is hosting the vLLM Meetup in Singapore (link to RSVP will be added closer to the event date)

_Latest News_ 🔥

- [2025/10] vLLM TPU 2.0: A New Unified-Backend Supporting PyTorch and JAX on TPU
<!--TODO: add link: Read Google Cloud's Blog Post about vLLM TPU 2.0!-->

<details>
<summary><i>Previous News</i> 🔥</summary>

- [2025/04] vLLM TPU 1.0 announced at Cloud Next 2025
</details>

---

## About

vLLM TPU is now powered by `tpu-inference`, an expressive and powerful new hardware plugin unifying JAX and PyTorch under a single lowering path within the vLLM project. It's faster than vLLM TPU 1.0 and offers broader model coverage and feature support. vLLM TPU now provides a framework for developers to:

- Push the limits of TPU hardware **performance** in open source.
- Provide more **flexibility** to JAX and PyTorch users by running PyTorch model definitions performantly on TPU without any additional code changes, while also extending native support to JAX.
- Retain vLLM **standardization** by keeping the same user experience, telemetry, and interface.

## Get started

Get started with vLLM on TPUs by following the [quickstart guide](https://github.com/vllm-project/tpu-inference/tree/main/docs/getting_started/quickstart.md).

Visit our [documentation](https://github.com/vllm-project/tpu-inference/tree/main/docs) to learn more:

- [Quickstart](https://github.com/vllm-project/tpu-inference/tree/main/docs/getting_started/quickstart.md)
<!--TODO: add link to list of supported models-->

## Contribute

We're always looking for ways to partner with the community to accelerate TPU development. If you're interested in contributing to this effort, here are some open feature requests we’d love your help on:

1. Pooling/Embedding models <!--TODO: add link to existing FR-->

## Contact us

- For technical questions and feature requests, use GitHub [Issues](https://github.com/vllm-project/tpu-inference/issues)
- For discussing with fellow users, use the [TPU support topic in the vLLM Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27)
- For coordinating contributions and development, use [Slack](https://slack.vllm.ai) (#sig-tpu)
- For collaborations and partnerships, contact us at [vllm-tpu@google.com](mailto:vllm-tpu@google.com)
