<p align="center">
   <!-- This image will ONLY show up in GitHub's dark mode -->
  <img src="assets/tpu_inference_dark_mode_short.png#gh-dark-mode-only" alt="vLLM TPU" style="width: 86%;">
    <!-- This image will ONLY show up in GitHub's light mode (and on other platforms) -->
  <img src="assets/tpu_inference_light_mode_short.png#gh-light-mode-only" alt="vLLM TPU" style="width: 86%;">
</p>
<p align="center">
| <a href="https://docs.vllm.ai/projects/tpu/en/latest/"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## About

vLLM TPU is now powered by `tpu-inference`, an expressive and powerful new hardware plugin unifying JAX and PyTorch under a single lowering path within the vLLM project. The new backend now provides a framework for developers to:

- Push the limits of TPU hardware performance in open source.
- Provide more flexibility to JAX and PyTorch users by running PyTorch model definitions performantly on TPU without any additional code changes, while also extending native support to JAX.
- Retain vLLM standardization: keep the same user experience, telemetry, and interface.

## What are you trying to do today?

<div class="grid cards three-columns" markdown>

- :material-rocket-launch:{ .lg .middle } __I'm New__

  ---

  Get started quickly with core concepts, hardware setup, and step-by-step tutorials.

  [:octicons-arrow-right-24: Quickstart: Serving a Model](getting_started/quickstart.md)

  [:octicons-arrow-right-24: Installation Guide](getting_started/installation.md)

  [:octicons-arrow-right-24: TPU Setup](getting_started/tpu_setup.md)

- :material-server-network:{ .lg .middle } __I Want to Deploy__

  ---

  Guides on infrastructure setup, deployment recipes, and hardware capabilities.

  [:octicons-arrow-right-24: Deploying on GCE (Ironwood)](deployment_guides/ironwood.md)

  [:octicons-arrow-right-24: Deploying on GCE (Trillium)](deployment_guides/trillium.md)

  [:octicons-arrow-right-24: Supported Models](recommended_models.md)

- :material-code-tags:{ .lg .middle } __I Want to Build__

  ---

  Contribute code, dive into inference examples, or explore the core architecture.

  [:octicons-arrow-right-24: Inference Examples](api_and_code_examples/multi_modal_inference.md)

  [:octicons-arrow-right-24: Developer Guides](developers_guide/contributing.md)

  [:octicons-arrow-right-24: Add a New Model to CI](developers_guide/buildkite.md)

</div>

## Contribute

We're always looking for ways to partner with the community to accelerate vLLM TPU development. If you're interested in contributing to this effort, check out the [Contributing guide](https://github.com/vllm-project/tpu-inference/blob/main/CONTRIBUTING.md) and [Issues](https://github.com/vllm-project/tpu-inference/issues) to start. We recommend filtering Issues on the [**good first issue** tag](https://github.com/vllm-project/tpu-inference/issues?q=is%3Aissue+state%3Aopen+label%3A%22good+first+issue%22) if it's your first time contributing.

## Contact us

- For technical questions and feature requests, open a GitHub [Issue](https://github.com/vllm-project/tpu-inference/issues)
- For feature requests, please open one on Github [here](https://github.com/vllm-project/tpu-inference/issues/new/choose)
- For discussing with fellow users, use the [TPU support topic in the vLLM Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27)
- For coordinating contributions and development, use the [Developer Slack](https://join.slack.com/share/enQtOTY2OTUxMDIyNjY1OS00M2MxYWQwZjAyMGZjM2MyZjRjNTA0ZjRkNjkzOTRhMzg0NDM2OTlkZDAxOTAzYmJmNzdkNDc4OGZjYTUwMmRh)
- For collaborations and partnerships, contact us at [vllm-tpu@google.com](mailto:vllm-tpu@google.com)
