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

## Recommended models and features

Although vLLM TPU’s new unified backend makes out-of-the-box high performance serving possible with any model supported in vLLM, the reality is that we're still in the process of implementing a few core components.

For this reason, we’ve provided a **[Recommended Models and Features](recommended_models_features.md)** page detailing the models and features that are validated through unit, integration, and performance testing.

## Getting Started

If you are new to vLLM on TPU, we recommend starting with the **[Quickstart](getting_started/quickstart.md)** guide. It will walk you through the process of setting up your environment and running your first model. For more detailed installation instructions, you can refer to the **[Installation](getting_started/installation.md)** guide.

**Compatible TPU Generations**

- Recommended: v7x, v5e, v6e
- Experimental: v3, v4, v5p

**Recipes**

Tested end-to-end guides for hosting specific models on specific TPU generations.

- [v7x (Ironwood) Recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM)
- [v6e (Trillium) Recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM)

## Developer Guides

If you are interested in contributing to the project or want to learn more about the internals, check out our developer guides:

- **[JAX Model Development](developer_guides/jax_model_development.md)**
- **[Torch Model Development](developer_guides/torchax_model_development.md)**

## Contribute

### 🤝 How to Contribute a Model in 3 Steps

Do you see a model in the **[Tested Models Matrix](recommended_models_features.md)** that is marked as **❓ Untested** or **❌ Failing**? These are our active targets! Follow this quick roadmap to claim and add support:

---

#### 1️⃣ Step 1: Pick and Claim Your Target 🎯
* **Find a target:** Look at the Recommended Models page above for any model marked **❓ Untested** or **❌ Failing**.
* **Claim it:** To avoid overlapping work, search our issues or open a new one titled `[Model Support] Add <Model Name>` to let the community know you are actively working on it.

---

#### 2️⃣ Step 2: Select Your Path & Set Up 💻
Choose your developer frontend path depending on your background and model:
* 🔹 **Torchax Path:** *Best for bringing existing upstream PyTorch vLLM models to TPU instantly.* Read the [Torchax Model Development Guide](developer_guides/torchax_model_development.md).
* 🔹 **JAX Native Path:** *Best for custom TPU layers, expert optimization, or Mixture of Experts (MoE).* Read the [JAX Model Development Guide](developer_guides/jax_model_development.md).

**Set up your workspace in 60 seconds:**
```bash
git clone https://github.com/vllm-project/tpu-inference.git && cd tpu-inference
pip install -e .
pip install pre-commit && pre-commit install
```

---

#### 3️⃣ Step 3: Run Local Tests & Submit PR 🚀
Verify your model's weight loader logic and layers locally before opening a PR:
```bash
# Verify your model locally
pytest tests/models/jax/test_your_model_name.py
```
* **Submit PR:** Open a pull request on GitHub. **Ensure you include/update the model's unit test in `tests/models/`!**
* **Watch it Land:** Our automated CI fleet will compile and benchmark your code on real Google Cloud TPU VM nodes. Once merged, **your model status in the live Support Matrix will automatically update to ✅ Passing within 24 hours!**


## Contact us

- For technical questions and feature requests, open a GitHub [Issue](https://github.com/vllm-project/tpu-inference/issues)
- For feature requests, please open one on Github [here](https://github.com/vllm-project/tpu-inference/issues/new/choose)
- For discussing with fellow users, use the [TPU support topic in the vLLM Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27)
- For coordinating contributions and development, use the [Developer Slack](https://join.slack.com/share/enQtOTY2OTUxMDIyNjY1OS00M2MxYWQwZjAyMGZjM2MyZjRjNTA0ZjRkNjkzOTRhMzg0NDM2OTlkZDAxOTAzYmJmNzdkNDc4OGZjYTUwMmRh)
- For collaborations and partnerships, contact us at [vllm-tpu@google.com](mailto:vllm-tpu@google.com)
