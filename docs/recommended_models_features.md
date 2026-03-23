# Recommended Model and Feature Matrices

Although vLLM TPU’s new unified backend makes out-of-the-box high performance serving possible with any model supported in vLLM, the reality is that we're still in the process of implementing a few core components.
For this reason, until we land more capabilities, we recommend starting from this list of stress tested models and features below.

We are still landing components in tpu-inference that will improve performance for larger scale, higher complexity models (XL MoE, +vision encoders, MLA, etc.).

If you’d like us to prioritize something specific, please submit a GitHub feature request [here](https://github.com/vllm-project/tpu-inference/issues/new/choose).

<details markdown="1">
<summary> <b>🚦 <i>Status Legend</i> </b> </summary>

> - ✅ **Passing:** Tested and works as expected. Ready for use.
> - ❌ **Failing:** Known to be broken or not functional. Help is wanted to fix this!
> - 🧪 **Experimental:** Works, but unoptimized or pending community validation.
> - 📝 **Planned:** Not yet implemented, but on the official roadmap.
> - ⛔️ **Unplanned:** There is no benefit to adding this.
> - ❓ **Untested:** The functionality exists but has not been recently or thoroughly verified.
</details>

## Recommended Models

These tables show the models currently tested for accuracy and performance.

### Models

--8<-- "docs/includes/model_support.md"

## Recommended Features

This table shows the features currently tested for accuracy and performance.

--8<-- "docs/includes/core_features.md"

## Kernel Support

This table shows the current kernel support status.

--8<-- "docs/includes/microbenchmarks.md"

## Parallelism Support

This table shows the current parallelism support status.

--8<-- "docs/includes/parallelism.md"

## Quantization Support

This table shows the current quantization support status.

--8<-- "docs/includes/quantization.md"
