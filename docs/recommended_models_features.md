# Recommended Model and Feature Matrices

Although vLLM TPU’s new unified backend makes out-of-the-box high performance serving possible with any model supported in vLLM, the reality is that we're still in the process of implementing a few core components.
For this reason, until we land more capabilities, we recommend starting from this list of stress tested models and features below.

We are still landing components in tpu-inference that will improve performance for larger scale, higher complexity models (XL MoE, +vision encoders, MLA, etc.).

If you’d like us to prioritize something specific, please submit a GitHub feature request [here](https://github.com/vllm-project/tpu-inference/issues/new/choose).

<details open markdown="1">
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

=== "Release"

    --8<-- "docs/includes/model_support.md"

=== "Nightly"

    --8<-- "docs/includes/nightly_model_support.md"

### Embedding Models

=== "v7x"

    {{ read_csv('support_matrices/release/v7x/default/model_support_matrix.csv') }}

=== "v6e"

    {{ read_csv('support_matrices/release/v6e/default/model_support_matrix.csv') }}

## Recommended Features

This table shows the features currently tested for accuracy and performance.

=== "Release"

    --8<-- "docs/includes/core_features.md"

=== "Nightly"

    --8<-- "docs/includes/nightly_core_features.md"

## Kernel Support

This table tracks high-level correctness and performance validation for distributed compute kernels.

--8<-- "docs/includes/kernel_support.md"

## Microbenchmark Kernel Support

This section outlines the detailed hardware and precision validation for our core microbenchmark kernels.

=== "Release"

    --8<-- "docs/includes/microbenchmarks.md"

=== "Nightly"

    --8<-- "docs/includes/nightly_microbenchmarks.md"

## Parallelism Support

This table shows the current parallelism support status.

=== "Release"

    --8<-- "docs/includes/parallelism.md"

=== "Nightly"

    --8<-- "docs/includes/nightly_parallelism.md"

## Quantization Support

This table shows the current quantization support status.

=== "Release"

    --8<-- "docs/includes/quantization.md"

=== "Nightly"

    --8<-- "docs/includes/nightly_quantization.md"
