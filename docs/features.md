# Feature Matrix

This page details the features, kernels, parallelism schemes, and quantization methods currently tested for accuracy and performance.

<details open markdown="1">
<summary> <b>🚦 <i>Status Legend</i> </b> </summary>

> - ✅ **Passing:** Tested and works as expected. Ready for use.
> - ❌ **Failing:** Known to be broken or not functional. Help is wanted to fix this!
> - 🧪 **Experimental:** Works, but unoptimized or pending community validation.
> - 📝 **Planned:** Not yet implemented, but on the official roadmap.
> - ⛔️ **Unplanned:** There is no benefit to adding this.
> - ❓ **Untested:** The functionality exists but has not been recently or thoroughly verified.
</details>

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
