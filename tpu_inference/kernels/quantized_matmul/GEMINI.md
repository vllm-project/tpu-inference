# Gemini Code Assistant Context

This document provides context for the Gemini code assistant regarding the `quantized_matmul` project.

## Project Overview

This project implements high-performance quantized matrix multiplication kernels for Google TPUs using JAX and its Pallas extension. The primary goal is to provide efficient, low-level routines for deep learning inference.

The directory contains two main kernel implementations:

1.  **`kernel.py`**: A standard quantized matrix multiplication kernel. It handles per-tensor quantization of activations and uses pre-quantized weights.
2.  **`kernel_2d.py`**: A 2D (block-wise or sub-channel) quantized matrix multiplication kernel. This version offers potentially higher accuracy by breaking the reduction dimension into smaller blocks, each with its own quantization scale.

Both kernels are heavily optimized for performance by using a lookup table of pre-tuned block sizes for various matrix shapes, data types, and TPU versions.

## Key Technologies

*   **JAX**: The primary framework for numerical computation and automatic differentiation.
*   **Pallas**: A JAX extension for writing low-level, high-performance kernels for TPUs.
*   **Python**: The implementation language.

## Building and Running

This is a library project, and as such, there are no top-level commands for building or running the code. The kernels are intended to be imported and used within a larger JAX-based application.

**Example Usage (inferred):**

```python
import jax.numpy as jnp
from tpu_inference.kernels.quantized_matmul import quantized_matmul_kernel

# Create dummy data
x = jnp.ones((1024, 4096))
w_q = jnp.ones((4096, 4096), dtype=jnp.int8)
w_scale = jnp.ones((4096,))

# Run the kernel
result = quantized_matmul_kernel(x, w_q, w_scale)
```

## Development Conventions

*   **Coding Style**: The code follows standard Python conventions with type hints.
*   **Licensing**: All files have an `SPDX-License-Identifier: Apache-2.0` header.
*   **Modularity**: The code is organized into separate files for kernels, utilities, and tuned parameters.
*   **Performance**: The `tuned_block_sizes.py` file contains a large, pre-computed table of optimal block sizes for different hardware and input shapes, which is critical for performance.
*   **Testing**: No test files are present in this directory, but the code structure suggests that testing would be done in a separate test directory, likely invoking the main kernel functions with various input shapes and data types.
