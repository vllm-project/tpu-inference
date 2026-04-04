# TPU Model Export

This guide describes how to export JAX models to the serialized StableHLO format for TPU inference.

## Overview

The Offline Model Export feature allows you to export a model's computational graph to a format that can be loaded and served by the TPU inference system. This is done on a CPU environment by tracing the model with dummy weights.

## Usage

To export a model, run the `examples/offline_export.py` script.

```bash
TPU_INFERENCE_EXPORT_PATH=/path/to/export/dir TPU_INFERENCE_EXPORT_TOPOLOGY=TPU7x:2x4 python examples/offline_export.py
```

### Environment Variables

*   `TPU_INFERENCE_EXPORT_PATH`: (Required) The directory where the exported model will be saved.
*   `TPU_INFERENCE_EXPORT_TOPOLOGY`: (Required) The target TPU topology (e.g., `TPU7x:2x4`).

## How it Works

The export process involves the following steps:

1.  **Dummy Loading**: The model is loaded using `vllm.LLM` with `load_format="dummy"` to avoid loading real weights, making it possible to run on a CPU.
2.  **Traced Export**: `CompilationManager` intercepts the model functions and uses `jax.export.export` to trace and serialize them.
3.  **Deduplication**: Exported functions are hashed (MD5), and duplicates are skipped to save space.
4.  **Format**: Each unique function is saved in its own subdirectory containing `metadata.json` and `.jax_exported` files.

## Troubleshooting

### Serialization Error: Integer-keyed dictionaries

If you see an error like:
`Serialization is supported only for dictionaries with string keys. Found key 0 of type <class 'int'>.`

This means a dictionary with integer keys was found in the PyTree metadata of the exported function. JAX serialization only supports string keys.

**Solution**:
Identify the data structure containing the integer keys and register it using `jax.export.register_pytree_node_serialization`. Alternatively, modify the code to use string keys if feasible.
