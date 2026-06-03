# TPU Inference Environment Variables Reference

This reference documents the custom environment variables introduced by the `tpu-inference` unified backend, supplementing the standard [vLLM Environment Variables](https://docs.vllm.ai/en/latest/api/vllm/envs/). These variables allow you to tune hardware allocation, model execution, compilation behavior, kernel performance, memory usage, and profiling when serving models on Google TPUs with vLLM.

## Scope / Path Definitions
Environment variables are scoped to specific model execution pathways:

* **Shared**: Applies to both native JAX/Flax model implementations (`MODEL_IMPL_TYPE="flax_nnx"`) and PyTorch/vLLM models run via `torchax` (`MODEL_IMPL_TYPE="vllm"`).
* **JAX/Flax**: Applies only to native JAX/Flax model definitions.
* **PyTorch/vLLM (torchax)**: Applies only to PyTorch models run via `torchax` JIT compilation.

---

## 1. Hardware & Platform Configuration

| Variable | Type | Default | Scope | Description |
| :--- | :--- | :--- | :--- | :--- |
| `JAX_PLATFORMS` | String | `""` | Shared | Selection of JAX platforms (e.g. `"tpu"`, `"cpu"`, `"proxy"`, `"proxy,cpu"`). |
| `TPU_ACCELERATOR_TYPE` | String | `None` | Shared | The specific TPU chip version and pod size configuration (e.g., `"v5litepod-16"`, `"v6e-8"`). |
| `TPU_NAME` | String | `None` | Shared | Name/ID of the TPU resource. |
| `TPU_WORKER_ID` | String | `None` | Shared | The index worker ID assigned to multi-host TPU VM setups. |
| `TPU_MULTIHOST_BACKEND` | String | `""` | Shared | Backend for multi-host VM communications. Currently supports `"ray"`. |

---

## 2. Serving Execution & Parallelism

| Variable | Type | Default | Scope | Description |
| :--- | :--- | :--- | :--- | :--- |
| `TPU_MULTIPROCESS_DP` | Boolean | `false` | Shared | Enable vLLM-native multi-process data parallelism (one engine process per DP rank) instead of single-process SPMD. Pins each DP replica to disjoint TPU chips. Dense models only. |
| `MODEL_IMPL_TYPE` | String | `"auto"` | Shared | Model implementation source. Options: `"auto"`, `"vllm"`, `"flax_nnx"`, `"jetpack"`. |
| `DRAFT_MODEL_IMPL_TYPE`| String | `"auto"` | Shared | Model implementation source for the draft model (used in speculative decoding). |
| `USE_2D_TP` | Boolean | `false` | Shared | Enable 2D tensor parallelism, sharding attention heads across multiple mesh axes. |
| `NEW_MODEL_DESIGN` | Boolean | `false` | Shared | Enable the multi-dimensional sharding mesh (up to 6D) and sharding layout strategies required for Attention DP, Context Parallelism (CP), and MLA (Multi-Head Latent Attention) model architectures. |
| `VLLM_TPU_PATCH_MM_EMBEDDINGS` | Boolean | `false` | PyTorch/vLLM (torchax) | Patch vLLM's multi-modal embedding merge logic with static JAX-compatible math operations to prevent dynamic compilation deadlocks during multi-modal model execution. |
| `NUM_SLICES` | Integer | `1` | Shared | Number of TPU slices for multi-slice mesh topologies. |
| `DP_SCHED_BATCH_PREFILL`| Boolean | `true` | Shared | Hold and batch incoming requests (prefills) to schedule and dispatch them together across DP replicas. |
| `DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS` | Integer | `30000` | Shared | Timeout in milliseconds to force-flush pending requests when batch prefilling is enabled. |

---

## 3. Disaggregated Serving (Prefill/Decode Separation)

| Variable | Type | Default | Scope | Description |
| :--- | :--- | :--- | :--- | :--- |
| `PREFILL_SLICES` | String | `""` | Shared | Mesh configuration indicating slices allocated to prefill-only workers. |
| `DECODE_SLICES` | String | `""` | Shared | Mesh configuration indicating slices allocated to decode-only workers. |

---

## 4. JAX/XLA Graph Compilation

| Variable | Type | Default | Scope | Description |
| :--- | :--- | :--- | :--- | :--- |
| `SKIP_JAX_PRECOMPILE` | Boolean | `false` | Shared | Skip JAX/XLA graph tracing and compilation during model initialization (useful for developer iteration). |
| `VLLM_XLA_CHECK_RECOMPILATION` | Boolean | `false` | Shared | Assert and crash/log if JAX re-compiles graphs during model execution. |
| `JITTED_MM_MODULE_KEYS`| List | `[]` | PyTorch/vLLM (torchax) | List of module keys (comma-separated) to jit compile explicitly. |
| `REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES` | List | `[]` | PyTorch/vLLM (torchax) | List of custom PyTree classes to register with JAX. |

---

## 5. Memory & KV Cache Offloading (DRAM Offload)

| Variable | Type | Default | Scope | Description |
| :--- | :--- | :--- | :--- | :--- |
| `TPU_OFFLOAD_SKIP_JAX_PRECOMPILE` | Boolean | `false` | Shared | Skip precompilation of memory swap-related JAX functions. |
| `TPU_OFFLOAD_DECODE_SAVE` | Boolean | `false` | Shared | Offload and save the KV cache state during the decode phase. |
| `TPU_OFFLOAD_NUM_CPU_CHUNKS` | Integer | `1024` | Shared | DRAM space allocation size in CPU memory chunks/blocks. |
| `TPU_OFFLOAD_NUM_STAGING_BLOCKS` | Integer | `128` | Shared | Size of the staging buffer (HBM) reserved for memory swapping. |
| `TPU_OFFLOAD_SAVE_THREADS` | Integer | `1` | Shared | Number of threads dedicated to asynchronous TPU-to-CPU data transfer. |
| `TPU_OFFLOAD_BATCHED_SAVE` | Boolean | `false` | Shared | Batch multiple requests' save operations into a single swap call. |
| `TPU_OFFLOAD_METRICS_LOG_INTERVAL` | Integer | `10` | Shared | Prometheus metric log interval in seconds for offloading stats. |
| `TPU_OFFLOAD_USE_UNPINNED_HOST` | Boolean | `false` | Shared | Use unpinned host memory for KV cache tensors residing on host DRAM. |

---

## 6. Attention & Kernel Tuning

| Variable | Type | Default | Scope | Description |
| :--- | :--- | :--- | :--- | :--- |
| `USE_BATCHED_RPA_KERNEL` | Boolean | `false` | Shared | Enable the batched Ragged Paged Attention (RPA) kernel. Solves VMEM OOMs on long context lengths (e.g. 8192). |
| `ATTN_BUCKETIZED_NUM_REQS` | Boolean | `false` | Shared | Precompile attention for power-of-two batch sizes between min and max requests. Yields faster execution but increases startup time. |
| `ATTN_CUSTOM_NUM_REQS_BUCKETS` | List | `[]` | Shared | Custom comma-separated list of batch sizes to compile attention for, reducing startup times compared to full power-of-two compilation. |
| `LAYOUT_Q_PROJ_AS_NDH`| Boolean | `false` | JAX/Flax | Layout query projections as NDH `[heads, model_dim, head_dim]` instead of DNH `[model_dim, heads, head_dim]`. |
| `RAGGED_GATED_DELTA_RULE_IMPL` | String | `"chunked_jax_pd"`| Shared | Kernel implementation for Ragged Gated Delta Rule (`"chunked_jax_pd"`, `"chunked_kernel_pd"`, `"chunked_kernel_p_recurrent_kernel_d"`). |
| `MLA_XPOSE_N_TILE_SIZE` | Integer | `160` | Shared | Tile size dimension for transpose steps inside MLA kernels. |
| `ONEHOT_MOE_PERMUTE_THRESHOLD` | Integer | `0` | Shared | Batch size threshold below which Onehot+Matmul is used for permuting MoE routing. `0` disables this path. |
| `MLA_KV_PACKING_SIZE` | Integer | `32` | Shared | Packing size parameter for Multi-Head Latent Attention KV cache representation. |

---

## 7. Mixture of Experts (MoE) Execution

| Variable | Type | Default | Scope | Description |
| :--- | :--- | :--- | :--- | :--- |
| `USE_MOE_EP_KERNEL` | Boolean | `false` | Shared | Use custom expert-parallel (EP) kernel for Mixture of Experts. |
| `USE_UNFUSED_MEGABLOCKS` | Boolean | `false` | JAX/Flax | Use megablocks sparse matrix multiplies for MoE with unfused weights. |
| `USE_DENSE_MOE` | Boolean | `false` | JAX/Flax | Use the dense backend mock implementation for MoE (unrecommended for production). |
| `FORCE_MOE_RANDOM_ROUTING` | Boolean | `false` | Shared | Force random routing of tokens to experts (useful for testing only). |
| `MOE_ALL_GATHER_ACTIVATION_DTYPE` | String | `""` | Shared | Precision datatype for MoE all-gather activations. |
| `MOE_APPROX_TOPK` | Boolean | `false` | Shared | Use approximate top-k selection for expert routing to speed up calculations. |
| `MOE_APPROX_TOPK_RECALL_TARGET` | Float | `0.9` | Shared | Recall target metric (e.g. `0.95`) for approximate top-k expert selection. |
| `ENABLE_RS_KERNEL` | Boolean | `false` | Shared | Enable hierarchical reduce-scatter kernel execution inside MoE. |

---

## 8. Quantization Levers

| Variable | Type | Default | Scope | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ENABLE_QUANTIZED_MATMUL_KERNEL` | Boolean | `false` | Shared | Enable experimental quantized matrix multiply kernels (W8A8/FP8 paths). |
| `REQUANTIZE_BLOCK_SIZE` | Integer | `None` | Shared | Block size specification for quantization scales. |
| `REQUANTIZE_WEIGHT_DTYPE` | String | `"float8_e4m3fn"`| Shared | Weight data type format for quantized linear operations. |
| `MOE_REQUANTIZE_WEIGHT_DTYPE` | String | `""` | Shared | Weight data type format for quantized MoE operations. |
| `MOE_REQUANTIZE_BLOCK_SIZE` | Integer | `None` | Shared | Requantization block size parameter for MoE expert layers. |
| `DISABLE_WEIGHT_REQUANTIZATION` | Boolean | `false` | Shared | Globally disable dynamic weight requantization. |
| `DISABLE_MLA_Q_ACTIVATION_QUANTIZATION` | Boolean | `false` | PyTorch/vLLM (torchax) | Disable MLA query activation quantization. |

---

## 9. Profiling & Debugging

| Variable | Type | Default | Scope | Description |
| :--- | :--- | :--- | :--- | :--- |
| `PHASED_PROFILING_DIR`| String | `""` | Shared | Directory location where phased profiling runs are saved. |
| `PYTHON_TRACER_LEVEL` | Integer | `1` | Shared | Logger trace level for internal python executors. |
| `USE_JAX_PROFILER_SERVER` | Boolean | `false` | Shared | Start the JAX profiling server during execution. |
| `JAX_PROFILER_SERVER_PORT` | Integer | `9999` | Shared | Network port for the JAX profiling server. |
| `AGGREGATED_STATS_DIR`| String | `""` | Shared | Output directory location for storing aggregated execution stats. |
| `PROFILE_SINGLE_DEVICE` | Boolean | `false` | Shared | Limit XLA tracing profiling to a single TPU device index. |
| `LORA_MODULE_PATH` | String | `""` | PyTorch/vLLM (torchax) | System directory path containing compiled LoRA weight adapters. |
