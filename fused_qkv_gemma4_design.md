# Fused QKV Projections for Gemma 4

**Author:** Jiries Kaileh  
**Last major update:** Jun 8, 2026  
**Bug:** [Gemma 4 Headroom] Implement Fused QKV Projections to Reduce HBM-to-VM Redundancy  
**Status:** In Review  

#begin-approvals-addon-section
| Username | Role | Status | Last change |
| :--- | :--- | :--- | :--- |
| lkchen | Approver | 🟡 Pending | - |

Approval Instructions: Please approve or LGTM through the G3 Assist sidebar.
For more information, see go/g3a-approvals-reviewing
#end-approvals-addon-section

## Overview
Currently, for Gemma 4, Q, K, and V projections are handled as separate operations. This requires the model to load the same input data from HBM to VMEM three separate times; once for each projection.

## Goals
To get past this issue, we will fuse the Q, K, and V projections into a single operation. By merging these, the input data is loaded from HBM into VMEM once, and all weight sums for Q, K, and V are performed simultaneously.

Furthermore, we will **consolidate the QKV parallel linear projection processing kernel** inside `layers/common` so that both JAX-native models (Flax NNX) and Torchax models (PyTorch-on-TPU) reuse the exact same optimized sharding, slicing, and head-replication SPMD algorithm.

---

## Implementation Details

We will create a JAX-native fused linear layer, `JaxQKVParallelLinear`, integrate it within `Gemma4Attention` on standard/local layers, and update the weight-loading infrastructure to rearrange and load checkpoint weights without triggering inter-chip communication.

### 1. Centralized Projection Kernel (`layers/common/utils.py`)

The core sharding, SRAM slicing, and GQA/MQA head replication logic is consolidated under `process_sharded_qkv_projection` in `tpu_inference/layers/common/utils.py`. Both Torchax and Flax call this centralized helper:

```python
def process_sharded_qkv_projection(
    out_jax: jax.Array,
    output_sizes: list[int],
    tp_size: int,
    num_kv_head_replicas: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Consolidated JAX/SPMD kernel that reorders, slices, and replicates sharded KV heads when TP > num_kv_heads."""
    from jax.sharding import PartitionSpec as P
    from jax.experimental.shard_map import shard_map
    from tpu_inference.layers.common.sharding import ShardingAxisName
    from tpu_inference.utils import get_mesh_shape_product

    # 1. Reorder concatenated tensor along the sharded axis
    out_jax = reorder_concatenated_tensor_for_sharding(
        out_jax,
        output_sizes,
        tp_size,
        dim=-1,
    )

    # 2. Slice sharded tensor into Q, K, and V
    q_jax, k_jax, v_jax = slice_sharded_tensor_for_concatenation(
        out_jax,
        output_sizes,
        tp_size,
    )

    # 3. Handle KV head replication if replicas > 1
    if num_kv_head_replicas > 1:
        mesh = jax.sharding.get_abstract_mesh()
        kv_head_axis = None
        for a in reversed(mesh.axis_names):
            if a in ShardingAxisName.ATTN_HEAD and get_mesh_shape_product(
                    mesh, a) >= num_kv_head_replicas:
                kv_head_axis = a
                break

        if kv_head_axis is None:
            raise ValueError(
                f"Cannot find a mesh axis to split for KV-head replication: "
                f"no axis in {mesh.axis_names} contains "
                f"{ShardingAxisName.ATTN_HEAD} and has size >= {num_kv_head_replicas}")

        replica_axis = 'replica'
        data_axis = ShardingAxisName.ATTN_DATA
        head_axis = ShardingAxisName.ATTN_HEAD

        i = mesh.axis_names.index(kv_head_axis)
        kv_size = mesh.axis_sizes[i]
        kv_type = mesh.axis_types[i]
        new_mesh = jax.sharding.AbstractMesh(
            mesh.axis_sizes[:i] + (kv_size // num_kv_head_replicas, num_kv_head_replicas) +
            mesh.axis_sizes[i + 1:],
            mesh.axis_names[:i] + (kv_head_axis, replica_axis) +
            mesh.axis_names[i + 1:],
            mesh.axis_types[:i] + (kv_type, kv_type) + mesh.axis_types[i + 1:],
        )
        if isinstance(head_axis, tuple):
            in_head_axis = list(head_axis)
            in_head_axis.insert(
                in_head_axis.index(kv_head_axis) + 1, replica_axis)
        else:
            in_head_axis = (head_axis, replica_axis)

        @shard_map(mesh=new_mesh,
                   in_specs=P(data_axis, in_head_axis),
                   out_specs=P(data_axis, head_axis),
                   check_vma=False)
        def _mark_kv_head_replicated(t):
            return t

        with jax.sharding.use_abstract_mesh(new_mesh):
            k_jax = _mark_kv_head_replicated(k_jax)
            v_jax = _mark_kv_head_replicated(v_jax)

    return q_jax, k_jax, v_jax
```

---

### 2. Flax Module Implementation (`layers/jax/linear.py`)

The new `JaxQKVParallelLinear` class is added to `tpu_inference/layers/jax/linear.py`. It inherits from `JaxModule` and encapsulates a single `JaxEinsum` projection while reusing our consolidated projection helper:

```python
class JaxQKVParallelLinear(JaxModule):
    """Fused QKV Parallel Linear layer for JAX-native models.

    Performs fused Q, K, and V projections in a single HBM read pass and
    partitions them locally per TPU device without incurring all-to-all collectives.
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 use_bias: bool,
                 dtype: jnp.dtype,
                 rngs: nnx.Rngs,
                 tp_size: int,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_bias = use_bias
        self.tp_size = tp_size

        self.q_size = num_heads * head_dim
        self.k_size = num_kv_heads * head_dim
        self.v_size = num_kv_heads * head_dim
        self.total_output_dim = self.q_size + self.k_size + self.v_size

        # Output dimension is sharded along the "model" (TP) axis
        self.proj = JaxEinsum(
            "TD,DV->TV",
            (hidden_size, self.total_output_dim),
            bias_shape=(self.total_output_dim, ) if use_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.uniform(),
                                              (None, "model")),
            bias_init=nnx.with_partitioning(nnx.initializers.uniform(),
                                            ("model", )) if use_bias else None,
            rngs=rngs,
            quant_config=quant_config,
            prefix=prefix + ".qkv_proj",
        )

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        from tpu_inference.layers.common.utils import \
            process_sharded_qkv_projection

        # Single projection operation (loads inputs from HBM once)
        outs = self.proj(x)  # Shape: [T, total_output_dim] sharded on "model" axis

        # Slice using the consolidated layers/common kernel
        split_sizes = [self.q_size, self.k_size, self.v_size]
        q_sharded, k_sharded, v_sharded = process_sharded_qkv_projection(
            outs, split_sizes, self.tp_size
        )

        # Reshape directly to their global multi-head shapes
        q = q_sharded.reshape(outs.shape[:-1] + (self.num_heads, self.head_dim))
        k = k_sharded.reshape(outs.shape[:-1] + (self.num_kv_heads, self.head_dim))
        v = v_sharded.reshape(outs.shape[:-1] + (self.num_kv_heads, self.head_dim))

        return q, k, v
```

---

### 3. Torchax Reuse (`layers/vllm/custom_ops/linear.py`)

Torchax's custom `VllmQKVParallelLinear.forward` method uses the identical sharding, slicing, and replication helper:

```python
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.num_kv_head_replicas == 1:
            return super().forward(x)

        out, bias = super().forward(x)
        out_jax = jax_view(out)

        from tpu_inference.layers.common.utils import \
            process_sharded_qkv_projection

        # Call the consolidated common JAX helper
        q_jax, k_jax, v_jax = process_sharded_qkv_projection(
            out_jax,
            self.output_sizes,
            self.tp_size,
            num_kv_head_replicas=self.num_kv_head_replicas,
        )

        out_jax = jnp.concatenate([q_jax, k_jax, v_jax], axis=-1)
        return torch_view(out_jax), bias
```

---

### 4. Integration with Gemma4Attention

We replace separate projections inside `Gemma4Attention` (in `gemma4.py`) with our new consolidated `JaxQKVParallelLinear` module only for sliding window/local attention layers where $Q, K, V$ are symmetrical and fully divisible by `tp_size` (`use_k_eq_v = False`).

For shared-KV global attention layers (`use_k_eq_v = True`), we keep them separate as `q_proj` and `k_proj` submodules (with $V=K$). This completely prevents layout-conversion collectives when `num_kv_heads < tp_size` during decode generation, retaining our full performance headroom of **25.7k Output Token Throughput**.

---

## Verification & Testing Strategy

1. **Unit and Correctness Testing:**
   * Run unit tests (`tests/layers/jax/test_linear.py`) verifying that the consolidated `slice_and_replicate_qkv_sharded_tensor` logic matches mathematical separate projections under multiple CPU TP sharding meshes.
2. **Integration and PP Loading Testing:**
   * Run the full integration loading test suite in `tests/models/jax/test_gemma4.py` verifying model instantiation, weight mapping, load-time caching and fusion, and multi-rank pipeline parallel execution (PP=1, PP=4) on TPU.

---

## Serving Performance Results

To evaluate the headroom gains, we benchmarked the highly optimized fused QKV model under a high-concurrency production load of **5,120 requests** using an FP8/QWIX-quantized Gemma-4-26B-A4B-it checkpoint.

Our highly optimized zero-copy JAX SPMD design outperformed the baseline un-fused run by a wide margin across all major latency and throughput metrics:

| Metric | Baseline (Unfused) | Our Optimized Fused QKV | Performance Delta | Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Output Token Throughput (tok/s)** | 23,999.56 | **25,712.36** | 🚀 **+7.14%** | **+1,712.80 extra tokens/sec** |
| **Total Token Throughput (tok/s)** | 73,150.65 | **78,371.29** | 🚀 **+7.14%** | **+5,220.64 extra tokens/sec** |
| **Request Throughput (req/s)** | 48.00 | **51.42** | 🚀 **+7.13%** | **More parallel requests/sec** |
| **Mean Time to First Token (TTFT)** | 27,894.21 ms | **24,456.54 ms** | ⚡ **-12.32%** | **Saves ~3.43 seconds of latency** |
| **Median Time to First Token (TTFT)** | 17,496.51 ms | **14,568.29 ms** | ⚡ **-16.74%** | **Saves ~2.93 seconds of latency** |
| **Mean Time per Output Token (TPOT)** | 108.00 ms | **103.88 ms** | ⚡ **-3.81%** | **Shaves ~4.12 ms off every token** |
| **Median Time per Output Token (TPOT)** | 115.65 ms | **110.85 ms** | ⚡ **-4.15%** | **Shaves ~4.80 ms off every token** |
| **Benchmark Duration (s)** | 106.67 s | **99.56 s** | ⚡ **-6.67%** | **Benchmark finished 7.11s faster** |
