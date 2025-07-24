# MLA (Multi-Level Attention) vs Normal Attention: Implementation Summary

## Overview

MLA (Multi-Level Attention) is a specialized attention mechanism that uses **latent KV vectors** and **up-projection** instead of storing separate K and V tensors. This document summarizes the key differences in implementation between MLA and normal attention.

## Key Differences

### 1. KV Cache Structure

#### Normal Attention
```python
# KV Cache: (L, S, 2*K, H)
# - L: number of blocks
# - S: block size  
# - 2*K: 2 * num_kv_heads (K and V stored separately)
# - H: head_dim

kv_cache = jnp.zeros((num_blocks, block_size, 2*num_kv_heads, head_dim))
```

#### MLA Attention
```python
# KV Cache: (L, S, 1, H) 
# - L: number of blocks
# - S: block size
# - 1: single latent vector (not separate K/V)
# - H: head_dim (padded to multiple of 128) [latent head_dim = kv_lora_rank + pk_rope_dim + padding]

kv_cache = jnp.zeros((num_blocks, block_size, 1, head_dim))
```

### 2. KV Cache Update Process

#### Normal Attention
```python
def update_kv_cache(k, v, kv_cache, slices, num_slices, mesh):
    # 1. Concatenate K and V along feature dimension
    kv = jnp.concat([k, v], axis=-1)  # (T, K*2, H)
    
    # 2. Update cache using standard kernel
    kv_cache = kv_cache_update(kv, slices, kv_cache, num_slices, mesh)
    
    return kv_cache
```

#### MLA Attention
```python
def update_mla_kv_cache(latent_kv, k_rope, kv_cache, slices, num_slices, mesh):
    # 1. Concatenate latent_kv and k_rope
    latent_kv_k_rope = jnp.concat([latent_kv, k_rope], axis=-1)
    
    # 2. Pad to multiple of 128 for TPU tiling
    padding_needed = 128*2 - latent_kv_k_rope.shape[-1] % (128*2)
    latent_kv_k_rope = jnp.pad(latent_kv_k_rope, ((0,0), (0,0), (0, padding_needed)))
    
    # 3. Reshape to match tiling expectations (2nd last dimension must be divisible by 2)
    latent_kv_k_rope = latent_kv_k_rope.reshape(latent_kv_k_rope.shape[0], 2, -1)
    kv_cache = kv_cache.reshape(-1, 2, H//2)
    
    # 4. Update cache using existing kernel
    kv_cache = kv_cache_update(latent_kv_k_rope, slices, kv_cache, num_slices, mesh)
    
    return kv_cache.reshape(L, S, 1, H)
```

### 3. Attention Computation

#### Normal Attention
```python
def normal_attention(q, kv_cache, ...):
    # 1. Extract K and V directly from cache
    k = kv_cache[..., :num_kv_heads, :]      # (L, S, K, H)
    v = kv_cache[..., num_kv_heads:, :]      # (L, S, K, H)
    
    # 2. Compute flash attention 
    output = flash_attention(q,k,v)
    
    return output
```

#### MLA Attention
```python
def mla_attention(q, k_rope, kv_cache, kv_up_proj_weights, ...):
    # 1. Extract latent KV vectors from cache
    latent_kv = kv_cache  # (L, S, 1, H)
    
    # 2. Perform up-projection to get K and V
    projected_kv = mla_up_projection(latent_kv, kv_up_proj_weights)
    # projected_kv: (num_kv_per_blk, num_heads, qk_nope_head_dim + v_head_dim)
    
    # 3. Split into K and V components
    k_nope = projected_kv[..., :qk_nope_head_dim]
    v = projected_kv[..., qk_nope_head_dim:]
    
    # 4. Combine k_nope with k_rope to form full key
    k = jnp.concatenate([k_nope, k_rope], axis=-1)
    
    # 5. Pad to TPU tiling constraints
    k = jnp.pad(k, ((0,0), (0, 128 - k.shape[1]%128)))
    v = jnp.pad(v, ((0,0), (0, 128 - v.shape[1]%128)))
    
    # 6. Compute flash attention
    output = flash_attention(q,k,v)
    
    return output
```

### 4. Up-Projection Function

#### MLA Up-Projection
```python
def mla_up_projection(latent_kv, kv_up_proj_weights):
    """
    Args:
        latent_kv: (num_kv_per_blk, kv_lora_rank) - Latent vectors
        kv_up_proj_weights: (kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim) - Up-projection weights
    
    Returns:
        projected_kv: (num_kv_per_blk, num_heads, qk_nope_head_dim + v_head_dim)
    """
    # Use einsum for efficient tensor contraction
    projected = lax.dot_general(
        latent_kv, 
        kv_up_proj_weights[...],
        dimension_numbers=(([1], [0]), ((), ())),
        preferred_element_type=jnp.float32 
    )
    return projected
```

## Implementation Details

### 1. Memory Layout

#### Normal Attention
```
KV Cache: [L, S, 2*K, H]
         |<-- K -->|<-- V -->|
         |         |         |
         |  Keys   | Values  |
```

#### MLA Attention
```
KV Cache: [L, S, 1, H]
         |<-- latent_kv + k_rope + padding -->|
         |                                    |
         |  Latent vectors + k_rope + pad    |
```

### 2. Weight Absorption

**Weight absorption** is a performance trick where the up-projection weights are integrated directly into the flash attention kernel:
- **Benefits**: Reduce computation overhead
- **Implementation**: Need to update `qk` computation and `v` computation

I haven't tried this trick yet but it seems very doable. 

### 4. Head Dimension Handling

#### Normal Attention
- **K and V**: Same head dimension
- **Q**: Same head dimension as K/V

#### MLA Attention
- **Q**: `qk_nope_head_dim + qk_rope_head_dim`
- **K**: `qk_nope_head_dim + qk_rope_head_dim` (concatenated)
- **V**: `v_head_dim` (can be different from K)
- **Latent**: `kv_lora_rank` (compressed representation)


## Key Files Modified

1. **`tpu_commons/kernels/ragged_paged_attention/mla_kernel.py`**
   - MLA-specific attention kernel
   - Up-projection implementation
   - Weight absorption support

2. **`tpu_commons/models/jax/attention_interface.py`**
   - `update_mla_kv_cache()` function
   - MLA attention interface

3. **`tpu_commons/models/jax/common/attention/deepseek_v3_attention.py`**
   - MLA attention module
   - Integration with existing attention system

## Summary

MLA attention introduces a **latent representation** for KV cache that requires:
1. **Up-projection** to recover K and V from latent vectors
2. **Weight absorption** for efficient TPU execution
3. **Careful memory layout** to respect TPU constraints
4. **Additional computation** but reduced memory usage

The implementation maintains compatibility with existing attention infrastructure while adding MLA-specific optimizations.

## Open Questions

### 1. Sharding Strategy

**Current Implementation:**
- **Normal Attention**: Shards along `num_kv_heads` dimension
- **MLA Attention**: Shards along `head_dim` dimension

**Questions:**
- **Is sharding by `head_dim` optimal for MLA?** Normal attention shards by heads because each head can be processed independently. With MLA, the latent vectors are shared across heads, so sharding by `head_dim` might not provide the same parallelism benefits.
- **What's the optimal sharding strategy for up-projection?** The up-projection weights have shape `(kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim)` - should we shard along `num_heads` duplicate?

**Potential Issues:**
- Sharding by `head_dim` might create load imbalance if different heads have different computational requirements

### 2. Pallas Block Size Tuning

**Current Situation:**
- **Normal Attention**: Uses tuned block sizes optimized for `(num_kv_heads, head_dim)` tensors
- **MLA Attention**: Latent KV cache is much smaller  

**Questions:**
- **Do we need different block sizes for MLA?** The current `get_tuned_block_sizes()` function is designed for normal attention. MLA's different tensor shapes might require different optimal block sizes.
- **What's the optimal `num_kv_pages_per_block` for MLA?** There is NO concept of kv heads in MLA. 

### 3. KV Cache Size and Tiling Constraints

**Current Implementation:**
- **KV Cache Shape**: `(L, S, 2, H)` where `H` is padded to multiple of 128
- **Tiling**: Uses `128*2` padding for TPU constraints

**Questions:**
- **What's the optimal KV cache shape for MLA?** The current `(L, S, 2, H)` shape was chosen to work with existing kernels, but might not be optimal.

