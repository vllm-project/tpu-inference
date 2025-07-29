import jax
import jax.numpy as jnp
import numpy as np

from tpu_commons.kernels.ragged_kv_cache_update import _dynamic_validate_inputs, kv_cache_update
from jax.sharding import PartitionSpec as P

# Generate test data
page_size = 64
page_num = 512
combined_kv_head_num = 16
head_dim = 128
padded_num_tokens = 128
padded_num_slices = 256


# Create PRNG key for reproducible random data
prng_key = jax.random.key(1234)

# Initialize KV cache (empty initially)
kv_cache = jnp.zeros(
    (page_num * page_size, combined_kv_head_num, head_dim),
    dtype=jnp.bfloat16
)

# Shard with DP and TP
mesh = jax.sharding.Mesh(
    np.array(jax.devices()).reshape(2,2),
    axis_names=("data", "model")
)

# New KV data to be inserted
new_kv = jax.random.normal(
    prng_key, (padded_num_tokens, combined_kv_head_num, head_dim),
    dtype=jnp.bfloat16
)

# Define where in the KV cache each slice should go
kv_cache_start_indices = np.array([
    64,   
    128,
    192,        
    256,   
], dtype=np.int32)

# Define where in the new_kv data each slice starts
new_kv_cache_indices = np.array([
    0, 
    32,
    64,      
    96,          
    # 130,        
], dtype=np.int32)

# Define slice lengths for different sequences
slice_lens = np.array([32, 32, 32, 32], dtype=np.int32)
num_slices = jnp.array([len(new_kv_cache_indices)//2, len(new_kv_cache_indices)//2], dtype=np.int32)

# Calculate sharding offsets for data parallelism
# Since we're using shard_map, each device will get local indices
# The indices should be relative to each device's local data range
num_devices_data = mesh.shape["data"]  # 2 devices along data axis
kv_cache_tokens_per_device = (page_num * page_size) // num_devices_data
new_kv_tokens_per_device = padded_num_tokens // num_devices_data

print(f"KV cache tokens per device: {kv_cache_tokens_per_device}")
print(f"New KV tokens per device: {new_kv_tokens_per_device}")
print(f"Original kv_cache_start_indices: {kv_cache_start_indices}")
print(f"Original new_kv_cache_indices: {new_kv_cache_indices}")

# Adjust indices to be local to each device's shard
# For device 0: kv_cache range [0, kv_cache_tokens_per_device)
# For device 1: kv_cache range [kv_cache_tokens_per_device, 2*kv_cache_tokens_per_device)
# But in shard_map, each device sees indices [0, kv_cache_tokens_per_device)

# We need to map global indices to local indices
# Assuming the slices are meant for device 0 (adjust as needed)
device_id = 0  # This example assumes slices are for the first device
kv_cache_offset = device_id * kv_cache_tokens_per_device
new_kv_offset = device_id * new_kv_tokens_per_device

# Convert to local indices
local_kv_cache_indices = kv_cache_start_indices - kv_cache_offset
local_new_kv_indices = new_kv_cache_indices - new_kv_offset

print(f"Local kv_cache_start_indices: {local_kv_cache_indices}")
print(f"Local new_kv_cache_indices: {local_new_kv_indices}")

# Create slot mapping: [3, num_slices] array with 
# (kv_cache_start, new_kv_start, slice_len) for each slice
slot_mapping_np = np.stack([
    local_kv_cache_indices, 
    local_new_kv_indices, 
    slice_lens
], axis=1)

slot_mapping_np = np.pad(
    slot_mapping_np,
    [[0, padded_num_slices - len(slot_mapping_np)], [0, 0]],
    constant_values=0)
slot_mapping_np = np.transpose(slot_mapping_np)
slices = jnp.array(slot_mapping_np, dtype=jnp.int32)

data_dim = "data"

slices = jax.device_put(slices, jax.sharding.NamedSharding(mesh, P(None, data_dim))) #(3, 256)
kv_cache = jax.device_put(kv_cache, jax.sharding.NamedSharding(mesh, P(data_dim, "model", None))) # (512*64, 16, 128)
new_kv = jax.device_put(new_kv, jax.sharding.NamedSharding(mesh, P(data_dim, "model", None))) # (128, 16, 128)

# Validate inputs
_dynamic_validate_inputs(slices, new_token_num=256, kv_cache_token_num=kv_cache.shape[0], 
                         num_slices=num_slices, page_size=page_size)
# Call the function
with jax.disable_jit():
    updated_kv_cache = kv_cache_update(
        new_kv=new_kv,
        slices=slices,
        kv_cache=kv_cache,
        num_slices=num_slices,
        page_size=page_size,
        num_slices_per_block=64,  
        mesh=mesh, 
        kv_cache_pspec=P(data_dim, "model", None),
        slices_spec=P(None, data_dim),
        num_slices_pspec=P(data_dim),
        dynamic_validate_inputs=False,
        vmem_limit_bytes=40 * 1024 * 1024,
    )

jax.block_until_ready(updated_kv_cache)

print("KV cache update completed successfully!")
print(f"Original kv_cache shape: {kv_cache.shape}")
print(f"Updated kv_cache shape: {updated_kv_cache.shape}")
print(f"Number of slices processed: {num_slices[0]}")
