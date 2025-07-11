import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.attention.attention import MLA, MLAConfig
from tpu_commons.models.jax.common.base import ParamFactory
from tpu_commons.models.jax.common.kv_cache import (
    KVCacheConfig,
)
from tpu_commons.models.jax.common.sharding import ShardingConfig


# Create MLA config
mla_config = MLAConfig(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=32,
    rope_theta=10000,
    dtype=jnp.bfloat16,
    q_lora_rank=512,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    rms_norm_eps=1e-5,
    rope_scaling={
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
)

cpu_mesh = Mesh(jax.devices("cpu"), axis_names=("model",))
tpu_mesh = Mesh(jax.devices("tpu"), axis_names=("model",))
param_factory = ParamFactory(
    kernel_initializer=nnx.initializers.xavier_normal(),
    scale_initializer=nnx.initializers.ones,
)
sharding_cfg = ShardingConfig()

mla_layer = MLA(
    cfg=mla_config,
    mesh=tpu_mesh,
    param_factory=param_factory,
    sharding_cfg=sharding_cfg,
)

mla_layer.generate_kernel(nnx.Rngs(42))

x = jnp.ones((128, 4096))

kv_cache_config = KVCacheConfig(
    batch_size=1,
    cache_len=128,
    num_kv_heads=32,
    head_dim=256,
    dtype=jnp.bfloat16,
)

# Create proper JAX arrays for the KV cache
# The KV cache should be shaped as (num_blocks, block_size, num_kv_heads*2, head_dim)
# where num_blocks = cache_len // block_size
block_size = 32  # Standard block size for TPU
num_blocks = kv_cache_config.cache_len // block_size
cache_shape = (num_blocks, block_size,
               kv_cache_config.num_kv_heads*2, kv_cache_config.head_dim)
key_cache_array = jnp.zeros(cache_shape, dtype=kv_cache_config.dtype)

print(f"KV cache shape: {key_cache_array.shape}")
print(f"Expected shape: ({num_blocks}, {block_size}, {kv_cache_config.num_kv_heads*2}, {kv_cache_config.head_dim})")


attention_metadata = AttentionMetadata(
    input_positions=jnp.arange(128, dtype=jnp.int32),
    slot_mapping=jnp.zeros((3, 1), dtype=jnp.int32),
    block_tables=jnp.zeros((1, 4), dtype=jnp.int32),
    seq_lens=jnp.ones((1,), dtype=jnp.int32) * 128,
    num_slices = jnp.ones((1, ), dtype=jnp.int32),
    num_seqs = jnp.ones((1, ), dtype=jnp.int32),
    query_start_loc=jnp.array([0, 128], dtype=jnp.int32)  # This is cu_q_lens

)

output = mla_layer(
    x, is_prefill=True, kv_cache=key_cache_array, attention_metadata=attention_metadata
)

print(output.shape)
