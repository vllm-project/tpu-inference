import jax.numpy as jnp

from tpu_commons.models.jax.common.attention.attention import MLA, MLAConfig

# Create MLA config
mla_config = MLAConfig(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=32,
    head_dim=128,
    rope_theta=10000,
    rope_scaling={},
    dtype=jnp.bfloat16,
    q_lora_rank=0,  # Set to >0 to enable Q LoRA
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128)

# Initialize MLA layer
mla_layer = MLA(cfg=mla_config,
                mesh=your_mesh,
                param_factory=your_param_factory,
                sharding_cfg=your_sharding_config)
