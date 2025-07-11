import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from tpu_commons.models.jax.common.attention.attention import MLA, MLAConfig
from tpu_commons.models.jax.common.base import ParamFactory
from tpu_commons.models.jax.common.sharding import ShardingConfig

# MLAConfig = make_dataclass(
#     "MLAConfig",
#     [
#         (HuggingFaceArgNames.Q_LORA_RANK.value, int),
#         (HuggingFaceArgNames.KV_LORA_RANK.value, int),
#         (HuggingFaceArgNames.QK_NOPE_HEAD_DIM.value, int),
#         (HuggingFaceArgNames.QK_ROPE_HEAD_DIM.value, int),
#         (HuggingFaceArgNames.V_HEAD_DIM.value, int),
#         (HuggingFaceArgNames.RMS_NORM_EPS.value, float),
#     ],
#     bases=(AttentionConfig,),
# )

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

cpu_mesh = Mesh(jax.devices("cpu"), axis_names=("data", ))
tpu_mesh = Mesh(jax.devices("tpu"), axis_names=("data", ))
param_factory = ParamFactory(
    kernel_initializer=nnx.initializers.xavier_normal(),
    scale_initializer=nnx.initializers.ones,
)
sharding_cfg = ShardingConfig()

mla_layer = MLA(
    cfg=mla_config,
    mesh=cpu_mesh,
    param_factory=param_factory,
    sharding_cfg=sharding_cfg,
)

mla_layer.generate_kernel()

x = jnp.ones((1, 1024, 4096))

output = mla_layer(x)

print(output.shape)
