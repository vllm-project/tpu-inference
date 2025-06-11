from tpu_commons.models.jax.layers.rope import apply_rope
from tpu_commons.models.jax.attention_interface import KVCache, attention
from tpu_commons.models.jax.attention_metadata import AttentionMetadata

# A lightweight block serves as a blue-print.
@dataclass(frozen=True)
class AttentionConfig(Config):
    d_model: int
    num_q_heads: int
    num_kv_heads: int   
    head_dim: int
    rope_theta
    rope_scaling

# A heavy weight module which serves as the stateful live blocks in serving
@dataclass
class Attention(nnx.Module):
    """Attention Block"""
    cfg: AttentionConfig
    attention_mask
    query_pos
    mesh: Mesh,
    kernel_init: Initializer # TODO create factories for initializer(?)
    router: Router,
    sharding_cfg: ShardingConfig
    quant: Quantization | None = None

    def setup(self):
        self.create_sharding()
        self._generate_kernel()

    def _generate_kernel(self):

        shape_q_proj = (self.cfg.num_q_heads, self.cfg.d_model, self.cfg.head_dim)
        shape_kv_proj = (self.cfg.num_kv_heads, self.cfg.d_model, self.cfg.head_dim)
        shape_o_proj = (self.cfg.num_q_heads, self.cfg.head_dim, self.cfg.d_model)

        self.kernel_q_proj_QDH = nnx.Param(
            self.kernel_init(shape_q_proj, self.cfg.dtype),
            sharding=self.qdh_sharding)
        self.kernel_k_proj_NDH = nnx.Param(
            self.kernel_init(shape_kv_proj, self.cfg.dtype),
            sharding=self.ndh_sharding)
        self.kernel_v_proj_NDH = nnx.Param(
            self.kernel_init(shape_kv_proj, self.cfg.dtype),
            sharding=self.ndh_sharding)
        self.kernel_o_proj_QHD = nnx.Param(
            self.kernel_init(shape_o_proj, self.cfg.dtype),
            sharding=self.qhd_sharding)

    def _create_named_sharding(self):
        named_sharding = dict()



    def create_sharding(self):

        mode_dependent_attrs = [
            "activation_attention_bsd",
            "activation_q_bsd",
            "query_bsqh",
            "keyvalue_bsnh",
            "activation_attention_out_bsd"
        ]
        for attr_name in mode_dependent_attrs:
            sharding_config = getattr(self.sharding_cfg, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(
                    self.mesh,
                    P(sharding_config.get_axes(OPERATION_MODE.PREFILL))
                ),
                'decode': NamedSharding(
                    self.mesh,
                    P(sharding_config.get_axes(OPERATION_MODE.DECODE))
                )
            }
            setattr(self, attr_name, sharding_dict)

        # static sharding for kernel/weights
        self.qdh_sharding = NamedSharding(
            self.mesh, P(self.sharding_cfg.attn_q_weight_qdh.get_axes())
        )
        self.ndh_sharding = NamedSharding(
            self.mesh, P(self.sharding_cfg.attn_k_weight_ndh.get_axes())
        )
        self.qhd_sharding = NamedSharding(
            self.mesh, P(self.sharding_cfg.attn_o_weight_qhd.get_axes())
        )

    def __call__(
        self, 
        x,
        op_mode,
        kv_cache: KVCache,
        attention_metadata: AttentionMetadata,
    ):  
        md = attention_metadata
        is_prefill = True if op_mode == OPERATION_MODE.PREFILL else False
        x = jnp.asarray(x, self.cfg.dtype)
        x_BSD = nnx.with_sharding_constraint(x, self.activation_attention_bsd[op_mode])
        x_q_BSD = nnx.with_sharding_constraint(x, self.activation_q_bsd[op_mode]) 


        with jax.named_scope("q_proj"):
            q_BSQH = jnp.einsum('BSD,QDH -> BSQH', x_q_BSD, self.kernel_q_proj_QDH.value)
            q_BSQH = apply_rope(
                q_BSQH, md.input_positions, self.cfg.head_dim, self.cfg.rope_theta,
                self.cfg.rope_scaling)
            q_BSQH = nnx.with_sharding_constraint(q_BSQH, self.query_bsqh[op_mode])

        with jax.named_scope("k_proj"):
            k_BSNH = np.einsum('BSD,NDH -> BSNH', x, self.kernel_k_proj_NDH.value)
            k_BSNH = apply_rope(
                k_BSNH, md.input_positions, self.cfg.head_dim, self.cfg.rope_theta,
                self.cfg.rope_scaling)
            k_BSNH = nnx.with_sharding_constraint(k_BSNH, self.keyvalue_bsnh[op_mode]) 
    
        with jax.named_scope("v_proj"):
            v_BSNH = np.einsum('BSD,NDH -> BSNH', x, self.kernel_v_proj_NDH.value)
            v_BSNH = nnx.with_sharding_constraint(v_BSNH, self.keyvalue_bsnh[op_mode]) 

        with jax.named_scope("attn_op"):
            new_kv_cache, outputs_BSQH = attention(
                is_prefill,
                kv_cache,
                q_BSQH,
                k_BSNH,
                v_BSNH,
                attention_metadata,
                self.mesh,
                self.cfg.num_q_heads,
                self.cfg.num_kv_heads,
            )

        with jax.named_scope("o_proj"):
            o_BSNH = np.einsum('BSQH,QHD -> BSD', outputs_BSQH, self.kernel_o_proj_QHD.value)
            o_BSNH = nnx.with_sharding_constraint(o_BSNH, self.activation_attention_out_bsd[op_mode])
        return new_kv_cache, o_BSNH


    def get_cfg(self) -> AttentionConfig:
        return self.cfg
    # other methods