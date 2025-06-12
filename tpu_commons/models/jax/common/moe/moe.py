
@dataclass
class MoEConfig(Config):
    d_model: int
    expert_hidden_size: int
    num_experts: int
    dtype: Any = jnp.float32
    sequence_len: int
    act: str
    apply_expert_weight_before_computation: bool = False,
    router_config: RoutingConfig

@dataclass
class MoE(nnx.Module):
  """Moe Routed MLP Layer"""
    cfg: MoEConfig,
    mesh: Mesh,
    kernel_init: Initializer # TODO create factories for initializer(?)
    router: Router,
    sharding_cfg: ShardingConfig
    quant: Quantization | None = None

    def setup(self):
        self.create_sharding()
        self._generate_kernel()

    def __call__(self, x: Float[Array, 'B S D'], op_mode):
        x = jnp.asarray(x, jnp.float32)
        x = nnx.with_sharding_constraint(x, self.activation_sharding[op_mode])       
        weights_BSK, indices_BSK  = self.router(x)
        if self.cfg.apply_expert_weight_before_computation:
            with jax.named_scope("pre_computing_weight"):
                # need optimization for the out-product
                weighted_x_BSKD = jnp.einsum('BSD,BSK->BSKD', x, weights_BSK)
            return self._moe_fwd(weighted_x_BSKD, jnp.ones_like(weights_BSK), op_mode)
        else:
            return self._moe_fwd(x, weights_BSK, op_mode)

    def _generate_kernel(self):

        shape_gating = (self.cfg.num_experts, self.cfg.d_model, self.cfg.hidden_size)
        shape_up = (self.cfg.num_experts, self.cfg.d_model, self.cfg.hidden_size)
        shape_down = (self.cfg.num_experts, self.cfg.hidden_size, self.cfg.d_model)

        self.kernel_gating_EDF = nnx.Param(
            self.kernel_init(shape_gating, self.cfg.dtype),
            sharding=self.edf_sharding)
        self.kernel_up_proj_EDF = nnx.Param(
            self.kernel_init(shape_up, self.cfg.dtype),
            sharding=self.edf_sharding)
        self.kernel_down_proj_EFD = nnx.Param(
            self.kernel_init(shape_down, self.cfg.dtype),
            sharding=self.efd_sharding)

    def _moe_fwd(self, x: Float[Array, 'B S D'], weights, op_mode):
        """
        basic moe forward without dropping, megablx etc
        """
        x = jnp.asarray(x, self.cfg.dtype)
        x = nnx.with_sharding_constraint(x, self.activation_sharding[op_mode]) 

        with jax.named_scope("gating"):
            gating_BSEF = jnp.einsum('BSD,EDF -> BSEF', x, self.kernel_gating_EDF.value)
            activated_gating_BSEF = modeling_flax_utils.ACT2FN[self.cfg.act](gating_BSEF)
        with jax.named_scope("up_projection"):
            up_proj_BSEF = jnp.einsum('BSD,EDF -> BSEF', x, self.kernel_up_proj_EDF.value)
        fuse_BSEF = activated_gating_BSEF * up_proj_BSEF
        with jax.named_scope("down_projection"):
            down_proj_BSED = jnp.einsum('BSEF,EFD -> BSED', fuse_BSEF, self.kernel_down_proj_EDF.value)
        with jax.named_scope("sum"):
            output_BSD = jnp.einsum('BSED,BSE -> BSD', down_proj_BSED, weights)
        return output_BSD.astype(self.cfg.dtype)

    def get_cfg(self) -> MoEConfig: 
        return self.cfg

    def create_sharding():
        self.activation_sharding = dict()
        self.activation_sharding['prefill'] =  NamedSharding(
            self.mesh.get_mesh(OPERATION_MODE.PREFILL), P(self.sharding_cfg.activation_bsd.get_axes(OPERATION_MODE.PREFILL)))
        self.activation_sharding['decode'] =  NamedSharding(
            self.mesh.get_mesh(OPERATION_MODE.DECODE), P(self.sharding_cfg.activation_bsd.get_axes(OPERATION_MODE.DECODE)))
        self.edf_sharding =  NamedSharding(
            self.mesh, P(self.sharding_cfg.moe_weights_edf.get_axes(op_mode)))
        self.efd_sharding =  NamedSharding(
            self.mesh, P(self.sharding_cfg.moe_weights_efd.get_axes(op_mode)))
            
        return 

@dataclass
class RoutingConfig(Config):
    d_model: int
    hidden_size: int
    num_experts: int
    expert_capacity: int
    num_experts_per_tok: int
    router_type: RouterType
    routed_bias: bool = False
    routed_scaling_factor: float
    act: str
    dtype: Any = jnp.float32

@dataclass
class Router(nnx.Module):
    cfg: RoutingConfig
    mesh: Mesh
    kernel_init
    sharding_cfg: ShardingConfig
    quant: Quantization | None = None

    def setup(self):
        self.create_sharding()
        self._generate_kernel()

    def __call__(self, x: x: Float[Array, 'B S D'], op_mode):
        x = jnp.asarray(x, self.cfg.dtype)
        x = nnx.with_sharding_constraint(x, self.activation_sharding[op_mode])
        router_logits_BSE = jnp.einsum('BSD,DE -> BSE', x, self.kernel_DE.value)
        activated_gating_BSF = modeling_flax_utils.ACT2FN[self.cfg.act](router_logits_BSE)
        weights_BSK, selected_experts_BSK = jax.lax.top_k(activated_gating_BSF, self.cfg.num_experts_per_tok)
        normalized_weights_BSK = nn.softmax(weights_BSK.astype(self.cfg.dtype), axis=-1)
        return normalized_weights_BSK

    def _generate_kernel(self):
        shape = (self.cfg.d_model, self.cfg.num_experts)
        self.kernel_DE = nnx.Param(
            self.kernel_init(shape, self.cfg.dtype),
            sharding=self.ed_sharding)

    def create_sharding():
        self.activation_sharding = dict()
        self.activation_sharding['prefill'] =  NamedSharding(
            self.mesh.get_mesh(OPERATION_MODE.PREFILL), P(self.sharding_cfg.activation_bsd.get_axes(OPERATION_MODE.PREFILL)))
        self.activation_sharding['decode'] =  NamedSharding(
            self.mesh.get_mesh(OPERATION_MODE.DECODE), P(self.sharding_cfg.activation_bsd.get_axes(OPERATION_MODE.DECODE)))
        self.ed_sharding =  NamedSharding(
            self.mesh, P(self.sharding_cfg.moe_router_de.get_axes()))

        return 



