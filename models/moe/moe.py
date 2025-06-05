
@dataclass
class MoEConfig(Config):
    d_model: int
    expert_hidden_size: int
    num_experts: int
    sequence_len: int
    router_config: RoutingConfig
    ...

    @classmethod
    def from_cfg(cls, flags_cfg: dict):
        required_params = {f.name for f in fields(cls)}
        provided_params = set(flags_cfg.keys())
        missing_params  = required_params - provided_params

        if missing_params:
            ...

        moe_flags = {k: flags_cfg[k] for k in required_params}
        return cls(**moe_flags)
    
    def make(self, name, runtime_param: Optional[layer.RuntimeParams] = None) -> MoE:
        ...
        return MoE(
            cfg=self,
            kernel=kernel,
            router=self.router_config.make()
            sharding_cfg=runtime_param.sharding_cfg,
            quantization=runtime_param.quantization,
        )

@dataclass
class MoE(nn.Module):
  """Moe Routed MLP Layer"""
    cfg: MoEConfig
    kernel: 
    router: Router
    sharding_cfg: ShardingConfig = default_sharding()
    quantization: Quantization | None = None

    def setup(self):
        ...
    def __call__(self,):
        #TODO implement the logics here
        ...
    def get_cfg(self) -> MoEConfig: 
        ...
    def sharding(op_mode):
        ...

@dataclass
class RoutingConfig:
    num_experts: int
    expert_capacity: int
    k: int
    router_type: RouterType
    routed_bias: bool = False
    routed_scaling_factor: float
    ...

    def __init__(self):
        ...
    def make(self) -> Router:
        ...
        match router_type:
            case RouterType.TopK:
                return TopkRouter(
                    cfg=self,
                    k=k,
                    num_experts=self.num_experts,
                    expert_capacity=self.expert_capacity
                    ...
                )
@dataclass
class TopkRouter:
    k: int
    num_experts: int
    expert_capacity: int
    ...

    def __init__(self):
        ...
    def router(self):
        ...



