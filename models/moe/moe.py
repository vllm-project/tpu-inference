# MoE metrics to be monitored
MOE_METRICS = (
    'per_layer_load_balancing_loss',
    'rms_logits',
    'per_layer_max_router_logits',
    'over_capacity',
    'expert_assignment_fraction',
    'dispatched_to_0',
    'dispatched_to_1',
    'dispatched_to_2',
    'total_dispatch_weight',
    'router_w_clusterfactor',
    'average_entropy',
    'entropy_average',
)

@dataclass
class MoEConfig:
    d_model: int
    expert_hidden_size: int
    num_experts: int
    router_config: RoutingConfig

    def __init__(self, flags_cfg):
        ...
    def make(self, name, sharding_cfg=None, quantization=None) -> MoE:
        ...
        return MoE(
            d_model=self.d_model
            expert_hidden_size=self.expert_hidden_size
            num_experts=self.num_experts
            router_config=self.router_config.make()
            cfg=self,
            sharding_cfg=sharding_cfg,
            quantization=quantization,
        )

@dataclass
class MoE(nn.Module):
    cfg: Config = None
    d_model: int
    expert_hidden_size: int
    num_experts: int
    router: Router
    sharding_cfg: ShardingConfig = default_sharding()
    quantization: Quantization | None = None

    def setup(self):
        ...
    def __call__(self,):
        ...
    def get_cfg(self) -> MoEConfig: 


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

class TopkRouter:
    k: int
    num_experts: int
    expert_capacity: int

    def __init__(seslf):
        ...
    def router(self):
        ...



