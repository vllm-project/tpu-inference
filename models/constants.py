class LOGICAL_MESH_AXIS_NAME(enum.Enum):
  # The constants as the name for mesh axis
  # logical equivalently, we could use 'x', 'y' or ('x', 'y'),
  # but specifying a name will give better readability. 
  # The axis should be 'x', 'y', 'z' by physical mesh
  # i.e. [x, y] -> [8, 8] for v6e-64
  BATCH_AXIS_NAME = 'x'
  SEQUENCE_AXIS_NAME = 'y'
  ATTN_HEAD_AXIS_NAME #str | tuple(str)
  ATTN_TENSOR_AXIS_NAME #str | tuple(str)
  MLP_TENSOR_AXIS_NAME = ('x', 'y')
  MOE_TENSOR_AXIS_NAME = 'x'
  EXPERT_AXIS_NAME = 'y'
  VOCAB_AXIS_NAME = ('x', 'y', 'z')


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