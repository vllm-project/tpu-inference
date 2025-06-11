class OPERATION_MODE(enum.Enum):
  PREFILL = 1
  DECODE = 2

# TODO we code the logical mesh axis name as a constant
# we need to make it more flexible in case more names 
# could be added for future models
class LOGICAL_MESH_AXIS_NAME:
  # The constants as the name for mesh axis
  # logical equivalently, we could use 'x', 'y' or ('x', 'y'),
  # but specifying a name will give better readability. 
  # The axis should be 'x', 'y', 'z' by physical mesh
  # i.e. [x, y] -> [8, 8] for v6e-64
  BATCH_AXIS_NAME = 'dp'
  SEQUENCE_AXIS_NAME = 'sp'
  ATTN_HEAD_AXIS_NAME = 'ep'
  ATTN_TENSOR_AXIS_NAME  = 'tp'
  MLP_TENSOR_AXIS_NAME = ('tp', 'ep')
  MOE_TENSOR_AXIS_NAME = 'tp'
  EXPERT_AXIS_NAME = 'ep'
  VOCAB_AXIS_NAME = ('dp', 'sp', 'tp', 'ep')


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