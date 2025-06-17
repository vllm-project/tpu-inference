import enum


class RouterType(enum.Enum):
    """Enum for router types."""
    TOP_K = 'top_k'


class OPERATION_MODE(enum.Enum):
    PREFILL = 1
    DECODE = 2


# TODO we code the logical mesh axis name as a constant
# we need to make it more flexible in case more names
# could be added for future models
class LOGICAL_MESH_AXIS_NAME(enum.Enum):
    # The constants as the name for mesh axis
    # logical equivalently, we could use 'x', 'y' or ('x', 'y'),
    # but specifying a name will give better readability.
    # The axis should be 'x', 'y', 'z' by physical mesh
    # i.e. [x, y] -> [8, 8] for v6e-64
    BATCH_AXIS_NAME = 'dp'
    SEQUENCE_AXIS_NAME = 'sp'
    ATTN_HEAD_AXIS_NAME = 'ep'
    ATTN_TENSOR_AXIS_NAME = 'tp'
    MLP_TENSOR_AXIS_NAME = ('tp', 'ep')
    MOE_TENSOR_AXIS_NAME = 'tp'
    EXPERT_AXIS_NAME = 'ep'
    VOCAB_AXIS_NAME = ('dp', 'sp', 'tp', 'ep')


"""
Current Used Abbreviation for Tensor Dimensions:
B: Batch size
T: Sequence Length (for Query tensors)
S: Sequence Length (for Key/Value tensors)
D: d_model, the embedding dimension of the model
F: d_ff, the hidden dimension of the feed-forward MLP layers
V: Vocab Size
H: Dimension of each attention head
N: Number of query heads in Attention
Q: Number of query heads (synonymous with N)
K: Number of Key/Value heads in Attention
C: Expert capacity in Mixture-of-Experts models
X: Number of activated experts per token in MoE
G: Number of groups in Grouped-Query Attention
E: Total number of experts in MoE
"""
