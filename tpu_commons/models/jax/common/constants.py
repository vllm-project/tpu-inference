import enum


class RouterType(enum.Enum):
    """Enum for router types."""
    TOP_K = 'top_k'


class OPERATION_MODE(enum.Enum):
    PREFILL = 1
    DECODE = 2


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
