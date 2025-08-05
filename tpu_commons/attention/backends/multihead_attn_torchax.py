from typing import Optional

import torch
import torch.nn as nn
from vllm.attention.layer import MultiHeadAttention as VllmMultiHeadAttention
from vllm.logger import init_logger

from tpu_commons.models.torchax.torchax_wrapper import flash_attention

logger = init_logger(__name__)


class MultiHeadAttention(nn.Module):
    """
    This is a replacement module of MultiHeadAttention in vllm for torchax runner path.
    It overrides the original dispathing logic with a torchax compatible flash attention pallas kernel call
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({self.num_heads}) is not " \
            f"divisible by num_kv_heads ({self.num_kv_heads})"
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    @staticmethod
    def from_vllm_cls(vllm_attn: VllmMultiHeadAttention):
        return MultiHeadAttention(vllm_attn.num_heads,
                                  vllm_attn.head_size,
                                  vllm_attn.scale,
                                  num_kv_heads=vllm_attn.num_kv_heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: batch_size x seq_len x hidden_size"""

        bsz, q_len, _ = query.size()
        kv_len = key.size(1)

        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=2)
            value = torch.repeat_interleave(value, num_repeat, dim=2)

        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        out = flash_attention(query,
                              key,
                              value,
                              q_len,
                              kv_len,
                              sm_scale=self.scale)
        return out.reshape(bsz, q_len, -1)
