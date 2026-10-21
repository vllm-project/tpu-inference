# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import \
    DeepseekScalingRotaryEmbedding


def rotate_gptj_tpu(x: torch.Tensor):
    original_shape = x.shape

    # Avoided strided slicing and stacking
    x_reshaped = x.reshape(*original_shape[:-1], -1, 2)

    mask = torch.tensor([-1.0, 1.0], device=x.device)
    x_flipped = x_reshaped.flip(-1) * mask

    return x_flipped.reshape(original_shape)


@DeepseekScalingRotaryEmbedding.register_oot
class VllmDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    """
    TPU-friendly modifications for efficient RoPE usage.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Intercepts PyTorch-native implementation with TPU friendlier RoPE rotation."""
        if self.is_neox_style:
            return super().forward_native(positions, query, key, offsets)

        assert key is not None
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        cos_sin = cos_sin_cache[torch.
                                add(positions, offsets
                                    ) if offsets is not None else positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        query_rot = query_rot * cos + rotate_gptj_tpu(query_rot) * sin
        key_rot = key_rot * cos + rotate_gptj_tpu(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key
