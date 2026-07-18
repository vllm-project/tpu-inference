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

from tpu_inference.layers.vllm.custom_ops import embedding as embedding
from tpu_inference.layers.vllm.custom_ops import fused_moe as fused_moe
from tpu_inference.layers.vllm.custom_ops import \
    gdn_attention_op as gdn_attention_op
from tpu_inference.layers.vllm.custom_ops import linear as linear
from tpu_inference.layers.vllm.custom_ops import mhc as mhc
from tpu_inference.layers.vllm.custom_ops import mla_attention as mla_attention
from tpu_inference.layers.vllm.custom_ops import rope as rope
from tpu_inference.layers.vllm.custom_ops import \
    sparse_attn_indexer as sparse_attn_indexer

# Register custom op to vLLM so that vLLM model implementation will instantiante
# classes with definitions in tpu-inference.

try:
    import sys

    import vllm.model_executor.layers.quantization.utils.fp8_utils as _vllm_fp8_utils

    def _tpu_per_token_group_quant_fp8(
        x: torch.Tensor,
        group_size: int,
        eps: float = 1e-10,
        dtype: torch.dtype = torch.float8_e4m3fn,
        column_major_scales: bool = False,
        scale_ub: torch.Tensor | None = None,
        use_ue8m0: bool = False,
    ):
        fp8_max = torch.finfo(dtype).max
        x_reshaped = x.view(-1, group_size)
        amax = torch.amax(torch.abs(x_reshaped), dim=-1,
                          keepdim=True).clamp(min=eps)
        x_s = amax / fp8_max
        x_q = (x_reshaped / x_s).to(dtype).view_as(x)
        if not column_major_scales:
            x_s = x_s.view(x.shape[:-1] + (x.shape[-1] // group_size, ))
        else:
            x_s = x_s.view(x.shape[0], x.shape[1] // group_size)
        return x_q, x_s

    _vllm_fp8_utils.per_token_group_quant_fp8 = _tpu_per_token_group_quant_fp8

    # Patch already imported modules in sys.modules
    for name, module in list(sys.modules.items()):
        if module is not None and name.startswith("vllm."):
            if hasattr(module, 'per_token_group_quant_fp8'):
                setattr(module, 'per_token_group_quant_fp8',
                        _tpu_per_token_group_quant_fp8)
except (ImportError, AttributeError):
    pass
