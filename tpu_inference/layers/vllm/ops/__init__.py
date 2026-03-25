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
import functools

import torch
import torch.nn
from torchax.ops.ops_registry import register_torch_function_op

from tpu_inference.layers.vllm.ops import gdn_attention as gdn_attention
from tpu_inference.layers.vllm.ops import \
    scaled_dot_product_attention as scaled_dot_product_attention
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


def register_ops():
    # Caution: there is no public api for restore the ops.
    # It need to patched again if the ops are jitted and mesh is change.
    # The overwritten ops should not be called after the end of model wrapper.

    # Import the registered ops at first and then we can overwrite them.
    import torchax.ops.jtorch  # noqa: F401

    vllm_model_wrapper_context = get_vllm_model_wrapper_context()
    mesh = vllm_model_wrapper_context.mesh

    # Patch sdpa from torch ops to flash attention to prevent OOM
    register_torch_function_op(
        torch.nn.functional.scaled_dot_product_attention,
        functools.partial(
            scaled_dot_product_attention.scaled_dot_product_attention,
            mesh=mesh),
        is_jax_function=True,
        needs_env=False,
    )
