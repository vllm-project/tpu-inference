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
"""Generic TPU patches to ensure compatibility."""

import torch
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def apply_generic_tpu_patches():
    """Apply global patches to vLLM operations for TPU compatibility."""
    _patch_rms_norm()
    _patch_default_unquantized_gemm()
    logger.info("Applied generic TPU patches.")


def _patch_rms_norm():
    """Patch vLLM's RMSNorm to avoid mixed-math errors in eager mode."""
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm

        if not hasattr(RMSNorm, "_original_forward_native"):
            RMSNorm._original_forward_native = RMSNorm.forward_native

            def _patched_rms_norm_forward_native(self, x, residual=None):
                weight_param = None
                if getattr(self, "has_weight", True) and hasattr(self, "weight"):
                    w = self.weight
                    if isinstance(w, torch.nn.Parameter):
                        w = w.data
                    
                    if type(w) is torch.Tensor:
                        w = w.to(device="jax")
                        
                    weight_param = w

                return self.forward_static(
                    x,
                    self.variance_epsilon,
                    self.hidden_size,
                    x.dtype,
                    weight_param,
                    residual,
                    self.variance_size_override,
                )

            RMSNorm.forward_native = _patched_rms_norm_forward_native
            logger.info("Patched vLLM's RMSNorm.forward_native.")
    except Exception as e:
        logger.warning(f"Failed to patch vLLM's RMSNorm: {e}")


def _patch_default_unquantized_gemm():
    """Patch vLLM's default_unquantized_gemm to unwrap parameters."""
    try:
        import vllm.model_executor.layers.utils as vllm_layer_utils

        if not hasattr(vllm_layer_utils, "_original_default_unquantized_gemm"):
            vllm_layer_utils._original_default_unquantized_gemm = (
                vllm_layer_utils.default_unquantized_gemm
            )

            def _patched_default_unquantized_gemm(layer, x, weight, bias=None):
                if isinstance(weight, torch.nn.Parameter):
                    weight = weight.data
                
                if type(weight) is torch.Tensor:
                    weight = weight.to(device="jax")
                
                if bias is not None:
                    if isinstance(bias, torch.nn.Parameter):
                        bias = bias.data
                    if type(bias) is torch.Tensor:
                        bias = bias.to(device="jax")

                return torch.nn.functional.linear(x, weight, bias)

            vllm_layer_utils.default_unquantized_gemm = _patched_default_unquantized_gemm
            logger.info("Patched vLLM's default_unquantized_gemm.")
    except Exception as e:
        logger.warning(f"Failed to patch vLLM's default_unquantized_gemm: {e}")
