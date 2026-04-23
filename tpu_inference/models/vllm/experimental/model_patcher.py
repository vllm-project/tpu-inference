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
"""Patch vllm models to be TPU friendly or JITtable.

Currently, it provides a way to JIT compile specific submodules of a
multimodal model using `torchax.interop.JittableModule`. This is useful
for offloading heavy vision or audio encoders to an Jitted graph while keeping
the rest of the model in eager mode.

"""

import functools
import importlib
from typing import TYPE_CHECKING, Sequence

import jax
import torchax
from torchax.interop import JittableModule, torch_view

from tpu_inference.logger import init_logger

if TYPE_CHECKING:
    from tpu_inference.models.vllm.vllm_model_wrapper import _VllmRunner

logger = init_logger(__name__)


def patch_mm_model(
    model: "_VllmRunner",
    params_and_buffers: dict[str, torchax.torch.Tensor],
    *,
    jitted_mm_module_keys: Sequence[str],
    register_mm_module_custom_pytree_classes: Sequence[str],
) -> tuple["_VllmRunner", dict[str, torchax.torch.Tensor]]:
    """Jit some modules in the multimodal.

    We add a wrapper to change the submodule call,
    which the params_and_buffers path would change.

    Caution: the submodule params_and_buffers would be put into
    the wrapper directly. params_and_buffers should be sharded to tpu
    and would not be used in the function args.

    Args:
        model: The vLLM runner model. Should not be used after calling.
        params_and_buffers: The parameters and buffers of the model. Should not be used after calling.
        jitted_mm_module_keys: A list of module paths (e.g., "model.vision_tower.encoder")
          to be wrapped with `torchax.interop.JittableModule` for JAX JIT compilation.
        register_mm_module_custom_pytree_classes: A list of fully qualified class names
          (e.g., "transformers.modeling_outputs.BaseModelOutputWithPast") to register as
          JAX pytrees, allowing them to pass through JIT boundaries.
          The class need to has functions `keys` and `values` and can be construct with them
          like a dict.

    Returns:
        The patched model and the updated parameters and buffers.
    """
    # Monkey patch vLLM's _merge_multimodal_embeddings to support JAX/torchax
    # broadcasting limitations with boolean masks.
    try:
        import vllm.model_executor.models.utils as vllm_utils

        # Save the original to avoid recursion if patched multiple times
        if not hasattr(vllm_utils, "_original_merge_multimodal_embeddings"):
            vllm_utils._original_merge_multimodal_embeddings = vllm_utils._merge_multimodal_embeddings

            def _jax_compatible_merge_multimodal_embeddings(
                    inputs_embeds, multimodal_embeddings, is_multimodal):
                if len(multimodal_embeddings) == 0:
                    return inputs_embeds

                import torch

                mm_embeds_flat = vllm_utils._flatten_embeddings(
                    multimodal_embeddings)
                input_dtype = inputs_embeds.dtype
                mm_embeds_flat = mm_embeds_flat.to(dtype=input_dtype)

                # PyTorch boolean indexing (inputs[mask] = values) requires dynamic 
                # host-synchronization. We use static math ops (cumsum, where) 
                # which trace perfectly via torchax into JAX without deadlocking.
                
                # Create a dummy row to handle indices for non-multimodal tokens.
                dummy_row = torch.zeros_like(mm_embeds_flat[0:1])
                # Prepend the dummy row.
                flattened_padded = torch.cat([dummy_row, mm_embeds_flat], dim=0)

                # For non-multimodal tokens, cumsum points to 0. 
                # For multimodal tokens, it points to their 1-based index in the padded array.
                gather_indices = is_multimodal.to(torch.int64).cumsum(dim=0)
                update_values = flattened_padded[gather_indices]

                condition = is_multimodal.unsqueeze(-1)
                new_embeds = torch.where(condition, update_values, inputs_embeds)

                return new_embeds

            vllm_utils._merge_multimodal_embeddings = _jax_compatible_merge_multimodal_embeddings
            logger.info(
                "Patched vLLM's _merge_multimodal_embeddings with static JAX logic."
            )
    except Exception as e:
        logger.warning(
            f"Failed to patch vLLM's _merge_multimodal_embeddings: {e}")

    # Patch vLLM's RMSNorm to avoid mixed-math errors with self.weight.data 
    # when using torchax + functional_call.
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm

        # Store original if not already patched
        if not hasattr(RMSNorm, "_original_forward_native"):
            RMSNorm._original_forward_native = RMSNorm.forward_native

            def _patched_rms_norm_forward_native(self, x, residual=None):
                import torch
                weight_param = None
                if getattr(self, "has_weight", True) and hasattr(self, "weight"):
                    w = self.weight
                    # In eager mode, self.weight is an nn.Parameter which TorchAX cannot
                    # directly do math with. We must unwrap it to get the TorchAX tensor.
                    if isinstance(w, torch.nn.Parameter):
                        w = w.data
                    
                    # If it somehow remained a raw CPU tensor (e.g. dummy torch.ones),
                    # push it to the JAX device to avoid mixed math errors.
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
            logger.info("Patched vLLM's RMSNorm.forward_native to use forward_static.")
    except Exception as e:
        logger.warning(f"Failed to patch vLLM's RMSNorm: {e}")

    # Patch vLLM's default_unquantized_gemm to unwrap custom ModelWeightParameters
    # which cause mixed-math errors in torchax eager mode.
    try:
        import vllm.model_executor.layers.utils as vllm_layer_utils

        if not hasattr(vllm_layer_utils, "_original_default_unquantized_gemm"):
            vllm_layer_utils._original_default_unquantized_gemm = (
                vllm_layer_utils.default_unquantized_gemm
            )

            def _patched_default_unquantized_gemm(layer, x, weight, bias=None):
                import torch
                # If weight is an nn.Parameter or custom vLLM parameter, 
                # we must unwrap it to get the raw tensor for torchax.
                if isinstance(weight, torch.nn.Parameter):
                    weight = weight.data
                
                # If it remained a raw CPU tensor, push to JAX.
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

    if not jitted_mm_module_keys:
        return model, params_and_buffers

    logger.warning_once(
        "JIT compilation for multi-modal (MM) modules is an experimental feature."
    )

    # Flatten custom pytree to jax for jit
    # eg. transformers.modeling_outputs.BaseModelOutputWithPast
    def _flatten_model_output(obj):
        return obj.values(), obj.keys()

    def _unflatten_model_output(aux, children, obj_type):
        keys = aux
        return obj_type(**dict(zip(keys, children)))

    for class_path in register_mm_module_custom_pytree_classes:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        class_to_register = getattr(module, class_name)
        jax.tree_util.register_pytree_node(
            class_to_register,
            _flatten_model_output,
            functools.partial(_unflatten_model_output,
                              obj_type=class_to_register),
        )
        logger.info("Register pytree node for %s", class_path)

    # Substitute the module with JittableModule, which jit the forward function
    for module_key in jitted_mm_module_keys:
        # module_key expect to be start with model
        # eg. "model.vision_tower.encoder" -> "vllm_model.vision_tower.encoder"

        cur_module = model.vllm_model
        module_names = module_key.split('.')
        if len(module_names) <= 1:
            raise ValueError(
                f"jit submodule only, but not the whole model. get {module_names=}"
            )

        # Start from 1 because 'model' is our root
        for name in module_names[1:-1]:
            cur_module = getattr(cur_module, name)

        target_module_name = module_names[-1]
        jitted_module = JittableModule(getattr(cur_module, target_module_name))
        setattr(cur_module, target_module_name, jitted_module)

        # params_and_buffers is a dict. for each key with prefix of the module,
        # add `._model` to the prefix since JittableModule wrapper put origin into _model.
        new_params_and_buffers = {}
        prefix = f"vllm_model.{'.'.join(module_names[1:])}"
        new_prefix = f"{prefix}._model"
        for k, v in params_and_buffers.items():
            if not k.startswith(prefix):
                new_params_and_buffers[k] = v
                continue

            new_key = k.replace(prefix, new_prefix, 1)
            new_params_and_buffers[new_key] = v
            # Also update the params and buffers in the JittableModule
            # They are not using params_and_buffs in the function args
            inner_k = k.replace(prefix + ".", "")
            if inner_k in jitted_module.params:
                jitted_module.params[inner_k] = torch_view(v)
            elif inner_k in jitted_module.buffers:
                jitted_module.buffers[inner_k] = torch_view(v)
            else:
                raise ValueError(f"Unexpect key {k} in params_and_buffers")

        params_and_buffers = new_params_and_buffers
        logger.info("Jit module %s", module_key)

    return model, params_and_buffers
