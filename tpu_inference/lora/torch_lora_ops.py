# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import jax
import jax.numpy as jnp
import torch
from torchax.interop import call_jax


@jax.jit
def bgmv_jax(
        inputs,  # [num_tokens, hidden_size]
        loras,  # [num_loras, lora_rank, hidden_size]
        idxs,  # [num_tokens]
):
    return jnp.einsum(
        "td,tX,Xld->tl",
        inputs,
        jax.nn.one_hot(idxs, loras.shape[0], dtype=inputs.dtype),
        loras,
    )


def bgmv_torch(
        inputs,  # [num_tokens, hidden_size]
        loras,  # [num_loras, 1, lora_rank, hidden_size]
        idxs,  # [num_tokens]
):  # [num_tokens, lora_rank]
    # TODO(xiowei): use the below one_hot impl (added in https://github.com/pytorch/xla/pull/9523) after we upgrade torchax version.
    # if len(loras.shape) == 4:
    #     loras = loras.squeeze(axis=1)
    # return torch.einsum(
    #     "td,tX,Xld->tl",
    #     inputs,
    #     torch.nn.functional.one_hot(idxs.long(), loras.shape[0]),
    #     loras,
    # )  # [num_tokens, lora_rank]

    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)
    return call_jax(bgmv_jax, inputs, loras, idxs)


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
    """
    Args:
        inputs (torch.Tensor): Input tensor of shape [num_tokens, hidden_size].
        lora_b_weights (torch.Tensor): LoRA weights of shape
            [max_loras, 1, max_lora_rank, hidden_size].
        output_tensor (torch.Tensor): (Unused) output tensor (placeholder).
        lora_indices_tensor (torch.Tensor): Tensor of shape [num_tokens]
            indicating which LoRA matrix to use for each token.
        scaling (float, optional): Scalar multiplier applied to the output.
    """
    return scaling * bgmv_torch(inputs, lora_b_weights, lora_indices_tensor)


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    """
    Args:
        inputs (torch.Tensor): Input tensor of shape [num_tokens, lora_rank].

        lora_b_weights (torch.Tensor): LoRA weights of shape
            [num_loras, 1, out_features, lora_rank].

        output_tensor (torch.Tensor): output tensor of shape
            [num_tokens, out_features * num_slices].

        lora_indices_tensor (torch.Tensor): Tensor of shape [num_tokens]
            indicating which LoRA matrix to use for each token.
        add_inputs (bool): Whether or not to add the input tensor to the output
            tensor.
    """
    outputs = bgmv_torch(inputs, lora_b_weights,
                         lora_indices_tensor)  # [num_tokens, out_features]

    # Create a padded tensor manually to avoid issues with F.pad on sharded tensors.
    # This is a more robust way to handle padding in a distributed environment.
    outputs_padded = torch.zeros_like(output_tensor)
    outputs_padded[:, slice_offset:slice_offset + slice_size] = outputs

    if add_inputs:
        return output_tensor + outputs_padded
    else:
        return outputs_padded
