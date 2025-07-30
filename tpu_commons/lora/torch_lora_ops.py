# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from tpu_commons.distributed.tpu_distributed_utils import \
    create_torchax_tensor_with_partition_spec

# import torch_xla.core.xla_builder as xb

# from torch_xla.experimental.custom_kernel import XLA_LIB, jax_import_guard

# @jax.jit
# def bgmv_jax(inputs, loras, idxs):
#     # inputs: [num_tokens, hidden_size]
#     # loras_b_weights: [num_loras, lora_rank, hidden_size]
#     # idxs: [num_tokens]
#     return jnp.einsum(
#         "td,tX,Xld->tl",
#         inputs,
#         jax.nn.one_hot(idxs, loras.shape[0], dtype=inputs.dtype),
#         loras,
#     )


def bgmv_torch(inputs, loras, idxs):
    # inputs: [num_tokens, hidden_size]
    # loras_b_weights: [num_loras, lora_rank, hidden_size]
    # idxs: [num_tokens]
    # print(f'{idxs.device=}, isinstance(idxs, torchax.tensor.Tensor)')  # should be 'jax'
    # idxs = idxs.to(torch.long)
    # return torch.einsum(
    #     "td,tX,Xld->tl",
    #     inputs,
    #     torch.nn.functional.one_hot(idxs, loras.shape[0]),
    #     loras,
    # )  # [num_tokens, lora_rank]
    # xw32q: when is len(loras.shape)==4 and when it is not?
    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)
    selected_loras = loras[idxs]
    selected_loras = create_torchax_tensor_with_partition_spec(selected_loras)
    print(f'xw32 bgmv_torch: {type(inputs)=}, {type(selected_loras)=}')
    return torch.einsum('td,tld->tl', inputs, selected_loras)


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    """
    Args:
        inputs (torch.Tensor): Input tensor of shape [num_tokens, hidden_size].

        lora_b_weights (torch.Tensor): LoRA weights of shape
            [num_loras, lora_rank, hidden_size].

        output_tensor (torch.Tensor): output tensor of shape
            [num_tokens, hidden_size * num_slices].

        lora_indices_tensor (torch.Tensor): Tensor of shape [num_tokens]
            indicating which LoRA matrix to use for each token.
        add_inputs (bool): Whether or not to add the input tensor to the output
            tensor.
    """

    # xw32q: when is it used?
    outputs = bgmv_torch(inputs, lora_b_weights, lora_indices_tensor)

    limit = output_tensor.shape[0]
    if outputs.shape[0] == 1 and output_tensor.shape[0] != 1:
        limit = 1

    if output_tensor.shape[1] > outputs.shape[1]:
        outputs = F.pad(outputs,
                        (0, output_tensor.shape[1] - outputs.shape[1], 0, 0))

    if add_inputs:
        return output_tensor + outputs[:limit, :output_tensor.shape[1]]
    else:
        return outputs[:limit, :output_tensor.shape[1]]


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
            [num_loras, lora_rank, hidden_size].
        output_tensor (torch.Tensor): (Unused) output tensor (placeholder).
        lora_indices_tensor (torch.Tensor): Tensor of shape [num_tokens]
            indicating which LoRA matrix to use for each token.
        scaling (float, optional): Scalar multiplier applied to the output.
    """
    # xw32q: when is it used? in qwen2.py.forward -> BaseLinearLayerWithLoRA.apply ->add_lora_linear -> add_shrink -> shrink -> bgmv_shrink.
    # Shouldn't lora_b_weights be named to lora_a_weights since it's shrink?
    # xw32q: what are input shapes?
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
        inputs (torch.Tensor): Input tensor of shape [num_tokens, hidden_size].

        lora_b_weights (torch.Tensor): LoRA weights of shape
            [num_loras, lora_rank, hidden_size].

        output_tensor (torch.Tensor): output tensor of shape
            [num_tokens, hidden_size * num_slices].

        lora_indices_tensor (torch.Tensor): Tensor of shape [num_tokens]
            indicating which LoRA matrix to use for each token.
        add_inputs (bool): Whether or not to add the input tensor to the output
            tensor.
    """
    # xw32q: when is it used? By punica_tpu.expand_slice. inputs.shape=[8192, 8], lora_b_weights.shape=[1, 1, 2048, 8], lora_indices_tensor.shape=[8192].
    # xw32q: How is it different from bgmv_expand? This function has 2 extra parameters: slice_offset and slice_size.
    outputs = bgmv_torch(inputs, lora_b_weights, lora_indices_tensor)

    outputs = F.pad(
        outputs,
        (
            slice_offset,
            output_tensor.shape[1] - (slice_offset + slice_size),
            0,
            0,
        ),
    )

    if add_inputs:
        return output_tensor + outputs
    else:
        return outputs
