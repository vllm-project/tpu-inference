# TODO: create a BaseLayerWithLoRA class because the test uses it.
# create TorchaxColumnParallelLinearWithLoRA

from typing import TYPE_CHECKING, Optional, Union, cast

import torch
import torch.nn as nn
from jax.sharding import PartitionSpec
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
# yapf: enable
from vllm.lora.layers import BaseLayerWithLoRA
# yapf: disable
from vllm.platforms import current_platform

from tpu_commons.distributed.tpu_distributed_utils import \
    create_torchax_tensor_with_partition_spec

if TYPE_CHECKING:
    from vllm.lora.punica_wrapper import PunicaWrapperBase

P = PartitionSpec

"""
Step1: create a base layer (e.g. MergedColumnParallelLinear) and a vLLM LoRA wrapper (via load_lora_model())
Here, we add a LoRA wrapper so the model becomes:
LinearWithLoRA {
    base_layer: Linear
}
The lora weight and linear weight should have been initialized.
Step2: shard_parallel_layers_to_tpu()
Here we replace the linear layer with Torchax layer
LinearWithLora {
    base_layer: JaxLinear
}
then we load the linear weight, shard it, but keep it on CPU.
Step3: move the linear weight to TPU.
Step4: write a function to replace the LoRA wrapper with our own wrapper JaxLinearWithLoRA:
JaxLinearWithLoRA {
    base_layer: JaxLinear
}
Step5: load the lora weight, shard it, and send it to TPU.
"""

class TorchaxBaseLayerWithLoRA(nn.Module):

    # We don't need this because we get the initalized weight from base lora layer.
    # def create_lora_weights(
    #     self,
    #     max_loras: int,
    #     lora_config: LoRAConfig,
    #     model_config: Optional[PretrainedConfig] = None,
    # ) -> None:
    #     """Initializes lora matrices. But I don't think I need it."""
    #     ...

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        ...

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ):
        """Overwrites lora tensors at index."""
        ...

    def set_mapping(
        self,
        punica_wrapper,
    ):
        self.punica_wrapper: PunicaWrapperBase = punica_wrapper

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        raise NotImplementedError

class TorchaxBaseLinearLayerWithLoRA(TorchaxBaseLayerWithLoRA):

    def __init__(self, base_lora_layer: BaseLayerWithLoRA):
        super().__init__()
        self.base_lora_layer = base_lora_layer
        self.base_layer = base_lora_layer.base_layer
        # self.input_size = self.base_layer.input_size
        # self.device = _get_lora_device(self.base_layer)  # do we need it?

        # self.lora_a_stacked, self.lora_b_stacked, self.lora_bias_stacked, self.lora_config are initialized in original LoRA wrapper's create_lora_weight().
        self.lora_config = base_lora_layer.lora_config
        self.lora_a_stacked = tuple(create_torchax_tensor_with_partition_spec(lora_a) for lora_a in base_lora_layer.lora_a_stacked)
        self.lora_b_stacked = tuple(create_torchax_tensor_with_partition_spec(lora_b) for lora_b in base_lora_layer.lora_b_stacked)
        if self.lora_config.bias_enabled:
            self.lora_bias_stacked: Optional[tuple[torch.Tensor, ...]] = tuple(create_torchax_tensor_with_partition_spec(lora_bias) for lora_bias in base_lora_layer.lora_bias_stacked)

        self.output_slices: tuple[int, ...]
        self.output_size: int  # xiowei: remove it.
        self.n_slices: int

    def reset_lora(self, index: int):
        # lora_a_stacked: tuple(torch.Tensor: [max_loras, 1, num_out_features, num_in_features])
        for s_index in range(self.n_slices):
            self.lora_a_stacked[s_index][index] = 0
            self.lora_b_stacked[s_index][index] = 0
            if self.lora_config.bias_enabled:
                # Make mypy happy
                self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
                                              self.lora_bias_stacked)
                self.lora_bias_stacked[s_index][index] = 0

    # def set_lora(
    #     self,
    #     index: int,
    #     lora_a: torch.Tensor,
    #     lora_b: torch.Tensor,
    #     embeddings_tensor: Optional[torch.Tensor],
    #     lora_bias: Optional[torch.Tensor] = None,
    # ):
    #     # Except for QKVParallelLinearWithLoRA and
    #     # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
    #     # store weights in a tuple of size 1. These two layers will
    #     # override this function.
    #     assert (len(self.lora_a_stacked) == len(self.lora_b_stacked) ==
    #             self.n_slices == 1)

    #     self.reset_lora(index)
    #     self.lora_a_stacked[0][index, 0].copy_(lora_a.T, non_blocking=True)
    #     self.lora_a_stacked[0].apply_jax_(jax.device_put, NamedSharding(self.mesh, P()))
    #     self.lora_b_stacked[0][index, 0].copy_(lora_b, non_blocking=True)
    #     self.lora_b_stacked[0].apply_jax_(jax.device_put, NamedSharding(self.mesh, P()))

    #     if lora_bias is not None:
    #         self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
    #                                       self.lora_bias_stacked)
    #         assert len(self.lora_bias_stacked)
    #         self.lora_bias_stacked[0][index, 0].copy_(lora_bias.T, non_blocking=True)
    #         self.lora_bias.stacked[0].apply_jax_(jax.device_put, NamedSharding(self.mesh, P()))
    #     # continue

    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # self.base_layer returns (output, output_bias), we only need the first one.
        output = self.base_layer(x)[0]  # x:[128, 4096], eg self.base_layer is JaxMergedColumnParallelLinear

        # In transformers backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        # TODO(xiowei): check if this is still needed.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        lora_output: Optional[
            torch.Tensor] = self.punica_wrapper.add_lora_linear(
                output, x, self.lora_a_stacked, self.lora_b_stacked,
                self.lora_bias_stacked, 1.0, self.output_slices)
        if not current_platform.can_update_inplace():
            output = lora_output

        return output

    @property
    def weight(self) -> torch.Tensor:
        if hasattr(self.base_layer, "weight"):
            return self.base_layer.weight
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if hasattr(self.base_layer, "bias"):
            return self.base_layer.bias
        else:
            return None


# TODO(xiowei): consider creating a TorchaxColumnParallelLinearWithLoRA next.

class TorchaxMergedColumnParallelLinearWithLoRA(TorchaxBaseLinearLayerWithLoRA):
    """ColumnParallelLinear layer that is composed of 2 sublayers (slices)
    packed together (eg. gate_proj + up_proj -> gate_up_proj).

    This means we have 2 LoRAs, each applied to one half of the layer.

    Both slices must have the same size.
    """

    def __init__(self, base_lora_layer: BaseLayerWithLoRA) -> None:
        # TODO(xiowei): add mesh to the __init__.
        super().__init__(base_lora_layer)
        output_sizes = self.base_layer.output_sizes
        self.output_slices = output_sizes
        # how should I write self.output_slices if I don't know the tp_size?
        self.n_slices = len(output_sizes)

    def forward(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)

        # Matrix multiply.
        output_parallel = self.apply(input_, bias)
        # print(f"{self.base_layer.gather_output=}, {self.base_layer.return_bias=}") print False and False
        if self.base_layer.gather_output:
            # All-gather across the partitions.
            # output = tensor_model_parallel_all_gather(output_parallel)
            pass
        else:
            output = output_parallel

        if not self.base_layer.return_bias:
            return output

        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)
        return output, output_bias


    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        lora_bias: Optional[torch.Tensor] = None,
    ):
        self.reset_lora(index)
        for i in range(self.n_slices):
            lora_a_i = create_torchax_tensor_with_partition_spec(lora_a[i])
            if lora_a_i is not None:
                self.lora_a_stacked[i][
                    index, 0, :lora_a_i.shape[1], :lora_a_i.shape[0]].copy_(
                        lora_a_i.T, non_blocking=True)
                # self.lora_a_stacked[i].apply_jax_(jax.device_put, NamedSharding(self.mesh, P(None, )))
            lora_b_i = create_torchax_tensor_with_partition_spec(lora_b[i])
            if lora_b_i is not None:
                self.lora_b_stacked[i][
                    index, 0, :lora_b_i.shape[1], :lora_b_i.shape[0]].copy_(
                        lora_b_i.T, non_blocking=True)

        if lora_bias is not None:
            self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
                                          self.lora_bias_stacked)
            for i in range(self.n_slices):
                lora_bias_i = create_torchax_tensor_with_partition_spec(lora_bias[i])
                if lora_bias_i is not None:
                    self.lora_bias_stacked[i][index,
                                              0, :lora_bias_i.shape[0]].copy_(
                                                  lora_bias_i.T,
                                                  non_blocking=True)
