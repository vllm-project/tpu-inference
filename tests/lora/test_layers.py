import random
from typing import Optional

import jax
import pytest
import torch
import torchax
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import torch_view
from torchax.ops.mappings import j2t, t2j
from vllm.config import LoRAConfig
# yapf conflicts with isort for this block
# yapf: disable
from vllm.lora.layers import LoRAMapping, MergedColumnParallelLinearWithLoRA
# yapf: enable
from vllm.lora.models import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.punica_wrapper import get_punica_wrapper
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.utils import set_random_seed
from vllm.platforms import current_platform

from tpu_commons.distributed.tpu_distributed_utils import \
    create_torchax_tensor_with_partition_spec
from tpu_commons.lora.layers import (TorchaxBaseLayerWithLoRA,
                                     TorchaxMergedColumnParallelLinearWithLoRA)
from tpu_commons.models.vllm.sharding import shard_parallel_layers_to_tpu

from .utils import DummyLoRAManager

# TODO(xiowei):
# - add test for multi-chip.
# - add equivalent test for ColumnParallelLinearWithShardedLoRA.

P = PartitionSpec

TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.float32: (5e-3, 5e-3),
    torch.bfloat16: (3e-2, 2e-2),
}

pytestmark = pytest.mark.skipif(not current_platform.is_tpu(),
                                reason="This test is only for TPU platform.")

NUM_RANDOM_SEEDS = 2

# prefill stage(True) or decode stage(False)
STAGES = [True, False]


def check_punica_wrapper(punica_wrapper) -> bool:
    from tpu_commons.lora.torch_punica_tpu import PunicaWrapperTPU
    return type(punica_wrapper) is PunicaWrapperTPU


def get_random_index_to_id(num_loras: int,
                           num_slots: int,
                           log: bool = True) -> list[Optional[int]]:
    """Creates a random index_to_lora_id mapping.

    Args:
        num_loras: The number of active loras in the mapping.
        num_slots: The number of slots in the mapping. Must be larger
            than num_loras.
        log: Whether to log the output.
    """

    if num_loras > num_slots:
        raise ValueError(
            f"num_loras is higher than num_slots: {num_loras} > {num_slots}. "
            "num_loras must be less than or equal to num_slots.")

    slots: list[Optional[int]] = [None] * num_slots
    random_slot_selections = (torch.randperm(num_slots)[:num_loras]).tolist()
    for lora_id, slot_idx in enumerate(random_slot_selections, start=1):
        slots[slot_idx] = lora_id

    if log:
        print(f"Created lora_id_to_index mapping: {slots}.")

    return slots


def populate_loras(
    index_to_id: list[Optional[int]],
    layer: TorchaxBaseLayerWithLoRA,
    layer_weights: torch.Tensor,
    generate_embeddings_tensor: int = 0,
    repeats: int = 1,
) -> tuple[dict[int, LoRALayerWeights], dict[int, list[LoRALayerWeights]]]:
    """This method populates the lora layers (TorchaxBaseLayerWithLoRA) with lora weights.

    Args:
        index_to_id: a list of lora ids. The index of the lora id
            represents which memory slot the lora matrices are
            stored in. A None value indicates a free slot.
        layer: the LoRAlayer to populate.
        layer_weights: the PyTorch tensor containing the layer's
            weights.
        generate_embeddings_tensor: whether to generate an
            embeddings tensor for each LoRA.
        repeats: must only be set for column parallel packed
            layers. Indicates the number of loras to compose
            together to create a single lora layer.
    """

    # Dictionary that maps the lora ID to the
    # corresponding lora weights.
    lora_dict: dict[int, LoRALayerWeights] = dict()

    # Dictionary that maps the lora ID to the
    # corresponding subloras.
    sublora_dict: dict[int, list[LoRALayerWeights]] = dict()

    for slot_idx, lora_id in enumerate(index_to_id):
        if lora_id is not None:
            subloras: list[LoRALayerWeights] = []
            sublora_len = layer_weights.shape[0] // repeats
            for i in range(repeats):
                sublora = DummyLoRAManager(
                    layer_weights.device).init_random_lora(
                        module_name=f"fake_{i}",
                        weight=layer_weights,
                        generate_embeddings_tensor=generate_embeddings_tensor,
                    )
                sublora.lora_b = sublora.lora_b[:, (sublora_len *
                                                    i):(sublora_len * (i + 1))]
                sublora.bias = sublora.bias[(sublora_len * i):(sublora_len *
                                                               (i + 1))]
                sublora.optimize()
                subloras.append(sublora)

            lora = PackedLoRALayerWeights.pack(
                subloras) if repeats > 1 else subloras[0]

            with torchax.default_env(), jax.default_device(
                    jax.devices("tpu")[0]):
                layer.set_lora(
                    slot_idx,
                    lora_a=lora.lora_a,
                    lora_b=lora.lora_b,
                    embeddings_tensor=lora.embeddings_tensor,
                    lora_bias=lora.bias,
                )

            lora_dict[lora_id] = lora
            sublora_dict[lora_id] = subloras

    return lora_dict, sublora_dict


def create_random_inputs(
    active_lora_ids: list[int],
    num_inputs: int,
    input_size: tuple[int, ...],
    input_range: tuple[float, float],
    input_type: torch.dtype = torch.int,
    device: torch.device = "cpu",
) -> tuple[list[torch.Tensor], list[int], list[int]]:
    """Creates random inputs.

    Args:
        active_lora_ids: lora IDs of active lora weights.
        num_inputs: the number of inputs to create. Or the number of requests.
        input_size: the size of each individual input. Or the number of tokens.
        input_range: the range of values to include in the input.
            input_range[0] <= possible input values < input_range[1]
        input_type: the type of values in the input.

    returns:
        inputs: a list of torch tensors of size num_inputs. Each input has shape `input_size`.
        index_mapping: maps each input token to a lora ID.
        prompt_mapping: maps each request to a lora ID.
    """

    low, high = input_range

    inputs: list[torch.Tensor] = []
    index_mapping: list[int] = []
    prompt_mapping: list[int] = []

    for _ in range(num_inputs):
        if input_type == torch.int:
            inputs.append(
                torch.randint(low=int(low),
                              high=int(high),
                              size=input_size,
                              device=device))
        else:
            inputs.append(
                torch.rand(size=input_size, dtype=input_type, device=device) *
                high + low)

        lora_id = random.choice(active_lora_ids)
        index_mapping += [lora_id] * input_size[0]
        prompt_mapping += [lora_id]

    return inputs, index_mapping, prompt_mapping


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [4])  # xw32: use [1,2,4,8]
@pytest.mark.parametrize("repeats", [2])
@pytest.mark.parametrize("fully_shard", [False])  # TODO(xiowei): add "True".
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("stage", [True, False])
@pytest.mark.parametrize("bias_enabled", [True, False])
def test_column_parallel_packed(dist_init, num_loras, repeats, fully_shard,
                                device, stage, bias_enabled) -> None:
    max_loras = 7
    max_num_batched_tokens = 8192
    max_batches = 256
    punica_wrapper = get_punica_wrapper(max_num_batched_tokens,
                                        max_batches,
                                        device,
                                        max_loras=max_loras)
    assert check_punica_wrapper(punica_wrapper)
    lora_config = LoRAConfig(max_loras=max_loras,
                             max_lora_rank=8,
                             fully_sharded_loras=fully_shard,
                             lora_dtype=torch.float16,
                             bias_enabled=bias_enabled)

    axis_names = ("data", "model")
    mesh_shape = (
        1, 1
    )  # TODO(xiowei): support multi-chip: mesh_shape = (1, len(jax.devices()))
    mesh = jax.make_mesh(mesh_shape, axis_names, devices=jax.devices())

    def create_column_parallel_packed_layer():
        # Step 1: create a base layer (e.g. MergedColumnParallelLinear) and a vLLM LoRA wrapper.
        if repeats == 2:
            linear = MergedColumnParallelLinear(
                4096,
                [4096] * repeats,  # input_size, output_size
                bias=False,
                params_dtype=torch.float16)
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = MergedColumnParallelLinearWithLoRA(
                linear
            )  # TODO(xiowei): add test for MergedColumnParallelLinearWithShardedLoRA (fully_shard == True)
        elif repeats == 3:
            # TODO(xiowei): add test for this case.
            raise NotImplementedError("NYI: for MergedQKVParallelLinear case")
        else:
            # TODO(xiowei): add test for this case.
            raise NotImplementedError("NYI: for QKVParallelLinear case")

        n_slices = repeats
        # create_lora_weights creates global shape weight.
        lora_linear.create_lora_weights(max_loras, lora_config)
        assert (lora_linear.n_slices == len(lora_linear.lora_a_stacked) == len(
            lora_linear.lora_b_stacked) == n_slices)
        if bias_enabled:
            assert len(lora_linear.lora_bias_stacked) == lora_linear.n_slices
        else:
            assert lora_linear.lora_bias_stacked is None

        with torchax.default_env(), jax.default_device(jax.devices("tpu")[0]):
            # Then we replace the base layer (e.g. MergedColumnParallelLinear) with the torchax one (e.g. JaxMergedColumnParallelLinear).
            vllm_config = dist_init
            shard_parallel_layers_to_tpu(lora_linear, mesh, vllm_config)

            # replace the LoRA wrapper with our own wrapper  (e.g. TorchaxMergedColumnParallelLinearWithLoRA)
            torchax_lora_linear = TorchaxMergedColumnParallelLinearWithLoRA(
                lora_linear)

        return linear, torchax_lora_linear

    for i in range(NUM_RANDOM_SEEDS):
        set_random_seed(i)

        linear, torchax_lora_linear = create_column_parallel_packed_layer()
        # linear.weight has type torch.nn.Parameter, lora_linear.weight has type torchax.tensor.Tensor
        # BaseLinearLayerWithLoRA.weight property guarantees this.
        with torchax.default_env():
            assert torch.equal(
                create_torchax_tensor_with_partition_spec(linear.weight.data),
                torchax_lora_linear.weight)
        torchax_lora_linear.set_mapping(punica_wrapper)

        # load the lora weight, shard it, and send it to TPU.
        # create a lora slot index to lora id mapping.
        index_to_id = get_random_index_to_id(num_loras, max_loras)
        # lora_dict: lora_id -> LoRALayerWeights|PackedLoRALayerWeights
        lora_dict, sublora_dict = populate_loras(
            index_to_id,
            layer=torchax_lora_linear,
            layer_weights=linear.weight,
            repeats=repeats,
        )

        # inputs: list[torch.Tensor] of size num_inputs. inputs[0].shape=[1, 4096].
        # index_mapping: list[int]
        # prompt_mapping: list[int]
        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=list(lora_dict.keys()),
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device)
        lora_mapping = LoRAMapping(index_mapping,
                                   prompt_mapping,
                                   is_prefill=stage)

        punica_wrapper.update_metadata(
            lora_mapping,
            index_to_id,
            max_loras,
            512,
            lora_config.lora_extra_vocab_size,
        )

        jax_inputs = []
        with torchax.default_env(), jax.default_device(jax.devices("tpu")[0]):
            for input in inputs:
                # without `torch_view`, you get an error `AttributeError: 'jaxlib._jax.ArrayImpl' object has no attribute 'apply_jax_'`
                # without `t2j`, you get an error `AttributeError: 'Tensor' object has no attribute 'apply_jax_'`
                jax_input = torch_view(t2j(input))
                jax_input.apply_jax_(jax.device_put,
                                     NamedSharding(mesh, P(None, None)))
                jax_inputs.append(jax_input)
        with torchax.default_env():
            lora_result = torchax_lora_linear(torch.cat(jax_inputs))[0]
            lora_result = j2t(lora_result)

        # xw32: what's the value of sublora.scaling? I think the test doesn't set it while it should. sublora.scaling seems to be wrong.
        expected_results: list[torch.Tensor] = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            # linear(input_) returns (output, output_bias) so we only need the first one.
            result = linear(input_)[0]
            subloras = sublora_dict[lora_id]
            for i, sublora in enumerate(subloras):
                result[:, sublora.lora_b.shape[1] * i:sublora.lora_b.shape[1] *
                       (i + 1)] += (input_ @ sublora.lora_a @ sublora.lora_b *
                                    sublora.scaling)
                if bias_enabled:
                    result[:, sublora.lora_b.shape[1] *
                           i:sublora.lora_b.shape[1] * (i + 1)] += sublora.bias
            expected_results.append(result)
        expected_result = torch.cat(expected_results)

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result,
                                   expected_result,
                                   rtol=rtol,
                                   atol=atol)
        # print(f'Output max diff: {torch.max(torch.abs(expected_result - lora_result))}')
        # print(f'Output mean diff: {torch.mean(torch.abs(expected_result - lora_result))}')

        # Check that resetting the lora weights succeeds
        for slot_idx in range(max_loras):
            torchax_lora_linear.reset_lora(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[
                0
            ],  # different from the above create_random_inputs
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device)
        lora_mapping = LoRAMapping(index_mapping,
                                   prompt_mapping,
                                   is_prefill=stage)

        punica_wrapper.update_metadata(
            lora_mapping,
            index_to_id,
            max_loras,
            512,
            lora_config.lora_extra_vocab_size,
        )

        jax_inputs = []
        with torchax.default_env(), jax.default_device(jax.devices("tpu")[0]):
            for input in inputs:
                jax_input = torch_view(t2j(input))
                jax_input.apply_jax_(jax.device_put,
                                     NamedSharding(mesh, P(None, None)))
                jax_inputs.append(jax_input)
        with torchax.default_env():
            lora_result = torchax_lora_linear(torch.cat(jax_inputs))[0]
            lora_result = j2t(lora_result)
        expected_result = linear(torch.cat(inputs))[0]

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result,
                                   expected_result,
                                   rtol=rtol,
                                   atol=atol)
