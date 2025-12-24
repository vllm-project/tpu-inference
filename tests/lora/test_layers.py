# Copyright 2025 Google LLC
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

import random
from typing import Optional

import jax
import pytest
import torch
import torchax
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.config import LoRAConfig
# yapf conflicts with isort for this block
# yapf: disable
from vllm.lora.layers import (BaseLayerWithLoRA, ColumnParallelLinearWithLoRA,
                              LoRAMapping, MergedColumnParallelLinearWithLoRA,
                              MergedQKVParallelLinearWithLoRA,
                              QKVParallelLinearWithLoRA,
                              ReplicatedLinearWithLoRA,
                              RowParallelLinearWithLoRA)
# yapf: enable
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.punica_wrapper import get_punica_wrapper
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.utils import set_random_seed
from vllm.platforms import current_platform

from tpu_inference.layers.vllm.quantization.common import JaxCommonLinearConfig
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedLinearMethod
from tpu_inference.layers.vllm.sharding import _shard_module_to_tpu

from .utils import DummyLoRAManager

P = PartitionSpec

TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.float32: (5e-3, 5e-3),
    torch.bfloat16: (3e-2, 2e-2),
}

pytestmark = pytest.mark.skipif(not current_platform.is_tpu(),
                                reason="This test is only for TPU platform.")

# prefill stage(True) or decode stage(False)
STAGES = [True, False]


def check_punica_wrapper(punica_wrapper) -> bool:
    from tpu_inference.lora.torch_punica_tpu import PunicaWrapperTPU
    return type(punica_wrapper) is PunicaWrapperTPU


def get_random_index_to_id(num_loras: int,
                           num_slots: int,
                           log: bool = True) -> list[Optional[int]]:
    """Creates a random index_to_lora_id mapping: slot[index] = lora_id.

    Args:
        num_loras: The number of active loras in the mapping.
        num_slots: The number of slots in the mapping. Must be larger
            than num_loras.
        log: Whether to log the output.

    returns:
        index_to_lora_id: a random index_to_lora_id mapping.
    """

    if num_loras > num_slots:
        raise ValueError(
            f"num_loras is higher than num_slots: {num_loras} > {num_slots}. "
            "num_loras must be less than or equal to num_slots.")

    slots: list[Optional[int]] = [None] * num_slots
    random_slot_selections = (torch.randperm(num_slots)[:num_loras]).tolist()
    for lora_id, slot_idx in enumerate(random_slot_selections, start=1):
        # The slot_idx start at 1.
        slots[slot_idx] = lora_id

    if log:
        print(f"Created lora_id_to_index mapping: {slots}.")

    return slots


def populate_loras(
    index_to_id: list[Optional[int]],
    lora_layer: BaseLayerWithLoRA,
    baselayer_weights: torch.Tensor,
    repeats: int = 1,
) -> tuple[dict[int, LoRALayerWeights], dict[int, list[LoRALayerWeights]]]:
    """This method populates the lora weights (lora_a and lora_b) in the lora layers (BaseLayerWithLoRA).

    Args:
        index_to_id: a list of lora ids. The index of the lora id
            represents which memory slot the lora matrices are
            stored in. A None value indicates a free slot.
        lora_layer: the LoRAlayer to populate.
        baselayer_weights: the PyTorch tensor containing the layer's
            weights.
        repeats: must only be set for column parallel packed
            layers. Indicates the number of loras to compose
            together to create a single lora layer.

    returns:
        lora_dict: a dictionary dict[int, LoRALayerWeights] that maps the lora ID to the corresponding lora weights.
        sublora_dict: a dictionary dict[int, list[LoRALayerWeights]] that maps the lora ID to the corresponding lora weights.
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
            sublora_len = baselayer_weights.shape[0] // repeats
            for i in range(repeats):
                sublora = DummyLoRAManager(
                    baselayer_weights.device).init_random_lora(
                        module_name=f"fake_{i}",
                        weight=baselayer_weights,
                    )
                sublora.lora_b = sublora.lora_b[(sublora_len *
                                                 i):(sublora_len * (i + 1)), :]
                sublora.optimize()
                subloras.append(sublora)

            lora = PackedLoRALayerWeights.pack(
                subloras) if repeats > 1 else subloras[0]

            # Some of the layer.lora is torchax tensor so it can only do math (slice op) in the torchax env.
            with torchax.default_env():
                lora_layer.set_lora(
                    slot_idx,
                    lora_a=lora.lora_a,
                    lora_b=lora.lora_b,
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
@pytest.mark.parametrize("num_loras", [1, 4, 9])
@pytest.mark.parametrize("repeats", [1, 2, 3])
@pytest.mark.parametrize("stage", [True, False])
def test_column_parallel_packed(dist_init, num_loras, repeats, stage) -> None:
    set_random_seed(6)

    max_loras = 9
    max_lora_rank = 8
    lora_config = LoRAConfig(
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        fully_sharded_loras=False,
        lora_dtype=torch.bfloat16,
    )
    vllm_config = dist_init
    vllm_config.lora_config = lora_config

    mesh = _create_mesh()
    linear, lora_linear = _create_column_parallel_packed_layer(
        repeats, vllm_config, mesh)
    _verify_lora_linear_layer(linear, lora_linear)

    # After we create the lora_config, the linear layer and the lora layer,
    # here are the steps to do next:
    # - create a punica wrapper.
    # - associate the punica wrapper with the lora layer.
    # - populate the lora matrices in the lora layer: use non-zero values for testing lora and zero values for testing the case where the layer doesn't have lora.
    # - create inputs and lora_mapping.
    # - update the metadata of the punica wrapper.
    # - convert the inputs to be torchax tensors.
    # - then run a forward on the lora layer to get the actual output.
    # - then run a reference implementation as the expected output.

    # Create a punica wrapper and associate it with the lora linear layer.
    max_num_batched_tokens = 8192
    max_batches = 256
    with torchax.default_env():
        punica_wrapper = get_punica_wrapper(max_num_batched_tokens,
                                            max_batches,
                                            'jax',
                                            max_loras=max_loras)
    assert check_punica_wrapper(punica_wrapper)
    lora_linear.set_mapping(punica_wrapper)

    # Populate lora matrices (lora_a and lora_b) in the lora layer.
    index_to_id = get_random_index_to_id(num_loras, max_loras)
    # lora_dict: lora_id -> LoRALayerWeights|PackedLoRALayerWeights
    lora_dict, sublora_dict = populate_loras(
        index_to_id,
        lora_layer=lora_linear,
        baselayer_weights=linear.weight,
        repeats=repeats,
    )

    # Create inputs and lora mappings.
    # inputs: list[torch.Tensor] of size num_inputs. inputs[i] corresponds to a request which has several token of shape=[num_tokens, 64].
    # index_mapping: list[int]
    # prompt_mapping: list[int]
    inputs, index_mapping, prompt_mapping = create_random_inputs(
        active_lora_ids=list(lora_dict.keys()),
        num_inputs=32,
        input_size=(1, 64),
        input_range=(0, 1),
        input_type=torch.bfloat16,
        device='cpu')

    _update_punica_wrapper_metadata(punica_wrapper, index_mapping,
                                    prompt_mapping, stage, index_to_id,
                                    lora_config)

    with torchax.default_env():
        torchax_inputs = _shard_and_move_inputs_to_tpu(inputs, mesh)
        actual_result = lora_linear(torchax_inputs)[0]

    expected_results: list[torch.Tensor] = []
    for input_, lora_id in zip(inputs, prompt_mapping):
        # linear(input_) returns (output, output_bias) so we only need the first one.
        result = linear(input_)[0]
        subloras = sublora_dict[lora_id]
        for i, sublora in enumerate(subloras):
            result[:, sublora.lora_b.shape[0] * i:sublora.lora_b.shape[0] *
                   (i + 1)] += (input_ @ sublora.lora_a.T @ sublora.lora_b.T *
                                sublora.scaling)
        expected_results.append(result)
    expected_result = torch.cat(expected_results)

    rtol, atol = TOLERANCES[actual_result.dtype]
    with torchax.default_env():
        actual_result_cpu = actual_result.to('cpu')
        torch.testing.assert_close(actual_result_cpu,
                                   expected_result,
                                   rtol=rtol,
                                   atol=atol)
        # print(
        #     f'Output max diff: {torch.max(torch.abs(expected_result - actual_result_cpu))}'
        # )
        # print(
        #     f'Output mean diff: {torch.mean(torch.abs(expected_result - actual_result_cpu))}'
        # )

    # Check that resetting the lora weights succeeds
    # Here we set all lora weight to be empty.
    for slot_idx in range(max_loras):
        lora_linear.reset_lora(slot_idx)

    inputs, index_mapping, prompt_mapping = create_random_inputs(
        active_lora_ids=[0],  # different from the above create_random_inputs
        num_inputs=32,
        input_size=(1, 64),
        input_range=(0, 1),
        input_type=torch.bfloat16,
        device='cpu')

    _update_punica_wrapper_metadata(punica_wrapper, index_mapping,
                                    prompt_mapping, stage, index_to_id,
                                    lora_config)

    with torchax.default_env():
        torchax_inputs = _shard_and_move_inputs_to_tpu(inputs, mesh)
        actual_result = lora_linear(torchax_inputs)[0]
    expected_result = linear(torch.cat(inputs))[0]

    rtol, atol = TOLERANCES[actual_result.dtype]
    with torchax.default_env():
        actual_result_cpu = actual_result.to('cpu')
        torch.testing.assert_close(actual_result_cpu,
                                   expected_result,
                                   rtol=rtol,
                                   atol=atol)


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 4, 9])
@pytest.mark.parametrize("layer_type", ["row", "column", "replicated"])
@pytest.mark.parametrize("stage", [True, False])
def test_linear_parallel(dist_init, num_loras, layer_type, stage) -> None:
    set_random_seed(6)

    max_loras = 9
    max_lora_rank = 8
    lora_config = LoRAConfig(
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        fully_sharded_loras=False,
        lora_dtype=torch.bfloat16,
    )
    vllm_config = dist_init
    vllm_config.lora_config = lora_config

    mesh = _create_mesh()
    linear, lora_linear = _create_random_linear_parallel_layer(
        layer_type, vllm_config, mesh)
    _verify_lora_linear_layer(linear, lora_linear)

    max_num_batched_tokens = 8192
    max_batches = 256
    with torchax.default_env():
        punica_wrapper = get_punica_wrapper(max_num_batched_tokens,
                                            max_batches,
                                            'jax',
                                            max_loras=max_loras)
    assert check_punica_wrapper(punica_wrapper)
    lora_linear.set_mapping(punica_wrapper)

    # Populate lora matrices (lora_a and lora_b) in the lora layer.
    index_to_id = get_random_index_to_id(num_loras, max_loras)
    # lora_dict: lora_id -> LoRALayerWeights|PackedLoRALayerWeights
    lora_dict, sublora_dict = populate_loras(
        index_to_id,
        lora_layer=lora_linear,
        baselayer_weights=linear.weight,
    )

    inputs, index_mapping, prompt_mapping = create_random_inputs(
        active_lora_ids=list(lora_dict.keys()),
        num_inputs=32,
        input_size=(1, 64),
        input_range=(0, 1),
        input_type=torch.bfloat16,
        device='cpu')

    _update_punica_wrapper_metadata(punica_wrapper, index_mapping,
                                    prompt_mapping, stage, index_to_id,
                                    lora_config)

    with torchax.default_env():
        torchax_inputs = _shard_and_move_inputs_to_tpu(inputs, mesh)
        actual_result = lora_linear(torchax_inputs)[0]

    expected_results: list[torch.Tensor] = []
    for input_, lora_id in zip(inputs, prompt_mapping):
        result = linear(input_)[0]
        lora = lora_dict[lora_id]
        lora_result = input_ @ lora.lora_a.T @ lora.lora_b.T * lora.scaling
        result += lora_result
        expected_results.append(result)
    expected_result = torch.cat(expected_results)

    rtol, atol = TOLERANCES[actual_result.dtype]
    with torchax.default_env():
        actual_result_cpu = actual_result.to('cpu')
        torch.testing.assert_close(actual_result_cpu,
                                   expected_result,
                                   rtol=rtol,
                                   atol=atol)

    # Check that resetting the lora weights succeeds
    # Here we set all lora weight to be empty.
    for slot_idx in range(max_loras):
        lora_linear.reset_lora(slot_idx)

    inputs, index_mapping, prompt_mapping = create_random_inputs(
        active_lora_ids=[0],  # different from the above create_random_inputs
        num_inputs=32,
        input_size=(1, 64),
        input_range=(0, 1),
        input_type=torch.bfloat16,
        device='cpu')
    _update_punica_wrapper_metadata(punica_wrapper, index_mapping,
                                    prompt_mapping, stage, index_to_id,
                                    lora_config)

    with torchax.default_env():
        torchax_inputs = _shard_and_move_inputs_to_tpu(inputs, mesh)
        actual_result = lora_linear(torchax_inputs)[0]
    expected_result = linear(torch.cat(inputs))[0]

    rtol, atol = TOLERANCES[actual_result.dtype]
    with torchax.default_env():
        actual_result_cpu = actual_result.to('cpu')
        torch.testing.assert_close(actual_result_cpu,
                                   expected_result,
                                   rtol=rtol,
                                   atol=atol)


def _create_random_linear_parallel_layer(layer_type, vllm_config, mesh):
    # We first create a base linear layer, then a lora layer to wrap it.
    if layer_type == "row":

        def _create_row_linear():
            return RowParallelLinear(
                64,  # input_size
                64,  # output_size
                bias=False,
                params_dtype=torch.bfloat16)

        linear = _create_row_linear()
        linear.weight.data = torch.rand_like(linear.weight.data)

        base_linear = _create_row_linear()
        lora_linear = _create_lora_wrapper(linear,
                                           base_linear,
                                           RowParallelLinearWithLoRA,
                                           vllm_config=vllm_config,
                                           mesh=mesh)
    elif layer_type == "column":

        def _create_column_linear():
            return ColumnParallelLinear(64,
                                        64,
                                        bias=False,
                                        params_dtype=torch.bfloat16)

        linear = _create_column_linear()
        linear.weight.data = torch.rand_like(linear.weight.data)

        base_linear = _create_column_linear()
        lora_linear = _create_lora_wrapper(linear,
                                           base_linear,
                                           ColumnParallelLinearWithLoRA,
                                           vllm_config=vllm_config,
                                           mesh=mesh)

    elif layer_type == "replicated":

        def _create_replicated_linear():
            return ReplicatedLinear(64,
                                    64,
                                    bias=False,
                                    params_dtype=torch.bfloat16)

        linear = _create_replicated_linear()
        linear.weight.data = torch.rand_like(linear.weight.data)

        base_linear = _create_replicated_linear()
        lora_linear = _create_lora_wrapper(linear,
                                           base_linear,
                                           ReplicatedLinearWithLoRA,
                                           vllm_config=vllm_config,
                                           mesh=mesh)

    else:
        raise NotImplementedError("Unknown layer type: {}".format(layer_type))

    return linear, lora_linear


def _get_devices():
    return jax.devices()


def _create_mesh():
    axis_names = ("data", "model")
    devices = _get_devices()
    mesh_shape = (1, len(devices))
    mesh = jax.make_mesh(mesh_shape, axis_names, devices=devices)
    return mesh


def _verify_lora_linear_layer(linear, lora_linear):
    with torchax.default_env():
        # lora_linear.weight has type torchax.tensor.Tensor
        # BaseLinearLayerWithLoRA.weight property guarantees this.
        # if len(devices) != 1, `reorder_concatenated_tensor_for_sharding` function may reorder the out_features dimension of the weight matrix.
        # So the below check will fail.
        if len(_get_devices()) == 1:
            assert torch.equal(linear.weight.data,
                               lora_linear.weight.to('cpu'))


def _shard_and_move_inputs_to_tpu(inputs, mesh):
    processed_inputs = []
    for input in inputs:
        # without `torch_view`, you get an error `AttributeError: 'jaxlib._jax.ArrayImpl' object has no attribute 'apply_jax_'`
        # without `t2j`, you get an error `AttributeError: 'Tensor' object has no attribute 'apply_jax_'`
        jax_input = torch_view(t2j(input))
        jax_input.apply_jax_(jax.device_put,
                             NamedSharding(mesh, P(None, None)))
        processed_inputs.append(jax_input)
    return torch.cat(processed_inputs)


def _update_punica_wrapper_metadata(punica_wrapper, index_mapping,
                                    prompt_mapping, stage, index_to_id,
                                    lora_config):
    lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)
    with torchax.default_env():
        # Here we move the metadata from cpu to tpu.
        punica_wrapper.update_metadata(
            lora_mapping,
            index_to_id,
            lora_config.max_loras,
            vocab_size=512,
        )
        assert jax_view(punica_wrapper._lora_indices_per_batch).platform(
        ) == 'tpu', 'punica_wrapper._lora_indices_per_batch should have been moved to TPU.'
        assert isinstance(
            jax_view(punica_wrapper._lora_indices_per_batch).sharding,
            jax.sharding.SingleDeviceSharding
        ), 'punica_wrapper._lora_indices_per_batch should have been moved to TPU.'


def _create_column_parallel_packed_layer(repeats, vllm_config, mesh):
    # We first create a base linear layer, then a lora layer to wrap it.
    if repeats == 2:
        # In e2e, MergedColumnParallelLinear is created when we load the model. The base_layer weights are sharded and moved to TPU in VllmUnquantizedLinearMethod.process_weights_after_loading.
        def _create_merged_column_linear():
            return MergedColumnParallelLinear(
                64,  # input_size
                [64] * repeats,  # output_size
                bias=False,
                params_dtype=torch.bfloat16)

        linear = _create_merged_column_linear()
        linear.weight.data = torch.rand_like(linear.weight.data)

        base_linear = _create_merged_column_linear()
        lora_linear = _create_lora_wrapper(linear, base_linear,
                                           MergedColumnParallelLinearWithLoRA,
                                           vllm_config, mesh, repeats)
    elif repeats == 3:

        def _create_qkv_linear():
            return QKVParallelLinear(64,
                                     64,
                                     32,
                                     bias=False,
                                     params_dtype=torch.bfloat16)

        linear = _create_qkv_linear()
        linear.weight.data = torch.rand_like(linear.weight.data)

        base_linear = _create_qkv_linear()
        lora_linear = _create_lora_wrapper(linear, base_linear,
                                           MergedQKVParallelLinearWithLoRA,
                                           vllm_config, mesh, repeats)
    else:

        def _create_qkv_linear():
            return QKVParallelLinear(64,
                                     64,
                                     32,
                                     bias=False,
                                     params_dtype=torch.bfloat16)

        linear = _create_qkv_linear()
        linear.weight.data = torch.rand_like(linear.weight.data)

        base_linear = _create_qkv_linear()
        lora_linear = _create_lora_wrapper(linear, base_linear,
                                           QKVParallelLinearWithLoRA,
                                           vllm_config, mesh, repeats)

    return linear, lora_linear


def _create_lora_wrapper(linear,
                         base_linear,
                         lora_cls,
                         vllm_config,
                         mesh,
                         repeats=1):
    base_linear.weight.data = linear.weight.data
    jax_config = JaxCommonLinearConfig(vllm_config, mesh, base_linear)
    linear_method = VllmUnquantizedLinearMethod(jax_config)
    base_linear.quant_method = linear_method
    linear_method.process_weights_after_loading(
        base_linear)  # here base_linear.weight is moved to TPU and sharded.
    assert jax_view(base_linear.weight).platform(
    ) == 'tpu', 'base_linear.weight should have been moved to TPU.'
    assert not isinstance(
        jax_view(base_linear.weight).sharding, jax.sharding.
        SingleDeviceSharding), 'base_linear.weight should have been sharded.'

    lora_linear = lora_cls(base_linear)

    lora_config = vllm_config.lora_config
    max_loras = lora_config.max_loras
    with torchax.default_env():
        lora_linear.create_lora_weights(max_loras, lora_config)
    # In the e2e, the lora_layer's weight is moved to TPU in _shard_module_to_tpu.
    _shard_module_to_tpu(lora_linear, mesh)

    assert jax_view(lora_linear.lora_a_stacked[0]).platform(
    ) == 'tpu', 'lora_a_stacked should have been moved to TPU.'
    assert not isinstance(
        jax_view(lora_linear.lora_a_stacked[0]).sharding, jax.sharding.
        SingleDeviceSharding), 'lora_a_stacked should have been sharded.'
    assert jax_view(lora_linear.lora_b_stacked[0]).platform(
    ) == 'tpu', 'lora_b_stacked should have been moved to TPU.'
    assert not isinstance(
        jax_view(lora_linear.lora_b_stacked[0]).sharding, jax.sharding.
        SingleDeviceSharding), 'lora_b_stacked should have been sharded.'
    n_slices = repeats
    assert (lora_linear.n_slices == len(lora_linear.lora_a_stacked) == len(
        lora_linear.lora_b_stacked) == n_slices)

    return lora_linear
