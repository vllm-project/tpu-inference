import functools
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_commons.runner.jax.input_batch_jax import InputBatch

DEFAULT_SAMPLING_PARAMS = dict(
    temperature=-1.0,
    top_k=0,
    top_p=1.0,
)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "temperature",
        "top_k",
        "top_p",
    ],
    meta_fields=["do_sampling"],
)
@dataclass
class TPUSupportedSamplingMetadata:
    temperature: Optional[jnp.ndarray] = None
    top_k: Optional[jnp.ndarray] = None
    top_p: Optional[jnp.ndarray] = None
    do_sampling: bool = False

    @classmethod
    def from_input_batch(
        cls,
        mesh: Mesh,
        input_batch: InputBatch,
        padded_num_reqs: int,
    ) -> "TPUSupportedSamplingMetadata":
        if input_batch.all_greedy:
            return cls(do_sampling=False)

        num_reqs = input_batch.num_reqs

        def fill_slice(cpu_torch_tensor: torch.Tensor,
                       fill_val: float) -> torch.Tensor:
            # Pad value is the default one.
            cpu_torch_tensor[num_reqs:padded_num_reqs] = fill_val
            return cpu_torch_tensor

        temp_tensor = fill_slice(input_batch.temperature_cpu,
                                 DEFAULT_SAMPLING_PARAMS["temperature"])
        top_k_tensor = fill_slice(input_batch.top_k_cpu,
                                  DEFAULT_SAMPLING_PARAMS["top_k"])
        top_p_tensor = fill_slice(input_batch.top_p_cpu,
                                  DEFAULT_SAMPLING_PARAMS["top_p"])

        def _device_array(cpu_tensor):
            sharding = NamedSharding(mesh, PartitionSpec(None))
            return jax.device_put(cpu_tensor, device=sharding)

        # Slice persistent device tensors to a fixed pre-compiled padded shape.
        return cls(
            temperature=_device_array(temp_tensor[:padded_num_reqs]),
            top_p=_device_array(top_p_tensor[:padded_num_reqs]),
            top_k=_device_array(top_k_tensor[:padded_num_reqs]),
            do_sampling=not input_batch.all_greedy,
        )
