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

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from tpu_inference.runner.input_batch import InputBatch

if TYPE_CHECKING:
    from tpu_inference.utils import DeviceBuffer

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
        "_cache_collision_dummy",
    ],
    meta_fields=["do_sampling", "logprobs"],
)
@dataclass
class TPUSupportedSamplingMetadata:
    temperature: Optional[jnp.ndarray] = None
    top_k: Optional[jnp.ndarray] = None
    top_p: Optional[jnp.ndarray] = None
    _cache_collision_dummy: Optional[jnp.ndarray] = None
    do_sampling: bool = False
    logprobs: bool = False

    @classmethod
    def add_to_device_buffer(
        cls,
        input_batch: InputBatch,
        padded_num_reqs: int,
        device_buffer: "DeviceBuffer",
    ):
        needs_logprobs = (input_batch.max_num_logprobs > 0
                          if input_batch.max_num_logprobs else False)

        # Use a dummy tensor with a unique shape for each logprobs config.
        # This avoids persistent cache collisions.
        dummy_shape = (1 if needs_logprobs else 2, )
        cache_collision_dummy_view = device_buffer.get_view(
            dummy_shape, key="cache_collision_dummy")
        cache_collision_dummy_view.fill(0)

        if input_batch.all_greedy:
            return

        num_reqs = input_batch.num_reqs

        temp_view = device_buffer.get_view((padded_num_reqs, ),
                                           key="temperature")
        top_k_view = device_buffer.get_view((padded_num_reqs, ), key="top_k")
        top_p_view = device_buffer.get_view((padded_num_reqs, ), key="top_p")

        # Pad values
        temp_view[num_reqs:].fill(
            np.array(DEFAULT_SAMPLING_PARAMS["temperature"],
                     dtype=np.float32).view(np.int32))
        top_k_view[num_reqs:].fill(DEFAULT_SAMPLING_PARAMS["top_k"])
        top_p_view[num_reqs:].fill(
            np.array(DEFAULT_SAMPLING_PARAMS["top_p"],
                     dtype=np.float32).view(np.int32))

        # Copy data
        np.copyto(temp_view[:num_reqs],
                  input_batch.temperature_cpu[:num_reqs].view(np.int32))
        np.copyto(top_k_view[:num_reqs], input_batch.top_k_cpu[:num_reqs])
        np.copyto(top_p_view[:num_reqs],
                  input_batch.top_p_cpu[:num_reqs].view(np.int32))

    @classmethod
    def from_unpacked_arrays(
        cls,
        mesh: Mesh,
        unpacked_metadata: Dict[str, jax.Array],
        input_batch: InputBatch,
    ) -> "TPUSupportedSamplingMetadata":
        needs_logprobs = (input_batch.max_num_logprobs > 0
                          if input_batch.max_num_logprobs else False)

        cache_collision_dummy = unpacked_metadata["cache_collision_dummy"]

        if input_batch.all_greedy:
            return cls(do_sampling=False,
                       logprobs=needs_logprobs,
                       _cache_collision_dummy=cache_collision_dummy)

        return cls(
            temperature=unpacked_metadata["temperature"].view(jnp.float32),
            top_k=unpacked_metadata["top_k"],
            top_p=unpacked_metadata["top_p"].view(jnp.float32),
            _cache_collision_dummy=cache_collision_dummy,
            do_sampling=True,
            logprobs=needs_logprobs,
        )
