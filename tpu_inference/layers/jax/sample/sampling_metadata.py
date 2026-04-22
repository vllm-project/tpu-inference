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
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tpu_inference.utils import DeviceBuffer

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from tpu_inference.runner.input_batch import InputBatch

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
        buffer: "DeviceBuffer",
        input_batch: InputBatch,
        padded_num_reqs_per_dp_rank: int,
        dp_size: int,
    ):
        """Add sampling parameters to the device buffer."""
        num_reqs = input_batch.num_reqs

        temp_view = buffer.get_view((padded_num_reqs_per_dp_rank, ),
                                    key="temperature")
        top_k_view = buffer.get_view((padded_num_reqs_per_dp_rank, ),
                                     key="top_k")
        top_p_view = buffer.get_view((padded_num_reqs_per_dp_rank, ),
                                     key="top_p")

        needs_logprobs = input_batch.max_num_logprobs > 0 if input_batch.max_num_logprobs else False
        dummy_len = 1 if needs_logprobs else 2
        dummy_view = buffer.get_view((dummy_len, ),
                                     key="cache_collision_dummy")
        dummy_view.fill(0)

        # Fill with defaults first (bitcasted to int32 for floats)
        default_temp_int = np.array([DEFAULT_SAMPLING_PARAMS["temperature"]],
                                    dtype=np.float32).view(np.int32)[0]
        default_top_p_int = np.array([DEFAULT_SAMPLING_PARAMS["top_p"]],
                                     dtype=np.float32).view(np.int32)[0]

        temp_view.fill(default_temp_int)
        top_k_view.fill(DEFAULT_SAMPLING_PARAMS["top_k"])
        top_p_view.fill(default_top_p_int)

        # If not all greedy, copy over actual values directly to the view
        if input_batch.all_greedy:
            return

        temp_view.ravel(
        )[:num_reqs] = input_batch.temperature_cpu[:num_reqs].view(np.int32)
        top_k_view.ravel()[:num_reqs] = input_batch.top_k_cpu[:num_reqs]
        top_p_view.ravel()[:num_reqs] = input_batch.top_p_cpu[:num_reqs].view(
            np.int32)

    @classmethod
    def from_unpacked_blob(
        cls,
        mesh: Mesh,
        metadata: dict[str, jax.Array],
        input_batch: InputBatch,
        padded_num_reqs: int,
    ) -> "TPUSupportedSamplingMetadata":
        """Unpack sampling parameters from the metadata blob."""
        needs_logprobs = input_batch.max_num_logprobs > 0 if input_batch.max_num_logprobs else False
        cache_collision_dummy = metadata["cache_collision_dummy"].ravel()

        if input_batch.all_greedy:
            return cls(do_sampling=False,
                       logprobs=needs_logprobs,
                       _cache_collision_dummy=cache_collision_dummy)

        temp = metadata["temperature"].ravel()
        top_k = metadata["top_k"].ravel()
        top_p = metadata["top_p"].ravel()

        # Bitcast back to float32 if needed
        temp = jax.lax.bitcast_convert_type(temp, jnp.float32)
        top_p = jax.lax.bitcast_convert_type(top_p, jnp.float32)

        return cls(
            temperature=temp[:padded_num_reqs],
            top_p=top_p[:padded_num_reqs],
            top_k=top_k[:padded_num_reqs],
            _cache_collision_dummy=cache_collision_dummy,
            do_sampling=not input_batch.all_greedy,
            logprobs=needs_logprobs,
        )
