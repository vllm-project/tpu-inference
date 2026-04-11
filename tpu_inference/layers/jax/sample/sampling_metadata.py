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
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh

from tpu_inference.runner.input_batch import InputBatch
from tpu_inference.utils import device_array

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
    def from_input_batch(
        cls,
        mesh: Mesh,
        input_batch: InputBatch,
        padded_num_reqs: int,
        sharding: Optional[jax.sharding.Sharding] = None,
    ) -> "TPUSupportedSamplingMetadata":
        needs_logprobs = input_batch.max_num_logprobs > 0 if input_batch.max_num_logprobs else False

        # Use a dummy tensor with a unique shape for each logprobs config.
        # This avoids persistent cache collisions.
        n = padded_num_reqs
        dummy_shape = (n if needs_logprobs else 2*n, )
        cache_collision_dummy = np.zeros(dummy_shape, dtype=np.int32)

        if input_batch.all_greedy:
            cache_collision_dummy_dev = device_array(mesh,
                                                     cache_collision_dummy,
                                                     sharding=None)
            return cls(do_sampling=False,
                       logprobs=needs_logprobs,
                       _cache_collision_dummy=cache_collision_dummy_dev)
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

        # Slice persistent device tensors to a fixed pre-compiled padded shape.
        # Concatenate all sampling params and the dummy tensor into a single
        # blob, transfer once, then split on device to avoid multiple
        # device_put calls.
        # top_k and dummy are int32 so we cast to float32 for the concat and
        # back after.
        n = padded_num_reqs
        sampling_blob = np.concatenate([
            temp_tensor[:n],
            top_p_tensor[:n],
            top_k_tensor[:n].astype(np.float32),
            cache_collision_dummy.astype(np.float32),
        ])
        sampling_blob_dev = jax.device_put(sampling_blob, sharding)
        split_indices = [n, 2 * n, 3 * n]
        temp_dev, top_p_dev, top_k_f32, dummy_f32 = jnp.split(
            sampling_blob_dev, split_indices)
        top_k_dev = top_k_f32.astype(jnp.int32)
        cache_collision_dummy_dev = dummy_f32.astype(jnp.int32)

        return cls(
            temperature=temp_dev,
            top_p=top_p_dev,
            top_k=top_k_dev,
            _cache_collision_dummy=cache_collision_dummy_dev,
            do_sampling=not input_batch.all_greedy,
            logprobs=needs_logprobs,
        )
