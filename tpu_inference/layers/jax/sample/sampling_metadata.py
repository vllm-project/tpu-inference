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
    min_p=0.0,
    repetition_penalty=1.0,
)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "repetition_penalty",
        "seen_token_ids_mask",
        "_cache_collision_dummy",
    ],
    meta_fields=["do_sampling", "logprobs", "has_repetition_penalty"],
)
@dataclass
class TPUSupportedSamplingMetadata:
    temperature: Optional[jnp.ndarray] = None
    top_k: Optional[jnp.ndarray] = None
    top_p: Optional[jnp.ndarray] = None
    min_p: Optional[jnp.ndarray] = None
    # Per-request repetition penalty (num_reqs,) and the incrementally
    # maintained "already seen (prompt+output) tokens" mask (num_reqs, vocab).
    # Both are only populated (and `has_repetition_penalty` set) when at least
    # one active request sets repetition_penalty != 1.0, so the common case
    # compiles to a program with no repetition-penalty work at all.
    repetition_penalty: Optional[jnp.ndarray] = None
    seen_token_ids_mask: Optional[jnp.ndarray] = None
    _cache_collision_dummy: Optional[jnp.ndarray] = None
    do_sampling: bool = False
    logprobs: bool = False
    has_repetition_penalty: bool = False

    @classmethod
    def from_input_batch(
        cls,
        mesh: Mesh,
        input_batch: InputBatch,
        padded_num_reqs: int,
        sharding: Optional[jax.sharding.Sharding] = None,
        seen_token_ids_mask: Optional[jax.Array] = None,
    ) -> "TPUSupportedSamplingMetadata":
        needs_logprobs = input_batch.max_num_logprobs > 0 if input_batch.max_num_logprobs else False

        # Use a dummy tensor with a unique shape for each logprobs config.
        # This avoids persistent cache collisions.
        dummy_shape = (1 if needs_logprobs else 2, )
        cache_collision_dummy = np.zeros(dummy_shape, dtype=np.int32)
        # Use replicated sharding for dummy tensor.
        cache_collision_dummy = device_array(mesh,
                                             cache_collision_dummy,
                                             sharding=None)

        num_reqs = input_batch.num_reqs

        def fill_slice(cpu_torch_tensor: torch.Tensor,
                       fill_val: float) -> torch.Tensor:
            # Pad value is the default one.
            cpu_torch_tensor[num_reqs:padded_num_reqs] = fill_val
            return cpu_torch_tensor

        # Repetition-penalty gate: only engage when some active request sets it
        # (!= 1.0). Keeping this a meta_field means the common case compiles to a
        # program with no repetition-penalty / seen-mask work at all. Applied
        # *before* temperature (matching vLLM's penalties -> temperature order),
        # so it affects both greedy and sampled rows -- hence it is populated
        # even for all-greedy batches.
        rep_cpu = input_batch.repetition_penalties_cpu
        has_rep = bool(np.any(rep_cpu[:num_reqs] != 1.0)) if num_reqs else False
        repetition_penalty = None
        if has_rep:
            assert seen_token_ids_mask is not None, (
                "repetition_penalty is active but the runner did not provide "
                "seen_token_ids_mask")
            rep_tensor = fill_slice(
                rep_cpu, DEFAULT_SAMPLING_PARAMS["repetition_penalty"])
            repetition_penalty = device_array(mesh,
                                              rep_tensor[:padded_num_reqs],
                                              sharding=sharding)
        # Only carry the (num_reqs, vocab) mask when the penalty is engaged.
        seen_mask = seen_token_ids_mask if has_rep else None

        if input_batch.all_greedy:
            return cls(do_sampling=False,
                       logprobs=needs_logprobs,
                       _cache_collision_dummy=cache_collision_dummy,
                       has_repetition_penalty=has_rep,
                       repetition_penalty=repetition_penalty,
                       seen_token_ids_mask=seen_mask)

        temp_tensor = fill_slice(input_batch.temperature_cpu,
                                 DEFAULT_SAMPLING_PARAMS["temperature"])
        top_k_tensor = fill_slice(input_batch.top_k_cpu,
                                  DEFAULT_SAMPLING_PARAMS["top_k"])
        top_p_tensor = fill_slice(input_batch.top_p_cpu,
                                  DEFAULT_SAMPLING_PARAMS["top_p"])
        min_p_tensor = fill_slice(input_batch.min_p_cpu,
                                  DEFAULT_SAMPLING_PARAMS["min_p"])

        # Slice persistent device tensors to a fixed pre-compiled padded shape.
        return cls(
            temperature=device_array(mesh,
                                     temp_tensor[:padded_num_reqs],
                                     sharding=sharding),
            top_p=device_array(mesh,
                               top_p_tensor[:padded_num_reqs],
                               sharding=sharding),
            top_k=device_array(mesh,
                               top_k_tensor[:padded_num_reqs],
                               sharding=sharding),
            min_p=device_array(mesh,
                               min_p_tensor[:padded_num_reqs],
                               sharding=sharding),
            repetition_penalty=repetition_penalty,
            seen_token_ids_mask=seen_mask,
            _cache_collision_dummy=cache_collision_dummy,
            do_sampling=not input_batch.all_greedy,
            logprobs=needs_logprobs,
            has_repetition_penalty=has_rep,
        )