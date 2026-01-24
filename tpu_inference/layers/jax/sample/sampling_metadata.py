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
    min_tokens=0,
)

# Max number of stop tokens to track per request for masking
MAX_STOP_TOKENS = 16


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "temperature",
        "top_k",
        "top_p",
        "min_tokens",
        "seq_lens",
        "stop_token_ids",
    ],
    meta_fields=["do_sampling", "logprobs"],
)
@dataclass
class TPUSupportedSamplingMetadata:
    temperature: Optional[jnp.ndarray] = None
    top_k: Optional[jnp.ndarray] = None
    top_p: Optional[jnp.ndarray] = None
    # Added for min_tokens support
    min_tokens: Optional[jnp.ndarray] = None
    seq_lens: Optional[jnp.ndarray] = None
    stop_token_ids: Optional[jnp.ndarray] = None

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
        if input_batch.all_greedy:
            return cls(do_sampling=False, logprobs=needs_logprobs)
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

        # Initialize arrays for min_tokens and stop_token_ids
        # We need to extract these from the input_batch dictionary
        min_tokens_arr = np.zeros(padded_num_reqs, dtype=np.int32)
        stop_token_ids_arr = np.full((padded_num_reqs, MAX_STOP_TOKENS),
                                     -1,
                                     dtype=np.int32)

        # input_batch.req_ids maps batch_index -> request_id
        # input_batch.min_tokens is dict[req_id, tuple[min_tokens, set[stop_ids]]]
        if hasattr(input_batch, 'min_tokens') and input_batch.min_tokens:
            req_ids = input_batch.req_ids_cpu.numpy()
            for i in range(num_reqs):
                req_id = req_ids[i]
                if req_id in input_batch.min_tokens:
                    m_tokens, stop_ids = input_batch.min_tokens[req_id]
                    min_tokens_arr[i] = m_tokens

                    # Convert set to list and pad/truncate
                    stop_ids_list = list(stop_ids)[:MAX_STOP_TOKENS]
                    if stop_ids_list:
                        stop_token_ids_arr[
                            i, :len(stop_ids_list)] = stop_ids_list

        # Get Sequence Lengths (needed to compare against min_tokens)
        # Assuming input_batch has seq_lens_cpu or similar tracking of current length
        if hasattr(input_batch, 'seq_lens_cpu'):
            seq_lens_tensor = fill_slice(input_batch.seq_lens_cpu, 0)
        else:
            # Fallback if seq_lens_cpu is missing (though standard vLLM has it)
            # This is critical for min_tokens to work
            seq_lens_tensor = torch.zeros(padded_num_reqs, dtype=torch.int32)

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
            min_tokens=device_array(mesh,
                                    torch.from_numpy(min_tokens_arr),
                                    sharding=sharding),
            seq_lens=device_array(mesh,
                                  seq_lens_tensor[:padded_num_reqs],
                                  sharding=sharding),
            stop_token_ids=device_array(mesh,
                                        torch.from_numpy(stop_token_ids_arr),
                                        sharding=sharding),
            do_sampling=not input_batch.all_greedy,
            logprobs=needs_logprobs,
        )
