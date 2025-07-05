# Copyright 2024 Google LLC
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
"""Defines the JetStream API.

These functions are the accelerator functions which an outer sampling loop
could want to call, enabling interleaved (continuous batching) inference.
"""

import abc
from typing import Any, Union

import jax
import numpy as np
from flax import struct

# The model parameters - their partitioning will be unique for different prefill
# and decode topoologies.
Params = Any
# The result of a prefill operation, often a batch size 1 KVCache.
Prefix = Any
# The inputs into a generation step, often a prefill and generate cache tuple.
DecodeState = Any
# Accelerator representation of tokens.
DeviceTokens = Any
# Cpus asscociated with the mesh.
CpuDevices = Any
# Tokenkizer used by the engine
Tokenizer = Any
# PRNG key used for prefilling
PRNGKeyType = Any


@struct.dataclass
class ExistingPrefix:
    cache: Any
    common_prefix_tokens: jax.Array


@struct.dataclass
class SlotData:
    """Class to store slot data."""

    tokens: Union[jax.Array, np.ndarray]
    valid: Union[jax.Array, np.ndarray]
    lengths: Union[jax.Array, np.ndarray]
    log_prob: Union[jax.Array, np.ndarray] = None


# pylint: disable=g-doc-args
@struct.dataclass
class ResultTokens(abc.ABC):
    """Class to store returned tokens in.

  We store everything in one array, and keep indexes - because copying
  a single array to host is much faster.
  Each tuple represents the indices of the relevant data.
  """

    # Shape: [batch, tokens.shape[1] + validity.shape[1] + lengths.shape[1]]
    data: Union[jax.Array, np.ndarray]
    # The range of indices which contain tokens.
    tokens_idx: tuple[int, int] = struct.field(pytree_node=False, )
    # The range of indices which contain the validity of
    # the tokens.
    valid_idx: tuple[int, int] = struct.field(pytree_node=False, )
    # The range of indices which contain the lengths up till now of the lengths
    # of each generated sequence.
    length_idx: tuple[int, int] = struct.field(pytree_node=False, )
    samples_per_slot: int = struct.field(pytree_node=False, )
    # log probabilities of the tokens. Shape: [batch, tokens]
    log_prob: Union[jax.Array, np.ndarray] = struct.field(
        pytree_node=False,
        default=None,
    )

    def copy_to_host_async(self: "ResultTokens") -> None:
        """Copy to host asynchronously."""
        # Do nothing for np array
        if isinstance(self.data, np.ndarray):
            return
        self.data.copy_to_host_async()

    def convert_to_numpy(self: "ResultTokens") -> "ResultTokens":
        """Converts to numpy."""
        return ResultTokens(
            np.array(self.data),
            self.tokens_idx,
            self.valid_idx,
            self.length_idx,
            self.samples_per_slot,
            self.log_prob,
        )

    def get_result_at_slot(self, slot: int) -> SlotData:
        """Returns the token at a given slot.

    Args:
      slot: An integer from [0, n) representing an index into the batch.

    Note: implementations of this method must correctly handle
    microbatches, if microbatches are used.
    """
        # Potentially get multiple beams for given slot.
        start_idx = slot * self.samples_per_slot
        end_idx = (slot + 1) * self.samples_per_slot
        # Mask out any non valid tokens.
        return SlotData(
            tokens=self.data[start_idx:end_idx,
                             self.tokens_idx[0]:self.tokens_idx[1]],
            valid=self.data[start_idx:end_idx,
                            self.valid_idx[0]:self.valid_idx[1]],
            # Only get a 1D representation here
            lengths=self.data[start_idx:end_idx,
                              self.length_idx[0]:self.length_idx[1]][:, 0],
        )

    def get_result_at_slots(self, slots: tuple[int]) -> SlotData:
        """Returns the tokens at given slots.

    Args:
      slots: a tuple of integers from [0, n) representing indices
      into the batch.

    """
        return SlotData(
            tokens=self.data[slots, self.tokens_idx[0]:self.tokens_idx[1]],
            valid=self.data[slots, self.valid_idx[0]:self.valid_idx[1]],
            # Only get a 1D representation here
            lengths=self.data[slots, self.length_idx[0]:self.length_idx[1]][:,
                                                                            0],
            log_prob=self.log_prob[slots, :]
            if self.log_prob is not None else None,
        )
