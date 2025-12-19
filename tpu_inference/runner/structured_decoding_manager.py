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
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp

from tpu_inference.utils import device_array

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput

    from tpu_inference.runner.tpu_runner import TPUModelRunner


class StructuredDecodingManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner

    @functools.partial(jax.jit, static_argnums=(0, ))
    def structured_decode_fn(self, require_struct_decoding: jax.Array,
                             grammar_bitmask: jax.Array, logits: jax.Array,
                             arange: jax.Array) -> jax.Array:
        return jax.lax.cond(
            jnp.any(require_struct_decoding),
            lambda: self._apply_grammar_bitmask_kernel(
                logits, grammar_bitmask, require_struct_decoding, arange),
            lambda: logits)

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _apply_grammar_bitmask_kernel(self, logits: jax.Array,
                                      grammar_bitmask: jax.Array,
                                      require_struct_decoding: jax.Array,
                                      arange: jax.Array) -> jax.Array:

        # Unpack the bitmask for the entire batch at once.
        # grammar_bitmask: (B, N) where B=num_reqs, N=cdiv(vocab_size, 32)
        # arange: (32,)
        # (B, N, 1) and (1, 1, 32) broadcast to (B, N, 32)
        unpacked_bitmask = jnp.right_shift(grammar_bitmask[:, :, None],
                                           arange[None, None, :]) & 1 == 0

        # Reshape to (B, vocab_size) and apply to logits.
        # (B, N * 32) -> (B, vocab_size)
        unpacked_bitmask = unpacked_bitmask.reshape(
            logits.shape[0], -1)[:, :self.runner.vocab_size]

        masked_logits = jnp.where(unpacked_bitmask, -jnp.inf, logits)

        return jnp.where(require_struct_decoding, masked_logits, logits)

    def prepare_structured_decoding_input(
        self, logits: jax.Array, grammar_output: "GrammarOutput"
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        grammar_bitmask = grammar_output.grammar_bitmask
        assert grammar_bitmask is not None
        num_reqs, _ = logits.shape

        # Reset pre-allocated tensors
        self.runner.grammar_bitmask_cpu.fill(0)
        self.runner.require_structured_out_cpu.fill(0)

        sorted_struct_requests = sorted(
            grammar_output.structured_output_request_ids)

        cumulative_mask_idx = 0
        for req_id in sorted_struct_requests:
            if req_id not in self.runner.input_batch.req_id_to_index:
                continue
            batch_index = self.runner.input_batch.req_id_to_index[req_id]
            self.runner.grammar_bitmask_cpu[batch_index] = grammar_bitmask[
                cumulative_mask_idx]
            # It's not guaranteed that all requests in this batch require
            # structured output, so create a bool tensor to represent
            # the requests that need structured output.
            self.runner.require_structured_out_cpu[batch_index] = True
            cumulative_mask_idx += 1

        (require_structured_out_cpu,
         grammar_bitmask_cpu, structured_decode_arange) = device_array(
             self.runner.mesh,
             (self.runner.require_structured_out_cpu[:num_reqs],
              self.runner.grammar_bitmask_cpu[:num_reqs],
              self.runner.structured_decode_arange))

        return (require_structured_out_cpu, grammar_bitmask_cpu,
                structured_decode_arange)
