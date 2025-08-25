import functools
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import \
        SchedulerOutput as VllmSchedulerOutput

    from tpu_commons.runner.jax.tpu_jax_runner import TPUModelRunner


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
        self, logits: jax.Array, scheduler_output: "VllmSchedulerOutput"
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        grammar_bitmask = scheduler_output.grammar_bitmask
        assert grammar_bitmask is not None
        num_reqs, _ = logits.shape

        # Reset pre-allocated tensors
        self.runner.grammar_bitmask_cpu.fill(0)
        self.runner.require_structured_out_cpu.fill(0)

        # We receive the structured output bitmask from the scheduler, but the
        # indices of the requests in the batch may not match the indices of
        # the bitmask since the scheduler doesn't know how the tpu runner is
        # ordering the requests in the batch. We need to match the order of
        # bitmask with the order of requests
        struct_out_indices: list[int] = []
        mask_indices: list[int] = []
        for req_id in self.runner.input_batch.req_ids:
            mask_index = scheduler_output.structured_output_request_ids.get(
                req_id)
            if mask_index is None:
                continue
            batch_index = self.runner.input_batch.req_id_to_index[req_id]
            struct_out_indices.append(batch_index)
            mask_indices.append(mask_index)
        self.runner.grammar_bitmask_cpu[struct_out_indices] = grammar_bitmask[
            mask_indices]
        # It's not guaranteed that all requests in this batch require
        # structured output, so create a bool tensor to represent
        # the requests that need structured output.
        self.runner.require_structured_out_cpu[struct_out_indices] = True

        (require_structured_out_cpu, grammar_bitmask_cpu,
         structured_decode_arange) = self.runner._device_array(
             (self.runner.require_structured_out_cpu[:num_reqs],
              self.runner.grammar_bitmask_cpu[:num_reqs],
              self.runner.structured_decode_arange))

        return (require_structured_out_cpu, grammar_bitmask_cpu,
                structured_decode_arange)
