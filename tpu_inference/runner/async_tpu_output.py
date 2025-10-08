# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
from typing import List, Optional

import jax
import numpy as np
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class AsyncTPUModelRunnerOutput(AsyncModelRunnerOutput):
    """Async wrapper for TPU model runner output.

    This class defers the expensive jax.device_get() operation until
    get_output() is called, allowing the forward pass to complete and
    return immediately while the token transfer happens asynchronously.
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        next_tokens_device: jax.Array,
        invalid_req_indices: List[int],
        executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
        request_seq_lens=None,
        input_batch=None,
        max_model_len=None,
    ):
        """Initialize async TPU output.

        Args:
            model_runner_output: The base model runner output
            next_tokens_device: The device tensor containing next tokens
            invalid_req_indices: Indices of requests that should be discarded
            executor: Thread pool executor for async operations
            request_seq_lens: Request sequence lengths for state updates
            input_batch: Input batch for state updates
            max_model_len: Maximum model length for validation
        """
        self._model_runner_output = model_runner_output
        self._next_tokens_device = next_tokens_device
        self._invalid_req_indices = invalid_req_indices
        self._request_seq_lens = request_seq_lens or []
        self._input_batch = input_batch
        self._max_model_len = max_model_len
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1)

        # Start the async copy operation immediately
        self._copy_future = self._executor.submit(self._copy_to_host)

    def _copy_to_host(self) -> np.ndarray:
        """Perform the blocking device-to-host copy."""
        return np.asarray(jax.device_get(self._next_tokens_device))

    def get_output(self) -> ModelRunnerOutput:
        """Get the ModelRunnerOutput.

        This is a blocking call that waits until the device-to-host copy
        is ready and then processes the sampled token IDs.
        """
        # Wait for the async copy to complete
        try:
            next_tokens_cpu = self._copy_future.result()
        except Exception as e:
            logger.error(f"Failed to copy tokens from device to host: {e}")
            raise

        # Process the tokens (similar to the original implementation)
        num_reqs = len(self._model_runner_output.req_ids)
        selected_token_ids = np.expand_dims(next_tokens_cpu[:num_reqs], 1)
        valid_sampled_token_ids = selected_token_ids.tolist()

        # Mask out tokens that should not be sampled
        for i in self._invalid_req_indices:
            valid_sampled_token_ids[i].clear()

        # Update internal state (crucial for maintaining context between iterations)
        # This follows the exact same logic as the synchronous implementation
        for req_idx, req_state, _ in self._request_seq_lens:
            sampled_ids = valid_sampled_token_ids[req_idx]
            if not sampled_ids:
                continue

            start_idx = self._input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)

            # Validate against max model length (using assert like sync version)
            assert end_idx <= self._max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self._max_model_len}")

            # Update the input batch state
            self._input_batch.token_ids_cpu[req_idx,
                                            start_idx:end_idx] = sampled_ids
            self._input_batch.num_tokens_no_spec[req_idx] = end_idx
            self._input_batch.num_tokens[req_idx] = end_idx

            # Update the request state - this is crucial for context continuity
            req_state.output_token_ids.extend(sampled_ids)

        # Update the output with the processed tokens
        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids

        return output


class AsyncTPUSpecDecodeOutput(AsyncModelRunnerOutput):
    """Async wrapper for TPU speculative decoding output.

    Similar to AsyncTPUModelRunnerOutput but handles speculative decoding
    where token processing is more complex.
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        next_tokens_device: jax.Array,
        rejection_sampler,
        vocab_size: int,
        draft_lengths_cpu: np.ndarray,
        draft_token_ids_shape: tuple,
        invalid_req_indices: List[int],
        executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
        request_seq_lens=None,
        input_batch=None,
        max_model_len=None,
    ):
        self._model_runner_output = model_runner_output
        self._next_tokens_device = next_tokens_device
        self._rejection_sampler = rejection_sampler
        self._vocab_size = vocab_size
        self._draft_lengths_cpu = draft_lengths_cpu
        self._draft_token_ids_shape = draft_token_ids_shape
        self._invalid_req_indices = invalid_req_indices
        self._request_seq_lens = request_seq_lens or []
        self._input_batch = input_batch
        self._max_model_len = max_model_len
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1)

        # Start the async processing
        self._process_future = self._executor.submit(
            self._process_spec_decode_tokens)

    def _process_spec_decode_tokens(self) -> List[List[int]]:
        """Process speculative decoding tokens asynchronously."""
        num_reqs = len(self._model_runner_output.req_ids)
        return self._rejection_sampler.parse_output(
            self._next_tokens_device, self._vocab_size,
            self._draft_lengths_cpu, num_reqs, self._draft_token_ids_shape[0])

    def get_output(self) -> ModelRunnerOutput:
        """Get the ModelRunnerOutput for speculative decoding."""
        try:
            valid_sampled_token_ids = self._process_future.result()
        except Exception as e:
            logger.error(f"Failed to process speculative decode tokens: {e}")
            raise

        # Mask out the sampled tokens that should not be sampled
        # (This is done in the synchronous path after token processing)
        for i in self._invalid_req_indices:
            valid_sampled_token_ids[i].clear()

        # Update internal state (crucial for maintaining context between iterations)
        # This follows the exact same logic as the synchronous implementation
        for req_idx, req_state, _ in self._request_seq_lens:
            sampled_ids = valid_sampled_token_ids[req_idx]
            if not sampled_ids:
                continue

            start_idx = self._input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)

            # Validate against max model length (using assert like sync version)
            assert end_idx <= self._max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self._max_model_len}")

            # Update the input batch state
            self._input_batch.token_ids_cpu[req_idx,
                                            start_idx:end_idx] = sampled_ids
            self._input_batch.num_tokens_no_spec[req_idx] = end_idx
            self._input_batch.num_tokens[req_idx] = end_idx

            # Update the request state - this is crucial for context continuity
            req_state.output_token_ids.extend(sampled_ids)

        # Update the output
        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids

        return output
