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

# /home/pooyam/tpu_inference/tests/models/jax/layers/test_sampling.py
import jax.numpy as jnp
import numpy as np
from vllm.v1.outputs import LogprobsTensors

from tpu_inference.layers.jax.sample.sampling import (compute_logprobs,
                                                      compute_processed_logprobs,
                                                      gather_logprobs)
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata


class TestSampling:

    def test_compute_logprobs(self):
        logits = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
                           dtype=jnp.float32)
        logprobs = compute_logprobs(logits)

        # Expected values computed with scipy.special.log_softmax
        expected_logprobs = np.array(
            [
                [-2.40760596, -1.40760596, -0.40760596],
                [-0.40760596, -1.40760596, -2.40760596],
            ],
            dtype=np.float32,
        )
        assert np.allclose(logprobs, expected_logprobs, atol=1e-6)

    def test_gather_logprobs(self):
        logprobs = jnp.array(
            [
                [-2.40760596, -1.40760596, -0.40760596, -3.40760596],
                [-0.40760596, -1.40760596, -2.40760596, -3.40760596],
            ],
            dtype=jnp.float32,
        )
        token_ids = jnp.array([2, 0], dtype=jnp.int32)
        num_logprobs = 2

        result: LogprobsTensors = gather_logprobs(logprobs, token_ids,
                                                  num_logprobs)

        # check indices
        expected_indices = np.array(
            [
                [2, 2, 1],  # token id 2, top-k are 2, 1
                [0, 0, 1],  # token id 0, top-k are 0, 1
            ],
            dtype=np.int32,
        )
        assert np.array_equal(result.logprob_token_ids, expected_indices)

        # check logprobs
        expected_logprobs_values = np.array(
            [
                [-0.40760596, -0.40760596, -1.40760596],
                [-0.40760596, -0.40760596, -1.40760596],
            ],
            dtype=np.float32,
        )
        assert np.allclose(result.logprobs,
                           expected_logprobs_values,
                           atol=1e-6)

        # check ranks
        expected_ranks = np.array([1, 1], dtype=np.int32)
        assert np.array_equal(result.selected_token_ranks, expected_ranks)

    def test_gather_logprobs_with_ties(self):
        logprobs = jnp.array(
            [
                [-1.0, -1.0, -2.0, -2.0],
            ],
            dtype=jnp.float32,
        )
        token_ids = jnp.array([1], dtype=jnp.int32)
        num_logprobs = 3

        result: LogprobsTensors = gather_logprobs(logprobs, token_ids,
                                                  num_logprobs)

        # check logprobs
        expected_logprobs_values = np.array(
            [
                [-1.0, -1.0, -1.0, -2.0],
            ],
            dtype=np.float32,
        )
        assert np.allclose(result.logprobs,
                           expected_logprobs_values,
                           atol=1e-6)

        # check ranks
        # rank of token 1 is 2 because there are 2 values >= -1.0
        expected_ranks = np.array([2], dtype=np.int32)
        assert np.array_equal(result.selected_token_ranks, expected_ranks)

        # check indices
        # The order of tied elements is not guaranteed.
        # token id is 1. top-k indices are a permutation of {0, 1, 2} or {0, 1, 3}.
        assert result.logprob_token_ids[0, 0] == 1
        top_k_indices = sorted(result.logprob_token_ids[0, 1:].tolist())
        assert top_k_indices == [0, 1, 2] or top_k_indices == [0, 1, 3]


class TestProcessedLogprobs:
    """Tests for the processed_logprobs mode (logprobs computed after
    temperature / top-k / top-p transforms)."""

    @staticmethod
    def _make_sampling_metadata(
        batch_size,
        temperature=0.7,
        top_k=0,
        top_p=1.0,
        do_sampling=True,
    ):
        """Helper to build a TPUSupportedSamplingMetadata for testing."""
        return TPUSupportedSamplingMetadata(
            temperature=jnp.full((batch_size, ), temperature,
                                 dtype=jnp.float32),
            top_k=jnp.full((batch_size, ), top_k, dtype=jnp.int32),
            top_p=jnp.full((batch_size, ), top_p, dtype=jnp.float32),
            _cache_collision_dummy=None,
            do_sampling=do_sampling,
            logprobs=True,
        )

    def test_processed_logprobs_with_temperature(self):
        """Temperature scaling should change the logprobs distribution."""
        logits = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)

        raw_logprobs = compute_logprobs(logits)

        metadata = self._make_sampling_metadata(1, temperature=0.5)
        processed = compute_processed_logprobs(logits, metadata)

        # With temperature < 1, processed logprobs should be more peaked
        # (higher max, lower others) compared to raw logprobs.
        assert not np.allclose(raw_logprobs, processed, atol=1e-4)
        # The argmax should still be the same token.
        assert np.argmax(processed[0]) == np.argmax(raw_logprobs[0])
        # The max logprob should be closer to 0 (more confident).
        assert float(jnp.max(processed[0])) > float(jnp.max(raw_logprobs[0]))

    def test_processed_logprobs_matches_manual_temperature(self):
        """Verify processed_logprobs produces the same result as manually
        dividing by temperature then computing log_softmax."""
        logits = jnp.array([[1.0, 2.0, 3.0, 0.5]], dtype=jnp.float32)
        temperature = 0.8

        metadata = self._make_sampling_metadata(1, temperature=temperature)
        processed = compute_processed_logprobs(logits, metadata)

        expected = jnp.log(
            jnp.exp(logits / temperature) /
            jnp.sum(jnp.exp(logits / temperature), axis=-1, keepdims=True))
        assert np.allclose(processed, expected, atol=1e-5)

    def test_processed_logprobs_with_topk(self):
        """After top-k masking, tokens outside top-k should get -inf logprobs."""
        logits = jnp.array([[1.0, 5.0, 3.0, 2.0, 4.0]], dtype=jnp.float32)

        metadata = self._make_sampling_metadata(
            1, temperature=1.0, top_k=2)
        processed = compute_processed_logprobs(logits, metadata)

        # Top-2 tokens are indices 1 (5.0) and 4 (4.0).
        # After masking, only those two should have non-tiny logprobs.
        processed_np = np.array(processed[0])
        top2_indices = set(np.argsort(processed_np)[-2:])
        assert top2_indices == {1, 4}
        # Masked tokens should have very negative logprobs.
        for i in range(5):
            if i not in top2_indices:
                assert processed_np[i] < -10.0

    def test_processed_logprobs_with_topp(self):
        """After top-p filtering, low-probability tokens should be masked."""
        # Make logits where one token dominates.
        logits = jnp.array([[10.0, 1.0, 0.0, -1.0]], dtype=jnp.float32)

        metadata = self._make_sampling_metadata(
            1, temperature=1.0, top_p=0.5)
        processed = compute_processed_logprobs(logits, metadata)

        # Token 0 has very high probability and should remain.
        processed_np = np.array(processed[0])
        assert processed_np[0] > -0.1  # close to 0 = probability close to 1

    def test_processed_logprobs_greedy_fallback(self):
        """For greedy requests (temperature < eps), processed logprobs should
        match raw logprobs."""
        logits = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)

        raw_logprobs = compute_logprobs(logits)

        # Temperature < _SAMPLING_EPS (1e-5)
        metadata = self._make_sampling_metadata(1, temperature=1e-7)
        processed = compute_processed_logprobs(logits, metadata)

        assert np.allclose(raw_logprobs, processed, atol=1e-6)
