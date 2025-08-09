# /home/pooyam/tpu_commons/tests/models/jax/layers/test_sampling.py
import jax.numpy as jnp
import numpy as np
from vllm.v1.outputs import LogprobsTensors

from tpu_commons.models.jax.layers.sample.sampling import (compute_logprobs,
                                                           gather_logprobs)


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
