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
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from vllm.v1.outputs import LogprobsTensors

from unittest import mock
from tpu_inference import envs
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.sampling import (
    PromptLogprobsAsyncData, PromptLogprobsReqSnap,
    _apply_sampling_transforms, _merge_topk_candidates, compute_logprobs,
    compute_prompt_logprobs, gather_logprobs, sample)
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata


class TestSampling:

    def test_distributed_candidates_match_full_vocab_filters(self):
        batch_size = 2
        num_shards = 8
        shard_vocab_size = 256
        vocab_size = num_shards * shard_vocab_size
        logits = jax.random.normal(jax.random.key(7),
                                   (batch_size, vocab_size),
                                   dtype=jnp.float32)
        temperature = jnp.array([1.0, 0.7], dtype=jnp.float32)
        top_p = jnp.array([0.95, 0.8], dtype=jnp.float32)
        metadata = TPUSupportedSamplingMetadata(
            temperature=temperature,
            top_k=jnp.full((batch_size, ), 64, dtype=jnp.int32),
            top_p=top_p,
            do_sampling=True,
            logprobs=False,
        )
        expected = _apply_sampling_transforms(logits, metadata)

        scaled = logits / temperature[:, None]
        sharded = scaled.reshape(batch_size, num_shards, shard_vocab_size)
        local_values, local_ids = jax.lax.top_k(sharded, 128)
        shard_offsets = (jnp.arange(num_shards, dtype=jnp.int32) *
                         shard_vocab_size)
        local_ids += shard_offsets[None, :, None]
        candidate_values = local_values.reshape(batch_size, -1)
        candidate_ids = local_ids.reshape(batch_size, -1)
        filtered_values, filtered_ids, incomplete = (
            _merge_topk_candidates(candidate_values, candidate_ids, top_p))

        actual = jnp.full_like(expected, -1e12)
        actual = actual.at[jnp.arange(batch_size)[:, None],
                           filtered_ids].set(filtered_values)
        assert not np.asarray(incomplete).any()
        np.testing.assert_array_equal(actual, expected)

    def test_distributed_candidates_preserve_topk_boundary_tie(self):
        candidate_values = jnp.linspace(1.0, 0.0, 256,
                                        dtype=jnp.float32)[None, :]
        candidate_values = candidate_values.at[0, 64].set(
            candidate_values[0, 63])
        candidate_ids = jnp.arange(256, dtype=jnp.int32)[None, :]
        filtered, _, incomplete = _merge_topk_candidates(
            candidate_values,
            candidate_ids,
            jnp.array([1.0], dtype=jnp.float32),
        )
        assert not bool(incomplete[0])
        assert int(jnp.sum(filtered[0] > -1e11)) >= 65

    def test_distributed_candidates_detect_truncated_tie_group(self):
        candidate_values = jnp.concatenate(
            (jnp.full((128, ), 10.0, dtype=jnp.float32),
             jnp.arange(0, -128, -1, dtype=jnp.float32)))[None, :]
        candidate_ids = jnp.arange(256, dtype=jnp.int32)[None, :]
        _, _, incomplete = _merge_topk_candidates(
            candidate_values,
            candidate_ids,
            jnp.array([0.95], dtype=jnp.float32),
        )
        assert bool(incomplete[0])

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
            temperature=jnp.full((batch_size, ),
                                 temperature,
                                 dtype=jnp.float32),
            top_k=jnp.full((batch_size, ), top_k, dtype=jnp.int32),
            top_p=jnp.full((batch_size, ), top_p, dtype=jnp.float32),
            _cache_collision_dummy=None,
            do_sampling=do_sampling,
            logprobs=True,
        )

    @staticmethod
    def _get_fake_mesh():
        """Create a fake mesh for testing purposes."""
        devices = jax.devices()
        num_devices = len(devices)
        mesh_shape = (num_devices, )
        axis_names = (ShardingAxisName.ATTN_DATA, )
        device_mesh = mesh_utils.create_device_mesh(mesh_shape, devices)
        return Mesh(device_mesh, axis_names)

    def test_processed_logprobs_with_temperature(self):
        """Temperature scaling should change the logprobs distribution."""
        logits = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)

        raw_logprobs = compute_logprobs(logits)

        metadata = self._make_sampling_metadata(1, temperature=0.5)
        with mock.patch.object(envs, "SAMPLING_MICROBATCH_SIZE", 16):
            _, processed_logits = sample(jax.random.PRNGKey(0),
                                         self._get_fake_mesh(), logits, metadata)
        processed = compute_logprobs(processed_logits)

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
        fake_mesh = self._get_fake_mesh()
        _, processed_logits = sample(jax.random.PRNGKey(0), fake_mesh, logits,
                                     metadata)
        processed = compute_logprobs(processed_logits)

        expected = jnp.log(
            jnp.exp(logits / temperature) /
            jnp.sum(jnp.exp(logits / temperature), axis=-1, keepdims=True))
        assert np.allclose(processed, expected, atol=1e-5)

    def test_processed_logprobs_with_topk(self):
        """After top-k masking, tokens outside top-k should get -inf logprobs."""
        logits = jnp.array([[1.0, 5.0, 3.0, 2.0, 4.0]], dtype=jnp.float32)

        metadata = self._make_sampling_metadata(1, temperature=1.0, top_k=2)
        with mock.patch.object(envs, "SAMPLING_MICROBATCH_SIZE", 16):
            _, processed_logits = sample(jax.random.PRNGKey(0),
                                         self._get_fake_mesh(), logits, metadata)
        processed = compute_logprobs(processed_logits)

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

        metadata = self._make_sampling_metadata(1, temperature=1.0, top_p=0.5)
        with mock.patch.object(envs, "SAMPLING_MICROBATCH_SIZE", 16):
            _, processed_logits = sample(jax.random.PRNGKey(0),
                                         self._get_fake_mesh(), logits, metadata)
        processed = compute_logprobs(processed_logits)

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
        with mock.patch.object(envs, "SAMPLING_MICROBATCH_SIZE", 16):
            _, processed_logits = sample(jax.random.PRNGKey(0),
                                         self._get_fake_mesh(), logits, metadata)
        processed = compute_logprobs(processed_logits)

        assert np.allclose(raw_logprobs, processed, atol=1e-6)

    def test_sampling_transforms_microbatch_preserves_results(self):
        """A batch of 32 should match two independent batches of 16."""
        batch_size = 32
        logits = jnp.arange(batch_size * 64, dtype=jnp.float32).reshape(
            batch_size, 64)
        logits = (logits % 37) / 10.0
        metadata = TPUSupportedSamplingMetadata(
            temperature=jnp.linspace(0.5, 1.5, batch_size),
            top_k=jnp.arange(batch_size, dtype=jnp.int32) % 8 + 1,
            top_p=jnp.linspace(0.6, 0.95, batch_size),
            do_sampling=True,
            logprobs=True,
        )

        with mock.patch.object(envs, "SAMPLING_MICROBATCH_SIZE", 16):
            _, processed_logits = sample(jax.random.PRNGKey(0),
                                         self._get_fake_mesh(), logits, metadata)

        chunk_results = []
        for start in (0, 16):
            chunk_metadata = TPUSupportedSamplingMetadata(
                temperature=metadata.temperature[start:start + 16],
                top_k=metadata.top_k[start:start + 16],
                top_p=metadata.top_p[start:start + 16],
                do_sampling=True,
                logprobs=True,
            )
            _, chunk_logits = sample(jax.random.PRNGKey(0),
                                     self._get_fake_mesh(),
                                     logits[start:start + 16], chunk_metadata)
            chunk_results.append(chunk_logits)

        assert np.array_equal(processed_logits,
                              jnp.concatenate(chunk_results, axis=0))


class TestComputePromptLogprobs:

    def test_compute_prompt_logprobs_success(self):
        from unittest.mock import MagicMock

        # Setup inputs
        full_logits = jnp.array([
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
                                dtype=jnp.float32)
        input_ids = jnp.array([0, 1, 2], dtype=jnp.int32)

        num_prompt_logprobs = {"req1": 2}

        # Mock CachedRequestState and VllmSchedulerOutput
        mock_req_state = MagicMock()
        mock_req_state.num_computed_tokens = 0
        mock_req_state.num_prompt_tokens = 3
        requests = {"req1": mock_req_state}

        mock_scheduler_output = MagicMock()
        mock_scheduler_output.num_scheduled_tokens = {"req1": 2}

        req_ids_dp = {0: ["req1"]}
        dp_size = 1

        res = compute_prompt_logprobs(
            full_logits=full_logits,
            input_ids=input_ids,
            num_prompt_logprobs=num_prompt_logprobs,
            requests=requests,
            scheduler_output=mock_scheduler_output,
            req_ids_dp=req_ids_dp,
            dp_size=dp_size,
            max_logprobs=2,
        )

        assert res is not None
        assert isinstance(res, PromptLogprobsAsyncData)
        assert len(res.req_snaps) == 1
        snap = res.req_snaps[0]
        assert isinstance(snap, PromptLogprobsReqSnap)
        assert snap.req_id == "req1"
        assert snap.num_k == 2
        assert snap.start_idx == 0
        assert snap.num_logits == 2
        assert snap.is_last_chunk is False
