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

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.sampling import (
    VOCAB_SHARDED_SAMPLING_NUM_CANDIDATES, PromptLogprobsAsyncData,
    PromptLogprobsReqSnap, _apply_sampling_transforms,
    _can_sample_vocab_sharded, compute_logprobs, compute_prompt_logprobs,
    gather_logprobs, sample)
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
        _, processed_logits = sample(jax.random.PRNGKey(0),
                                     self._get_fake_mesh(), logits, metadata)
        processed = compute_logprobs(processed_logits)

        assert np.allclose(raw_logprobs, processed, atol=1e-6)


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


class TestVocabShardedSampling:
    """USE_VOCAB_SHARDED_SAMPLING: sampling over per-shard candidates.

    The mesh shards the vocab over every local device, so on a multi-device
    host (or `--xla_force_host_platform_device_count`) the candidates really
    are merged across shards; the assertions hold for any device count.
    """

    @staticmethod
    def _mesh():
        devices = np.array(jax.devices())
        return Mesh(devices.reshape(1, -1), ("data", ShardingAxisName.MODEL))

    @staticmethod
    def _metadata(temperature, top_k, top_p):
        return TPUSupportedSamplingMetadata(
            temperature=jnp.asarray(temperature, dtype=jnp.float32),
            top_k=jnp.asarray(top_k, dtype=jnp.int32),
            top_p=jnp.asarray(top_p, dtype=jnp.float32),
            _cache_collision_dummy=None,
            do_sampling=True,
            logprobs=False,
        )

    def _sample(self, monkeypatch, enabled, rng, logits, metadata):
        monkeypatch.setenv("USE_VOCAB_SHARDED_SAMPLING",
                           "1" if enabled else "0")
        # The env var is read at trace time; do not reuse a trace made with
        # the other setting.
        sample.clear_cache()
        tokens, _ = sample(rng, self._mesh(), logits, metadata)
        return np.asarray(tokens)

    @staticmethod
    def _allowed(logits, metadata):
        processed = _apply_sampling_transforms(logits, metadata)
        return np.asarray(processed > -1e11)

    def test_can_sample_vocab_sharded(self):
        cap = VOCAB_SHARDED_SAMPLING_NUM_CANDIDATES
        # Real rows with 0 < top_k <= cap and top_p > 0, plus padded slots
        # (DEFAULT_SAMPLING_PARAMS: temperature -1, top_k 0) which are greedy.
        ok = self._metadata([0.7, 1.0, -1.0, -1.0], [50, cap, 0, 0],
                            [0.95, 1.0, 1.0, 1.0])
        assert bool(_can_sample_vocab_sharded(ok))
        # A real row without a top-k filter, or above the candidate budget,
        # forces the replicated path.
        no_top_k = self._metadata([0.7, 0.7], [50, 0], [1.0, 1.0])
        assert not bool(_can_sample_vocab_sharded(no_top_k))
        too_many = self._metadata([0.7, 0.7], [50, cap + 1], [1.0, 1.0])
        assert not bool(_can_sample_vocab_sharded(too_many))
        # Greedy rows are exact whatever their top_k.
        greedy = self._metadata([0.0, 0.0], [0, cap + 1], [1.0, 1.0])
        assert bool(_can_sample_vocab_sharded(greedy))

    def test_greedy_rows_match_argmax(self, monkeypatch):
        logits = jax.random.normal(jax.random.PRNGKey(1), (8, 1024))
        # Rows 0-3 greedy (incl. padded-slot params), rows 4-7 sampled.
        metadata = self._metadata([0.0, -1.0, 0.0, -1.0] + [0.7] * 4,
                                  [0, 0, 5, 0] + [20] * 4,
                                  [1.0, 1.0, 0.5, 1.0] + [0.9] * 4)
        tokens = self._sample(monkeypatch, True, jax.random.PRNGKey(0), logits,
                              metadata)
        np.testing.assert_array_equal(
            tokens[:4],
            np.asarray(jnp.argmax(logits, axis=-1))[:4])

    def test_greedy_ties_take_lowest_id(self, monkeypatch):
        logits = jnp.zeros((2, 1024)).at[0, 7].set(3.0).at[0, 900].set(3.0)
        logits = logits.at[1, 1023].set(2.0).at[1, 500].set(2.0)
        metadata = self._metadata([0.0, 0.0], [0, 0], [1.0, 1.0])
        tokens = self._sample(monkeypatch, True, jax.random.PRNGKey(0), logits,
                              metadata)
        np.testing.assert_array_equal(tokens, [7, 500])

    def test_sampled_tokens_stay_in_top_k_top_p_set(self, monkeypatch):
        logits = 3.0 * jax.random.normal(jax.random.PRNGKey(2), (64, 1024))
        metadata = self._metadata([0.7] * 64, [20] * 64, [0.9] * 64)
        allowed = self._allowed(logits, metadata)
        for seed in range(4):
            tokens = self._sample(monkeypatch, True, jax.random.PRNGKey(seed),
                                  logits, metadata)
            assert allowed[np.arange(64), tokens].all()

    def test_distribution_matches_replicated_path(self, monkeypatch):
        num_rows, vocab = 8192, 256
        row = 2.0 * jax.random.normal(jax.random.PRNGKey(3), (vocab, ))
        logits = jnp.broadcast_to(row, (num_rows, vocab))
        metadata = self._metadata([0.8] * num_rows, [6] * num_rows,
                                  [0.95] * num_rows)
        reference = np.asarray(
            jax.nn.softmax(_apply_sampling_transforms(logits, metadata)[0]))
        tokens = self._sample(monkeypatch, True, jax.random.PRNGKey(0), logits,
                              metadata)
        empirical = np.bincount(tokens, minlength=vocab) / num_rows
        # Every draw is inside the kept set and the frequencies match the
        # reference probabilities to well within binomial noise.
        assert empirical[reference == 0].sum() == 0
        tolerance = 5 * np.sqrt(reference * (1 - reference) / num_rows)
        assert (np.abs(empirical - reference) <= tolerance + 1e-3).all()

    def test_falls_back_when_a_row_exceeds_the_candidate_budget(
            self, monkeypatch):
        logits = jax.random.normal(jax.random.PRNGKey(4), (8, 1024))
        metadata = self._metadata([0.7] * 8, [50] * 7 + [2000], [0.95] * 8)
        rng = jax.random.PRNGKey(5)
        sharded = self._sample(monkeypatch, True, rng, logits, metadata)
        replicated = self._sample(monkeypatch, False, rng, logits, metadata)
        # The fallback is the replicated path itself, same key, same draw.
        np.testing.assert_array_equal(sharded, replicated)

    def test_env_off_keeps_the_replicated_path(self, monkeypatch):
        logits = jax.random.normal(jax.random.PRNGKey(6), (4, 1024))
        metadata = self._metadata([0.7] * 4, [50] * 4, [0.95] * 4)
        rng = jax.random.PRNGKey(7)
        tokens = self._sample(monkeypatch, False, rng, logits, metadata)
        allowed = self._allowed(logits, metadata)
        assert allowed[np.arange(4), tokens].all()
