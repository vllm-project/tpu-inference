# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Unit tests for tpu_inference.rl.raiden_worker_sync."""

import unittest
import unittest.mock
from types import SimpleNamespace

from tpu_inference.rl import raiden_worker_sync as rws


class MaxTextForCausalLM:
    """Named to match the real class by `__name__`; see `is_maxtext_model`."""


class TestIsMaxtextModel(unittest.TestCase):

    def test_matches_by_class_name(self):
        self.assertTrue(rws.is_maxtext_model(MaxTextForCausalLM()))

    def test_none_model(self):
        self.assertFalse(rws.is_maxtext_model(None))

    def test_other_model(self):
        self.assertFalse(rws.is_maxtext_model(object()))


class TestExtractWeightState(unittest.TestCase):

    def test_maxtext_unwraps_model_key(self):
        state = {"model": {"params": 1}}
        result = rws.extract_weight_state(state, MaxTextForCausalLM())
        self.assertEqual(result, {"base": {"params": 1}})

    def test_maxtext_missing_model_key_falls_back_to_state(self):
        state = {"other": {"params": 1}}
        result = rws.extract_weight_state(state, MaxTextForCausalLM())
        self.assertIs(result, state)

    def test_non_maxtext_returns_state_as_is(self):
        state = {"params": 1}
        result = rws.extract_weight_state(state, object())
        self.assertIs(result, state)

    def test_no_state_no_model_returns_none(self):
        self.assertIsNone(rws.extract_weight_state(None, None))

    def test_maxtext_model_without_state_and_no_inner_model_returns_none(self):
        model = MaxTextForCausalLM()  # no `.model` attribute
        self.assertIsNone(rws.extract_weight_state(None, model))


class TestFlattenWeights(unittest.TestCase):

    def test_flattens_leaves_with_shape_and_dtype(self):
        leaf = SimpleNamespace(shape=(2, 2), dtype="float32")
        names, arrays = rws.flatten_weights({"w": leaf})
        self.assertEqual(len(names), 1)
        self.assertIs(arrays[0], leaf)

    def test_skips_leaves_without_shape_or_dtype(self):
        names, arrays = rws.flatten_weights({"w": 3})
        self.assertEqual(names, [])
        self.assertEqual(arrays, [])


class TestAxisName(unittest.TestCase):

    def test_none(self):
        self.assertEqual(rws._axis_name(None), "")

    def test_str(self):
        self.assertEqual(rws._axis_name("fsdp"), "fsdp")

    def test_tuple(self):
        self.assertEqual(rws._axis_name(("fsdp", "tp")), "fsdp,tp")


class TestIsWeight(unittest.TestCase):
    """`_filter_bindable` drops KV-cache leaves; the trainer has no counterpart
    for them and the controller pairs by name."""

    def test_keeps_ordinary_weights(self):
        self.assertTrue(
            rws._is_weight("['base']['decoder']['layers_0']['mlp']['kernel']"))

    def test_drops_cache_leaves(self):
        self.assertFalse(
            rws._is_weight(
                "['base']['decoder']['layers_0']['attention']['cache']"
                "['cached_prefill_key']"))

    def test_filter_bindable_drops_cache_without_touching_weights(self):
        leaf = SimpleNamespace(shape=(2, ), dtype="float32")
        names = ["['w']", "['attention']['cache']['cached_prefill_key']"]
        kept_names, kept_arrays = rws._filter_bindable(names, [leaf, leaf])
        # The cache leaf is dropped before `_bindable` is consulted, so it goes
        # even though it is an ordinary float array.
        self.assertEqual(kept_names, [])
        self.assertEqual(kept_arrays, [])


class TestWaitUntilSettled(unittest.TestCase):

    def _sync_with_digests(self, digests):
        sync = rws.RaidenWorkerSync("rollout")
        sync.arrays = [object()]
        seq = iter(digests)
        last = [digests[-1]]

        def fake_l1_norm(_arrays):
            try:
                last[0] = next(seq)
            except StopIteration:
                pass
            return last[0]

        return sync, fake_l1_norm

    def test_returns_once_the_digest_stops_changing(self):
        sync, fake = self._sync_with_digests([1.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        with unittest.mock.patch.object(rws, "_l1_norm", fake), \
             unittest.mock.patch.object(rws.time, "sleep", lambda _s: None):
            sync._wait_until_settled(timeout_s=5.0, interval_s=0.0)

    def test_returns_early_when_the_transfer_has_not_started(self):
        """Known limitation: a digest that has not moved yet reads as settled.

        `stable` counts *unchanged* reads and there is no "saw it change at
        least once" precondition, so if h2d() returns before Raiden has DMA'd
        any bytes this declares success on the pre-sync weights.
        """
        sync, fake = self._sync_with_digests([7.0] * 6)
        with unittest.mock.patch.object(rws, "_l1_norm", fake), \
             unittest.mock.patch.object(rws.time, "sleep", lambda _s: None):
            sync._wait_until_settled(timeout_s=5.0, interval_s=0.0)


class TestRaidenWorkerSyncMetadataDict(unittest.TestCase):

    def test_raises_without_a_sharded_array(self):
        sync = rws.RaidenWorkerSync("rollout")
        sync.names = ["w"]
        sync.arrays = [
            SimpleNamespace(shape=(2, ),
                            dtype=SimpleNamespace(itemsize=4),
                            sharding=None,
                            ndim=1)
        ]
        with self.assertRaises(RuntimeError):
            sync.metadata_dict()

    def test_bound_is_false_until_bind(self):
        sync = rws.RaidenWorkerSync("rollout")
        self.assertFalse(sync.bound)
        sync.names = ["w"]
        self.assertTrue(sync.bound)


if __name__ == "__main__":
    unittest.main()
