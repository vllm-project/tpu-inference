# Copyright 2026 Google LLC
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

import unittest
from unittest.mock import MagicMock

from tpu_inference.layers.jax.pp_utils import PPMissingLayer


class TestPPMissingLayer(unittest.TestCase):

    def test_load_weights_returns_set_and_consumes_iterator(self):
        # vLLM's AutoWeightsLoader logs a per-layer warning (with the full
        # module repr) whenever a module's load_weights returns None. The
        # placeholder layer must return a set so a pipeline-parallel load
        # of a many-layer model does not spam one warning per missing layer.
        layer = PPMissingLayer()
        weights = iter([
            ("layers.0.mlp.gate_proj.weight", MagicMock()),
            ("layers.0.mlp.up_proj.weight", MagicMock()),
        ])

        loaded = layer.load_weights(weights)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded, set())
        # The weights iterator must be fully drained.
        self.assertEqual(list(weights), [])

    def test_call_passes_through_first_arg(self):
        layer = PPMissingLayer()
        sentinel = object()
        self.assertIs(layer(sentinel, "other"), sentinel)
        self.assertIs(layer(hidden_states=sentinel), sentinel)


if __name__ == "__main__":
    unittest.main()
