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

from tpu_inference.models.jax.deepseek_v3 import MLAEinsum


class TestMLAEinsumLoadWeights(unittest.TestCase):

    def test_partial_load_returns_loaded_names(self):
        # vLLM's AutoWeightsLoader logs a warning (dumping the full module
        # repr) for every module whose load_weights returns None. MLAEinsum
        # receives its two params (weight, weight_scale_inv) in separate
        # calls, so each call must report the names it loaded to avoid one
        # multi-line warning per decoder layer.
        fake_einsum = MagicMock()
        fake_einsum.loaded = set()
        weight_param = MagicMock()
        scale_param = MagicMock()
        fake_einsum.named_parameters.return_value = [
            ("weight", weight_param),
            ("weight_scale_inv", scale_param),
        ]

        loaded = MLAEinsum.load_weights(fake_einsum, [("weight", MagicMock())])

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded, {"weight"})
        self.assertEqual(fake_einsum.loaded, {"weight"})
        weight_param.weight_loader.assert_called_once()

    def test_load_more_than_two_params_raises(self):
        fake_einsum = MagicMock()
        fake_einsum.loaded = {"weight", "weight_scale_inv"}
        fake_einsum.named_parameters.return_value = []

        with self.assertRaises(ValueError):
            MLAEinsum.load_weights(fake_einsum, [("extra", MagicMock())])


if __name__ == "__main__":
    unittest.main()
