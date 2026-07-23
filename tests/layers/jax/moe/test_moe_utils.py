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
from unittest import mock

import jax

from tpu_inference import envs
from tpu_inference.layers.jax.moe.utils import MoEBackend, select_moe_backend


class TestMoESelector(unittest.TestCase):

    def setUp(self):
        self.key = jax.random.PRNGKey(0)

    def test_select_moe_backend_defaults(self):
        """Test default backend selection (GMM_TP/GMM_EP)."""
        # Ensure all explicit backend flags are False
        with mock.patch.object(envs, 'USE_MOE_EP_KERNEL', False):

            # Case 1: No EP enabled -> Fallback to GMM_TP
            backend_tp = select_moe_backend(use_ep=False)
            self.assertEqual(backend_tp, MoEBackend.GMM_TP)

            # Case 2: EP enabled -> Fallback to GMM_EP
            backend_ep = select_moe_backend(use_ep=True)
            self.assertEqual(backend_ep, MoEBackend.GMM_EP)

    def test_select_moe_backend_priority(self):
        """Test priority logic for backend selection."""

        # Case 1: Fused MoE (Highest priority when USE_MOE_EP_KERNEL=True AND use_ep=True)
        with mock.patch.object(envs, 'USE_MOE_EP_KERNEL', True):
            self.assertEqual(select_moe_backend(use_ep=True),
                             MoEBackend.FUSED_MOE)

        # Case 1.5: Fused MoE Flag is True, but use_ep is False.
        # Should fall through to defaults (GMM_TP) because Fused kernel requires EP.
        with mock.patch.object(envs, 'USE_MOE_EP_KERNEL', True), \
             mock.patch.object(envs, 'USE_DENSE_MOE', False):
            self.assertEqual(select_moe_backend(use_ep=False),
                             MoEBackend.GMM_TP)

        # Case 2: Dense (only when EP is not enabled)
        with mock.patch.object(envs, 'USE_MOE_EP_KERNEL', False), \
             mock.patch.object(envs, 'USE_DENSE_MOE', True):
            self.assertEqual(select_moe_backend(use_ep=True),
                             MoEBackend.GMM_EP)
            self.assertEqual(select_moe_backend(use_ep=False),
                             MoEBackend.DENSE_MAT)

        # Default: GMM_TP
        with mock.patch.object(envs, 'USE_MOE_EP_KERNEL', False), \
             mock.patch.object(envs, 'USE_DENSE_MOE', False):
            self.assertEqual(select_moe_backend(use_ep=True),
                             MoEBackend.GMM_EP)
            self.assertEqual(select_moe_backend(use_ep=False),
                             MoEBackend.GMM_TP)


if __name__ == "__main__":
    unittest.main()
