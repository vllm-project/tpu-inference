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

import pytest
from absl.testing import parameterized

from tpu_inference.layers.common.fused_moe_gmm import _resolve_topk_backend


class ResolveTopkBackendTest(parameterized.TestCase):

    @parameterized.named_parameters(
        ("topk", "topk", "topk", 0.9),
        ("pallas_topk", "pallas_topk", "pallas_topk", 0.9),
        ("approx_topk_default", "approx_topk", "approx_topk", 0.9),
        ("approx_topk_explicit", "approx_topk:recall_target=0.95",
         "approx_topk", 0.95),
    )
    def test_parses(self, backend_str, expected_name, expected_recall):
        name, recall_target = _resolve_topk_backend(backend_str)
        self.assertEqual(name, expected_name)
        self.assertEqual(recall_target, expected_recall)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
