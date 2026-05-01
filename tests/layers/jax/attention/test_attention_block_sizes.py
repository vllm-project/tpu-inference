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
"""Tests for RPA block size env var parsing."""

import unittest
from unittest.mock import patch

from tpu_inference.layers.common.utils import parse_block_sizes


class TestEnvVarBlockSizePassthrough(unittest.TestCase):
    """Tests that env vars are parsed correctly."""

    @patch("tpu_inference.layers.common.utils.envs")
    def test_env_vars_forwarded_as_block_size_args(self, mock_envs):
        """When env vars are set, they should be parsed into the right tuples."""
        mock_envs.RPA_D_BLOCK_SIZES = "1,4096,1,256"
        mock_envs.RPA_P_BLOCK_SIZES = "32,4096,32,256"
        mock_envs.RPA_M_BLOCK_SIZES = "16,2048,16,128"

        d = parse_block_sizes(mock_envs.RPA_D_BLOCK_SIZES, "RPA_D_BLOCK_SIZES")
        p = parse_block_sizes(mock_envs.RPA_P_BLOCK_SIZES, "RPA_P_BLOCK_SIZES")
        m = parse_block_sizes(mock_envs.RPA_M_BLOCK_SIZES, "RPA_M_BLOCK_SIZES")

        self.assertEqual(d, (1, 4096, 1, 256))
        self.assertEqual(p, (32, 4096, 32, 256))
        self.assertEqual(m, (16, 2048, 16, 128))

    @patch("tpu_inference.layers.common.utils.envs")
    def test_none_env_vars_produce_none_block_sizes(self, mock_envs):
        """When env vars are unset, block sizes should be None (use defaults)."""
        mock_envs.RPA_D_BLOCK_SIZES = None
        mock_envs.RPA_P_BLOCK_SIZES = None
        mock_envs.RPA_M_BLOCK_SIZES = None

        self.assertIsNone(
            parse_block_sizes(mock_envs.RPA_D_BLOCK_SIZES,
                              "RPA_D_BLOCK_SIZES"))
        self.assertIsNone(
            parse_block_sizes(mock_envs.RPA_P_BLOCK_SIZES,
                              "RPA_P_BLOCK_SIZES"))
        self.assertIsNone(
            parse_block_sizes(mock_envs.RPA_M_BLOCK_SIZES,
                              "RPA_M_BLOCK_SIZES"))


if __name__ == "__main__":
    unittest.main()
