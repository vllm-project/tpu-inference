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
from unittest.mock import MagicMock, patch

from tpu_inference.distributed import jax_parallel_state


class TestJaxParallelState(unittest.TestCase):

    def setUp(self):
        jax_parallel_state._PP = jax_parallel_state.GroupCoordinator(0, 2)

    @patch("tpu_inference.distributed.jax_parallel_state.transfer")
    def test_init_pp_distributed_environment_ipv4(self, mock_transfer):
        mock_device = MagicMock()
        mock_transfer_server = MagicMock()
        mock_transfer.start_transfer_server.return_value = mock_transfer_server

        jax_parallel_state.init_pp_distributed_environment(ip="127.0.0.1",
                                                           rank=0,
                                                           world_size=2,
                                                           device=mock_device,
                                                           need_pp=True)

        mock_transfer.start_transfer_server.assert_called_once_with(
            mock_device.client, "127.0.0.1:5000",
            ["127.0.0.1:0", "127.0.0.1:0"])
        pp_group = jax_parallel_state.get_pp_group()
        self.assertEqual(pp_group.transfer_server, mock_transfer_server)

    @patch("tpu_inference.distributed.jax_parallel_state.transfer")
    def test_init_pp_distributed_environment_ipv6(self, mock_transfer):
        mock_device = MagicMock()
        mock_transfer_server = MagicMock()
        mock_transfer.start_transfer_server.return_value = mock_transfer_server

        jax_parallel_state.init_pp_distributed_environment(ip="2001:db8::1",
                                                           rank=1,
                                                           world_size=2,
                                                           device=mock_device,
                                                           need_pp=True)

        mock_transfer.start_transfer_server.assert_called_once_with(
            mock_device.client,
            "[2001:db8::1]:5001",
            ["[2001:db8::1]:0", "[2001:db8::1]:0"],
        )
        pp_group = jax_parallel_state.get_pp_group()
        self.assertEqual(pp_group.transfer_server, mock_transfer_server)

    def test_connect_ipv4(self):
        mock_transfer_server = MagicMock()
        mock_conn = MagicMock()
        mock_transfer_server.connect.return_value = mock_conn

        pp_group = jax_parallel_state.get_pp_group()
        pp_group.transfer_server = mock_transfer_server

        jax_parallel_state.connect(prev_ip="127.0.0.1", prev_rank=0)

        mock_transfer_server.connect.assert_called_once_with("127.0.0.1:5000")
        self.assertEqual(pp_group.connection, mock_conn)

    def test_connect_ipv6(self):
        mock_transfer_server = MagicMock()
        mock_conn = MagicMock()
        mock_transfer_server.connect.return_value = mock_conn

        pp_group = jax_parallel_state.get_pp_group()
        pp_group.transfer_server = mock_transfer_server

        jax_parallel_state.connect(prev_ip="2001:db8::1", prev_rank=0)

        mock_transfer_server.connect.assert_called_once_with(
            "[2001:db8::1]:5000")
        self.assertEqual(pp_group.connection, mock_conn)


if __name__ == "__main__":
    unittest.main()
