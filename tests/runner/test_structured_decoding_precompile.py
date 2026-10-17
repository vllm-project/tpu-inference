# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

from unittest.mock import MagicMock, patch

from tpu_inference.runner.compilation_manager import CompilationManager


def test_skip_structured_decoding_precompile():
    manager = CompilationManager.__new__(CompilationManager)
    manager.runner = MagicMock()

    with patch(
            "tpu_inference.runner.compilation_manager.envs."
            "SKIP_STRUCTURED_DECODING_PRECOMPILE", True):
        manager._precompile_structured_decoding()

    assert manager.runner.mock_calls == []
