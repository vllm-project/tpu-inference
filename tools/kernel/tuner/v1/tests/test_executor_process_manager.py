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

import io
import json
import subprocess
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from tools.kernel.tuner.v1.common.tuner_datatypes import (RunConfig,
                                                          TuningStatus)
from tools.kernel.tuner.v1.executor_process_manager import \
    ExecutorProcessManager


@dataclass(frozen=True)
class MockParams:
    size: int = 1


class ExecutorProcessManagerTest(unittest.TestCase):

    def test_ignores_unprefixed_json_output(self):
        response = {
            "status": "SUCCESS",
            "avg_latency_ns": 1234,
            "total_latency_ns": 6170,
        }
        log_values = [{
            "message": "compiling"
        }, {
            "status": "ready"
        },
                      dict(response, avg_latency_ns=0), [], None, 123,
                      "log message"]
        for log_value in log_values:
            with self.subTest(log_value=log_value):
                manager = ExecutorProcessManager("unused", RunConfig())
                manager._proc = SimpleNamespace(stdout=io.StringIO(
                    json.dumps(log_value) + "\n__JSON__" +
                    json.dumps(response) + "\n"))
                self.assertEqual(manager._read_json_response(timeout=5),
                                 response)

    def test_ignores_blank_text_and_invalid_json_lines(self):
        manager = ExecutorProcessManager("unused", RunConfig())
        manager._proc = SimpleNamespace(stdout=io.StringIO(
            '\ncompiler log\n__JSON__invalid\n__JSON__{"status":"ready"}\n'))
        self.assertEqual(manager._read_json_response(timeout=5),
                         {"status": "ready"})

    def test_unprefixed_json_followed_by_eof_is_not_a_response(self):
        manager = ExecutorProcessManager("unused", RunConfig())
        manager._proc = SimpleNamespace(
            stdout=io.StringIO('{"status":"ready"}\n'))
        with self.assertRaises(BrokenPipeError):
            manager._read_json_response(timeout=5)

    def test_json_logs_do_not_disrupt_persistent_executor(self):
        manager = ExecutorProcessManager("unused", RunConfig())
        manager._proc = subprocess.Popen(
            [
                sys.executable, "-u", "-c", """
import json
import sys

print(json.dumps({"message": "initializing"}))
print('__JSON__' + json.dumps({"status": "ready"}))
for line in sys.stdin:
    request = json.loads(line)
    if request["cmd"] == "shutdown":
        break
    print(json.dumps({"message": "running"}))
    print('__JSON__' + json.dumps({
        "status": "SUCCESS",
        "avg_latency_ns": 1234,
        "total_latency_ns": 1234 * request["iters"],
    }))
"""
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        proc = manager._proc
        self.addCleanup(proc.stderr.close)
        self.addCleanup(proc.stdout.close)
        self.addCleanup(manager.shutdown)

        self.assertEqual(manager._read_json_response(timeout=5),
                         {"status": "ready"})
        for iters in (1, 5):
            self.assertEqual(
                manager.execute_run(MockParams(),
                                    MockParams(),
                                    iters=iters,
                                    timeout_seconds=5),
                (TuningStatus.SUCCESS, 1234, 1234 * iters))
        self.assertEqual(manager.restart_count, 0)


if __name__ == "__main__":
    unittest.main()
