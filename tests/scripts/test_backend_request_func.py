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

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web

_BENCHMARKING_DIR = (Path(__file__).parents[2] / "scripts" / "vllm" /
                     "benchmarking")
sys.path.insert(0, str(_BENCHMARKING_DIR))

from backend_request_func import (  # noqa: E402
    async_request_openai_chat_completions, async_request_openai_completions)
from benchmark_core import BenchmarkContext, SampleRequest  # noqa: E402


class StreamingRequestTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.events = []
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._stream)
        app.router.add_post("/v1/completions", self._stream)
        runner = web.AppRunner(app)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}/v1"

    async def _stream(self, request):
        await request.json()
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in self.events:
            await response.write(f"data: {json.dumps(event)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def test_chat_control_chunks_do_not_count_as_tokens(self):
        for initial_delta in ({
                "role": "assistant"
        }, {
                "role": "assistant",
                "content": ""
        }, {
                "content": None
        }):
            with self.subTest(initial_delta=initial_delta):
                self.events = [
                    {
                        "choices": [{
                            "delta": initial_delta
                        }]
                    },
                    {
                        "choices": [{
                            "delta": {
                                "content": "Hello"
                            }
                        }]
                    },
                    {
                        "choices": [{
                            "delta": {
                                "content": " world"
                            }
                        }]
                    },
                    {
                        "choices": [{
                            "delta": {},
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "completion_tokens": 2,
                            "prompt_tokens": 4
                        },
                    },
                ]
                ctx = BenchmarkContext(
                    api_url=f"{self.base_url}/chat/completions", model="test")
                request = SampleRequest(messages=[{
                    "role": "user",
                    "content": "Hi"
                }])
                with patch("backend_request_func.time.perf_counter",
                           side_effect=[0.0, 1.0, 1.0, 2.0]):
                    output = await async_request_openai_chat_completions(
                        ctx, request)

                self.assertTrue(output.success, output.error)
                self.assertEqual(output.generated_text, "Hello world")
                self.assertEqual(output.ttft, 1.0)
                self.assertEqual(output.itl, [1.0])
                self.assertEqual(output.latency, 2.0)
                self.assertEqual(output.output_tokens, 2)
                self.assertEqual(output.prompt_tokens, 4)

    async def test_chat_without_content_is_not_successful(self):
        self.events = [
            {
                "choices": [{
                    "delta": {
                        "role": "assistant",
                        "content": ""
                    }
                }]
            },
            {
                "choices": [{
                    "delta": {},
                    "finish_reason": "stop"
                }]
            },
        ]
        output = await async_request_openai_chat_completions(
            BenchmarkContext(api_url=f"{self.base_url}/chat/completions",
                             model="test"), SampleRequest(messages=[]))
        self.assertFalse(output.success)
        self.assertIn("Never received a valid chunk", output.error)
        self.assertEqual(output.itl, [])

    async def test_completions_preserve_empty_special_token_chunks(self):
        self.events = [
            {
                "choices": [{
                    "text": ""
                }]
            },
            {
                "choices": [{
                    "text": "Hello"
                }]
            },
            {
                "choices": [],
                "usage": {
                    "completion_tokens": 2
                }
            },
        ]
        with patch("backend_request_func.time.perf_counter",
                   side_effect=[0.0, 1.0, 1.0, 2.0]):
            output = await async_request_openai_completions(
                BenchmarkContext(api_url=f"{self.base_url}/completions",
                                 model="test"), SampleRequest(prompt="Hi"))
        self.assertTrue(output.success, output.error)
        self.assertEqual(output.generated_text, "Hello")
        self.assertEqual(output.ttft, 1.0)
        self.assertEqual(output.itl, [1.0])
        self.assertEqual(output.output_tokens, 2)


if __name__ == "__main__":
    unittest.main()
