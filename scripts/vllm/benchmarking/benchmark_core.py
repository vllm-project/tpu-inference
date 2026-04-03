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

from dataclasses import dataclass, field
from typing import Any

from vllm.inputs import MultiModalDataDict
from vllm.lora.request import LoRARequest


@dataclass
class BenchmarkContext:
    api_url: str
    model: str
    model_name: str | None = None
    logprobs: int | None = None
    extra_body: dict | None = None
    ignore_eos: bool = False
    language: str | None = None


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.

    prompt and prompt_len are only used for completions API, and messages
    is only used for chat-completions API.

    The should be used exclusively.
    """

    prompt: str | list[str] | list[int] | list[list[int]] | None = None
    """
    The prompt field for completions API.

    https://developers.openai.com/api/reference/resources/completions/methods/create
    """

    prompt_len: int = 0
    """
    The length of prompt

    Not guarantee to have reasonable value, as we reuse SampleRequest for
    multiple kind of API for now.
    """

    messages: list[Any] | None = None
    """
    The messages field for chat-completions API.

    Must be something that's serialize-able by json lib.
    It's not easy to have schema-correct typing here, please see document.
    https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    """

    expected_output_len: int = 256

    multi_modal_data: MultiModalDataDict | dict | list[dict] | None = None

    lora_request: LoRARequest | None = None

    completion: str | None = None
    """For MMLMDataset, MLPerfDataset"""

    request_id: str | None = None
    """For Random (synthetic), Sonnet"""


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(
        default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    error: str = ""
    input_request: SampleRequest = field(default_factory=SampleRequest)
