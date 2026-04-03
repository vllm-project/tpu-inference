# Copied from vLLM: https://github.com/vllm-project/vllm/blob/02f0c7b/benchmarks/backend_request_func.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file implements the request logic needed for serving benchmark requests.
"""

import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Literal, Optional, Protocol, Union

import aiohttp
import huggingface_hub.constants
from benchmark_core import BenchmarkContext, RequestFuncOutput, SampleRequest
from tqdm.asyncio import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

logger = logging.getLogger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


async def start_stop_profile(base_url: str, action: Literal["start", "stop"]):
    """A start / stop profile utility for vLLM server."""
    api_url = base_url + f"/{action}_profile"

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:

        async with session.post(url=api_url) as response:
            if (code := response.status) != 200:
                logger.warning(f"{action=} profile failed: {code=}")
                return False
            return True


async def async_request_openai_completions(
    ctx: BenchmarkContext,
    request_func_input: SampleRequest,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = ctx.api_url
    assert api_url.endswith("completions"), (
        "OpenAI Completions API URL must end with 'completions'")

    payload = {
        "model": ctx.model_name if ctx.model_name else ctx.model,
        "prompt": request_func_input.prompt,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": request_func_input.expected_output_len,
        "logprobs": ctx.logprobs,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    if ctx.ignore_eos:
        payload["ignore_eos"] = ctx.ignore_eos
    if ctx.extra_body:
        payload.update(ctx.extra_body)

    output = await _openai_fetch(
        ctx=ctx,
        payload=payload,
        extract_from=lambda choices: choices[0].get("text"),
    )

    # XXX: not a good pattern to mutate the field after object creation.
    output.input_request = request_func_input

    if pbar:
        pbar.update(1)

    return output


async def async_request_openai_chat_completions(
    ctx: BenchmarkContext,
    request_func_input: SampleRequest,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:

    # Note: the major difference from openai_completions are
    # - different api_url suffix checking.
    # - "messages" field in payload instead of "prompt".
    # - the extraction logic on "choices" field of response

    api_url = ctx.api_url
    assert api_url.endswith("chat/completions"), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'")

    payload = {
        "model": ctx.model_name if ctx.model_name else ctx.model,
        "messages": request_func_input.messages,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": request_func_input.expected_output_len,
        "logprobs": ctx.logprobs,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    if ctx.ignore_eos:
        payload["ignore_eos"] = ctx.ignore_eos
    if ctx.extra_body:
        payload.update(ctx.extra_body)

    output = await _openai_fetch(
        ctx=ctx,
        payload=payload,
        extract_from=lambda choices: choices[0]["delta"]["content"],
    )

    # XXX: not a good pattern to mutate the field after object creation.
    output.input_request = request_func_input

    if pbar:
        pbar.update(1)

    return output


class _ExtractFunc(Protocol):

    def __call__(self, choices) -> str:
        """Given the "choices" field, extract the message payload."""
        ...


async def _openai_fetch(
    ctx: BenchmarkContext,
    payload: Any,
    extract_from: _ExtractFunc,
) -> RequestFuncOutput:
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:

        output = RequestFuncOutput()

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=ctx.api_url,
                                    json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = extract_from(choices)
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            if usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!")
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv("VLLM_USE_MODELSCOPE", "False").lower() == "true":
        from modelscope import snapshot_download
        from vllm.model_executor.model_loader.weight_utils import get_lock

        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(pretrained_model_name_or_path):
            model_path = snapshot_download(
                model_id=pretrained_model_name_or_path,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
            )

            return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path is not None and not os.path.exists(
            pretrained_model_name_or_path):
        pretrained_model_name_or_path = get_model(
            pretrained_model_name_or_path)
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    if tokenizer_mode == "mistral":
        try:
            from vllm.transformers_utils.tokenizer import MistralTokenizer
        except ImportError as e:
            raise ImportError("MistralTokenizer requires vllm package.\n"
                              "Please install it with `pip install vllm` "
                              "to use mistral tokenizer mode.") from e
        return MistralTokenizer.from_pretrained(
            str(pretrained_model_name_or_path))
    else:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
    "vllm-chat": async_request_openai_chat_completions,
}

OPENAI_COMPATIBLE_BACKENDS = [
    "vllm",
    "vllm-chat",
]
