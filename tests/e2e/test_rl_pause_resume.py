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

# End-to-end tests for the pause/resume window an async RL trainer uses to
# swap policy weights, driven through AsyncLLM.
#
# The loop under test: batches k and k+1 are in flight, k finishes, the engine
# is paused, weights are synced, the engine resumes. Batch k+1 legitimately
# spans two policies; batch k+2 must not reuse KV computed under the old one.
#
# What each test pins down:
#   1. mode="keep" freezes in-flight work without corrupting it -- the tokens
#      produced across a pause match an uninterrupted run exactly.
#   2. Requests submitted during a pause are queued, not aborted.
#   3. clear_cache=True completes instead of failing on a missing worker RPC.
#
# cache_salt behaviour is deliberately not retested here: the block hasher and
# block pool are vLLM code this repo reuses unmodified, and upstream covers it
# in tests/v1/core/test_prefix_caching.py.

from __future__ import annotations

import asyncio

import pytest
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_MODEL_LEN = 2048
MAX_NUM_SEQS = 32

# Long enough that the pause lands mid-generation rather than after the
# request has already finished.
GEN_TOKENS = 64
PAUSE_AFTER_TOKENS = 8
# How long to hold the pause while asserting nothing advances.
PAUSE_HOLD_S = 3.0

PROMPT = ("Count from one to forty, writing each number as a word, "
          "separated by commas.")


@pytest.fixture(scope="module")
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def engine(loop):
    """One AsyncLLM for the module; each test drives it on the shared loop."""

    async def _build() -> AsyncLLM:
        return AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=MODEL_NAME,
                max_model_len=MAX_MODEL_LEN,
                max_num_seqs=MAX_NUM_SEQS,
                enable_prefix_caching=True,
            ))

    eng = loop.run_until_complete(_build())
    yield eng
    eng.shutdown()


async def _collect(engine: AsyncLLM,
                   request_id: str,
                   prompt: str = PROMPT,
                   max_tokens: int = GEN_TOKENS):
    """Runs one request to completion, returning (token_ids, num_cached)."""
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    token_ids: list[int] = []
    num_cached = None
    async for out in engine.generate(prompt, params, request_id):
        token_ids = list(out.outputs[0].token_ids)
        num_cached = out.num_cached_tokens
    return token_ids, num_cached


class TestPauseKeepPreservesGeneration:
    """mode="keep" must freeze in-flight work, not drop or corrupt it."""

    def test_generation_across_pause_matches_uninterrupted_run(
            self, loop, engine):

        async def scenario():
            baseline, _ = await _collect(engine, "baseline")
            assert len(baseline) == GEN_TOKENS, (
                f"baseline stopped early ({len(baseline)} tokens); the pause "
                f"would not land mid-generation")

            produced: list[int] = []
            observed: dict[str, int] = {}

            async def pauser():
                while len(produced) < PAUSE_AFTER_TOKENS:
                    await asyncio.sleep(0.01)
                    if len(produced) >= GEN_TOKENS:
                        break
                await engine.pause_generation(mode="keep", clear_cache=False)
                assert await engine.is_paused()
                observed["at_pause"] = len(produced)
                await asyncio.sleep(PAUSE_HOLD_S)
                observed["after_hold"] = len(produced)
                await engine.resume_generation()
                assert not await engine.is_paused()

            task = asyncio.create_task(pauser())
            params = SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS)
            async for out in engine.generate(PROMPT, params, "paused"):
                produced[:] = list(out.outputs[0].token_ids)
            await task

            assert observed["at_pause"] < GEN_TOKENS, (
                "request finished before the pause; increase GEN_TOKENS")
            # The freeze itself: no forward progress while paused.
            assert observed["after_hold"] == observed["at_pause"], (
                f"generation advanced during the pause: "
                f"{observed['at_pause']} -> {observed['after_hold']} tokens")
            # The request was frozen, not aborted.
            assert len(produced) == GEN_TOKENS, (
                f"request did not finish after resume "
                f"({len(produced)}/{GEN_TOKENS} tokens)")
            # And freezing did not corrupt it.
            assert produced == baseline, (
                "tokens generated across a pause differ from an "
                "uninterrupted run")

        loop.run_until_complete(scenario())


class TestRequestsSubmittedWhilePaused:
    """A pause defers new work; it must not reject or abort it."""

    def test_request_added_during_pause_completes_after_resume(
            self, loop, engine):

        async def scenario():
            await engine.pause_generation(mode="keep", clear_cache=False)
            assert await engine.is_paused()

            task = asyncio.create_task(
                _collect(engine, "queued-during-pause", max_tokens=16))
            # Give it time to be admitted and sit in the waiting queue.
            await asyncio.sleep(PAUSE_HOLD_S)
            assert not task.done(), (
                "a request submitted during a pause produced output before "
                "resume")

            await engine.resume_generation()
            token_ids, _ = await asyncio.wait_for(task, timeout=120)
            assert len(token_ids) == 16

        loop.run_until_complete(scenario())


class TestPauseWithCacheClear:
    """clear_cache=True reaches reset_encoder_cache over collective_rpc.

    That RPC resolves by name and is issued regardless of modality, so before
    TPUWorker grew the method this failed on every TPU model, text-only
    included, with `NotImplementedError: Method 'reset_encoder_cache' is not
    implemented.` Verified by deleting the method and re-running this test.
    Placed last: it aborts in-flight work and wipes the prefix cache, which
    would disturb the tests above.
    """

    def test_pause_clear_cache_then_engine_still_serves(self, loop, engine):

        async def scenario():
            await engine.pause_generation(mode="abort", clear_cache=True)
            assert await engine.is_paused()
            await engine.resume_generation()
            assert not await engine.is_paused()

            token_ids, _ = await _collect(engine,
                                          "after-cache-clear",
                                          max_tokens=8)
            assert len(token_ids) == 8, (
                "engine did not serve correctly after a cache-clearing pause")

        loop.run_until_complete(scenario())
