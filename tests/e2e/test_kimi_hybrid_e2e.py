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
"""End-to-end serving for the Kimi hybrid stack (KDA + MLA in one model).

This is the first JAX-native model whose layers do not all take the same kind
of KV cache: the KDA layers own recurrent state in the mamba pool while the
MLA layers use the paged attention cache. The checks here are about that
plumbing surviving a real engine step -- spec emission, per-layer slot
mapping, mamba slot assignment across a batch -- so they compare against the
reference decode captured from the HF implementation rather than against a
loose text heuristic.

Set ``K3_TINY_CKPT`` to a Kimi-shaped checkpoint directory containing
``goldens_run1.npz``; the tests skip when it is unset.
"""

import os

import numpy as np
import pytest
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

CKPT = os.environ.get("K3_TINY_CKPT", "")
TP = int(os.environ.get("K3_E2E_TP", "1"))

pytestmark = pytest.mark.skipif(not CKPT, reason="K3_TINY_CKPT unset")


@pytest.fixture(scope="module")
def goldens():
    return np.load(os.path.join(CKPT, "goldens_run1.npz"))


@pytest.fixture(scope="module")
def llm():
    engine = LLM(
        model=CKPT,
        trust_remote_code=True,
        # Everything here is driven by the reference capture's token ids, and
        # the fixture checkpoint ships no tokenizer.
        skip_tokenizer_init=True,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        tensor_parallel_size=TP,
        # Prefix caching is off: recurrent state is not block-addressable, so
        # a cache hit would skip the prefill that builds it.
        enable_prefix_caching=False,
        additional_config={
            "sharding": {
                "sharding_strategy": {
                    "enable_dp_attention": True
                }
            }
        },
    )
    try:
        yield engine
    finally:
        engine.llm_engine.engine_core.shutdown()


def _greedy(llm, prompt_ids, n):
    out = llm.generate([TokensPrompt(prompt_token_ids=list(prompt_ids))],
                       SamplingParams(temperature=0.0,
                                      max_tokens=n,
                                      ignore_eos=True))
    return [int(t) for t in out[0].outputs[0].token_ids]


def _model_runner(llm):
    """The in-process model runner, or None when the engine runs out of
    process (multiproc executor) and cannot be introspected directly."""
    node = llm.llm_engine.engine_core
    for attr in ("engine_core", "model_executor", "driver_worker",
                 "model_runner"):
        node = getattr(node, attr, None)
        if node is None:
            return None
    return node


def test_hybrid_cache_layout_is_per_layer(llm):
    """The KDA layers must hold recurrent state (a tuple of arrays) and the
    MLA layers a paged attention array -- a model-wide `use_mla` would give
    every layer the same kind."""
    from vllm.v1.kv_cache_interface import MambaSpec

    runner = _model_runner(llm)
    if runner is None:
        pytest.skip("engine runs out of process; runner not introspectable")
    spec = runner.get_kv_cache_spec()
    mamba_layers = {n for n, s in spec.items() if isinstance(s, MambaSpec)}
    assert mamba_layers, "no recurrent layers: hybrid spec emission is off"
    assert len(mamba_layers) < len(spec), "every layer came out recurrent"

    index = runner.layer_name_to_kvcache_index
    assert len(index) == len(spec)
    # Distinct arrays for distinct layers, and of the right kind each.
    assert sorted(index.values()) == list(range(len(runner.kv_caches)))
    for name, idx in index.items():
        entry = runner.kv_caches[idx]
        assert isinstance(entry, tuple) == (name in mamba_layers), (
            f"{name} -> kv_caches[{idx}] is the wrong kind of cache")


def test_prefill_argmax_matches_the_reference(llm, goldens):
    """Prefill the reference prompt at a range of lengths; the next token
    must be the one the fp32 reference picked.

    Only prefill is compared against the fp32 reference. Multi-step greedy
    decode is *not*: on this random-weight fixture the reference logits are
    nearly degenerate (the step-7 top-1 leads top-2 by 0.055), while a bf16
    forward moves them by ~2-4. The offline harness reproduces the same
    divergence from the same reference without the engine in the picture, so
    asserting exact decode tokens here would be testing bf16 arithmetic, not
    the serving path. Token-level decode fidelity is gated on the real-weight
    model, whose logit margins are orders of magnitude wider.

    Each length runs in its own request so no case gets chunked -- chunked
    prefill splits a request across steps, which is a separate path.
    """
    prompt_ids = [int(t) for t in goldens["d32.input_ids"][0]]
    ref = [int(t) for t in goldens["d32.greedy_tokens"][0]]
    mismatches = []
    for k in range(len(ref)):
        got = _greedy(llm, prompt_ids + ref[:k], 1)[0]
        if got != ref[k]:
            mismatches.append((len(prompt_ids) + k, got, ref[k]))
    print(f"[observed] prefill argmax agreement "
          f"{len(ref) - len(mismatches)}/{len(ref)}")
    assert not mismatches, (
        f"prefill picked a different token than the reference at "
        f"(prefill_len, got, ref) = {mismatches}")


def test_batched_requests_do_not_share_recurrent_state(llm, goldens):
    """Each request gets its own mamba slot. Running a prompt alongside
    unrelated traffic must give the same continuation as running it alone --
    if slots aliased, the recurrent state would be cross-contaminated.

    Compared against this engine's own solo run rather than the reference, so
    the check is about slot isolation and holds at any dtype.
    """
    prompt_ids = [int(t) for t in goldens["d32.input_ids"][0]]
    n = len(goldens["d32.greedy_tokens"][0])
    solo = _greedy(llm, prompt_ids, n)

    distractors = [
        list(prompt_ids[:16]),
        list(reversed(prompt_ids)),
        list(prompt_ids[8:]),
    ]
    prompts = [TokensPrompt(prompt_token_ids=list(prompt_ids))
               ] + [TokensPrompt(prompt_token_ids=d) for d in distractors]
    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=n,
                                ignore_eos=True))
    batched = [int(t) for t in outs[0].outputs[0].token_ids]
    print(f"[observed] solo={solo} batched={batched}")
    assert batched == solo, (
        "the same prompt decoded differently when batched with other "
        "requests: recurrent state is leaking across mamba slots")


def test_repeated_requests_reset_the_recurrent_state(llm, goldens):
    """A finished request's slot is recycled. The same prompt run twice must
    give the same tokens both times -- otherwise state from the previous
    occupant leaked into the new request.

    Self-consistency, so it holds at any dtype."""
    prompt_ids = [int(t) for t in goldens["d32.input_ids"][0]]
    n = len(goldens["d32.greedy_tokens"][0])
    first = _greedy(llm, prompt_ids, n)
    # Churn the slot pool with other requests in between.
    _greedy(llm, list(reversed(prompt_ids)), n)
    second = _greedy(llm, prompt_ids, n)
    assert first == second, (
        "the same prompt decoded differently after the slot was recycled: "
        "recurrent state is not being reset on slot reuse")
