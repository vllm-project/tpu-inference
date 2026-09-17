# SPDX-License-Identifier: Apache-2.0
"""The logprobs tensors the runner fetches must be replicated.

`_jax_logprobs_copy_to_host_async()` feeds `jax.device_get()`, which on a
multi-host mesh can only read fully replicated arrays. Under DP attention the
logits are sharded over ATTN_DATA, an axis whose replica groups interleave
across hosts, so without `out_shardings` the outputs inherit a sharding no
single process can fetch.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.jax.sample.sampling import (
    compute_and_gather_logprobs, compute_and_gather_prompt_logprobs,
    compute_logprobs, gather_logprobs)

VOCAB = 128
MAX_LOGPROBS = 2

# Mirrors the serving mesh: 'data' x 'attn_dp' carries requests (ATTN_DATA),
# 'model' carries the vocab split. Sized off the devices actually present.
ATTN_DATA = ("data", "attn_dp")
MLP_TENSOR = ("attn_dp", "model")


def _mesh():
    n = len(jax.devices())
    attn_dp, model = (n // 2, 2) if n % 2 == 0 else (n, 1)
    return Mesh(
        np.array(jax.devices()).reshape(1, attn_dp, model),
        ("data", "attn_dp", "model"))


def _inputs(mesh, logits_spec):
    num_reqs = max(2, mesh.shape["data"] * mesh.shape["attn_dp"])
    logits = jax.device_put(
        jnp.arange(num_reqs * VOCAB,
                   dtype=jnp.float32).reshape(num_reqs, VOCAB),
        NamedSharding(mesh, logits_spec))
    token_ids = jax.device_put(jnp.zeros((num_reqs, ), dtype=jnp.int32),
                               NamedSharding(mesh, P()))
    return logits, token_ids


def _unconstrained(logits, token_ids):
    """What the functions returned before out_shardings was added."""
    return jax.jit(
        lambda a, b: gather_logprobs(compute_logprobs(a), b, MAX_LOGPROBS))(
            logits, token_ids)


@pytest.mark.parametrize(
    "fn", [compute_and_gather_logprobs, compute_and_gather_prompt_logprobs])
def test_outputs_are_replicated_under_dp_attention(fn):
    """Replicated outputs, and the values are unchanged by the constraint."""
    mesh = _mesh()
    logits, token_ids = _inputs(mesh, P(ATTN_DATA, None))

    with jax.set_mesh(mesh):
        got = fn(logits, token_ids, MAX_LOGPROBS)

    for name in ("logprob_token_ids", "logprobs", "selected_token_ranks"):
        assert getattr(got, name).sharding.is_fully_replicated, name

    if fn is compute_and_gather_logprobs:
        want = _unconstrained(logits, token_ids)
        for name in ("logprob_token_ids", "logprobs", "selected_token_ranks"):
            assert np.allclose(np.asarray(getattr(got, name)),
                               np.asarray(getattr(want, name))), name


@pytest.mark.skipif(len(jax.devices()) < 2, reason="needs a sharded mesh")
def test_the_constraint_is_what_replicates_them():
    """Without it the outputs really are unfetchable, so the test above is not
    passing for an unrelated reason."""
    mesh = _mesh()
    logits, token_ids = _inputs(mesh, P(ATTN_DATA, None))

    plain = _unconstrained(logits, token_ids)

    assert not plain.logprobs.sharding.is_fully_replicated
    assert not plain.selected_token_ranks.sharding.is_fully_replicated


def _collective_count(fn, mesh, logits, token_ids):
    with jax.set_mesh(mesh):
        text = jax.jit(lambda a, b: fn(a, b, MAX_LOGPROBS)).lower(
            logits, token_ids).compile().as_text()
    return len([
        m for m in re.finditer(r'%(\S+) = \S+ (all-gather|all-reduce)\(', text)
        if "start" not in m.group(1) and "prepare" not in m.group(1)
    ])


@pytest.mark.skipif(len(jax.devices()) < 2, reason="needs a sharded mesh")
def test_replication_is_free_for_the_raw_modes():
    """Raw modes shard the vocab, not the requests, so the outputs are already
    replicated and the constraint must fold away to nothing. This is why the
    guarantee can be unconditional rather than a per-call opt-in."""
    mesh = _mesh()
    logits, token_ids = _inputs(mesh, P("data", MLP_TENSOR))

    constrained = _collective_count(compute_and_gather_logprobs, mesh, logits,
                                    token_ids)
    plain = _collective_count(
        lambda a, b, k: gather_logprobs(compute_logprobs(a), b, k), mesh,
        logits, token_ids)

    assert constrained == plain, (
        f"replication added {constrained - plain} collectives in a raw mode, "
        "where the outputs are already replicated")
