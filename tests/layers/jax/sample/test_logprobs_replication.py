# SPDX-License-Identifier: Apache-2.0
"""The logprobs tensors the runner fetches must be replicated.

`_jax_logprobs_copy_to_host_async()` feeds `jax.device_get()`, which on a
multi-host mesh can only read fully replicated arrays. Under DP attention the
logits are sharded over ATTN_DATA, an axis that spans hosts, so the outputs
inherit a sharding no single process can fetch.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.jax.sample.sampling import (
    compute_and_gather_logprobs, compute_and_gather_logprobs_for_host,
    compute_and_gather_prompt_logprobs,
    compute_and_gather_prompt_logprobs_for_host)

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


@pytest.mark.parametrize("host_fn,plain_fn", [
    (compute_and_gather_logprobs_for_host, compute_and_gather_logprobs),
    (compute_and_gather_prompt_logprobs_for_host,
     compute_and_gather_prompt_logprobs),
])
def test_host_variants_replicate_without_changing_values(host_fn, plain_fn):
    """Replicated outputs, identical values to the unreplicated call."""
    mesh = _mesh()
    logits, token_ids = _inputs(mesh, P(ATTN_DATA, None))

    with jax.set_mesh(mesh):
        replicated = host_fn(logits, token_ids, MAX_LOGPROBS)
    plain = plain_fn(logits, token_ids, MAX_LOGPROBS)

    for name in ("logprob_token_ids", "logprobs", "selected_token_ranks"):
        got, want = getattr(replicated, name), getattr(plain, name)
        assert got.sharding.is_fully_replicated, name
        assert np.allclose(np.asarray(got), np.asarray(want)), name


@pytest.mark.skipif(len(jax.devices()) < 2, reason="needs a sharded mesh")
def test_plain_variant_is_not_replicated_under_dp_attention():
    """The split is load-bearing: the plain call is what the fused decode loop
    uses per step, and it must stay unreplicated so the loop can gather its
    accumulated buffers once instead of once per step."""
    mesh = _mesh()
    logits, token_ids = _inputs(mesh, P(ATTN_DATA, None))

    plain = compute_and_gather_logprobs(logits, token_ids, MAX_LOGPROBS)

    assert not plain.logprobs.sharding.is_fully_replicated
    assert not plain.selected_token_ranks.sharding.is_fully_replicated


def _collective_count(fn, mesh, logits_spec):
    logits, token_ids = _inputs(mesh, logits_spec)
    with jax.set_mesh(mesh):
        text = jax.jit(lambda a, b: fn(a, b, MAX_LOGPROBS)).lower(
            logits, token_ids).compile()
    return len([
        m for m in re.finditer(r'%(\S+) = \S+ (all-gather|all-reduce)\(',
                               text.as_text())
        if "start" not in m.group(1) and "prepare" not in m.group(1)
    ])


@pytest.mark.skipif(len(jax.devices()) < 2, reason="needs a sharded mesh")
def test_replication_is_free_for_the_raw_modes():
    """Raw modes shard the vocab, not the requests, so the outputs are already
    replicated and the constraint must fold away to nothing. This is why the
    guarantee can be unconditional rather than a per-call opt-in."""
    mesh = _mesh()
    raw_spec = P("data", MLP_TENSOR)

    plain = _collective_count(compute_and_gather_logprobs, mesh, raw_spec)
    host = _collective_count(compute_and_gather_logprobs_for_host, mesh,
                             raw_spec)

    assert host == plain, (
        f"replication added {host - plain} collectives in a raw mode, "
        "where the outputs are already replicated")
