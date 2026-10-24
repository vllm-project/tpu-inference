# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Trim the fixed per-step host->device path (env HOST_INPUT_FAST_PATH=1, default off).

Values reaching the device are unchanged. HOST_INPUT_FAST_PATH_PARTS=default|all|none|comma list picks the parts
(default = all = pack,mds,gbm):
  pack  ONE jax.device_put per _prepare_inputs for the sampling metadata + positions + request_distribution +
        metadata blob; a sampling field whose bytes equal the last put of that (field, shape, dtype, sharding)
        reuses its device array (stock: up to 7 separate puts, each through general_device_put).
  mds   the continue-decode max_decode_steps device scalar is cached per (value, ambient mesh) instead of an
        eager jnp.array per chunk.
  gbm   log only: reports the DPScheduler grammar fan-out counters of DP_SCHED_SKIP_GRAMMAR_FANOUT.
The first HOST_INPUT_FAST_PATH_CHECK (default 3) uses of pack/mds are verified against the stock values (bytes,
dtype, sharding); a mismatch turns that part off for the process (stock code from then on). Log lines carry the
[host-input-fast-path] tag: the config gate, the checks, the first continue-decode chunk and every
HOST_INPUT_FAST_PATH_LOG_EVERY chunks.

`pack` is refused for mrope, PCP, speculative decoding, non-align mamba, the "ray" multi-host backend and any
mesh that spans devices of other processes (only the default and the host-local "spmd" backends qualify).
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from tpu_inference import envs

_ALL = ("pack", "mds", "gbm")
_ST = {
    "on":
    False,
    "act":
    set(),
    "why": {},
    "inert":
    set(),
    "chunks":
    0,
    "cache": {},
    "mdsc": {},
    "check":
    3,
    "every":
    100,
    "c":
    dict.fromkeys(("puts", "arrays", "hit", "miss", "pk_ck", "mds_hit",
                   "mds_miss", "mds_ck"), 0),
}


def _log(msg):
    print("[host-input-fast-path] " + msg, file=sys.stderr, flush=True)


def _off(h, why):
    _ST["act"].discard(h)
    _ST["why"][h] = why
    _log("check=FAIL part=%s: %s -> %s OFF for this process (stock code "
         "from now on)" % (h, why, h))


def _gbm():
    """[skipped, fanned_out] grammar fan-out counters, or None when DP_SCHED_SKIP_GRAMMAR_FANOUT is off."""
    if not envs.DP_SCHED_SKIP_GRAMMAR_FANOUT:
        return None
    try:
        from tpu_inference.core.sched.dp_scheduler import DPScheduler
        return DPScheduler.skip_grammar_stats
    except Exception:
        return None


def _mesh_is_local(mesh):
    try:
        pid = jax.process_index()
        return all(d.process_index == pid for d in mesh.devices.flat)
    except Exception:
        return False


def gate(runner):
    """End of TPUModelRunner.__init__: resolve the requested parts for this runner; one log line. OFF: no-op."""
    _ST["on"] = bool(envs.HOST_INPUT_FAST_PATH)
    if not _ST["on"]:
        return
    _ST["check"] = max(0, envs.HOST_INPUT_FAST_PATH_CHECK)
    _ST["every"] = max(1, envs.HOST_INPUT_FAST_PATH_LOG_EVERY)
    raw = (envs.HOST_INPUT_FAST_PATH_PARTS or "default").replace(" ", "")
    req = {"default": list(_ALL), "all": list(_ALL), "none": []}.get(raw)
    if req is None:
        req = [h for h in raw.split(",") if h]
        unknown = [h for h in req if h not in _ALL]
        if unknown:
            raise RuntimeError(
                "[host-input-fast-path] FATAL: unknown part(s) %s in "
                "HOST_INPUT_FAST_PATH_PARTS=%r (known %s)" %
                (unknown, raw, _ALL))
    why = {}
    mhb = str(envs.TPU_MULTIHOST_BACKEND or "").lower()
    local = _mesh_is_local(runner.mesh)
    pcp = runner.pcp_preprocessor is not None
    if (mhb not in ("", "spmd") or not local or runner.uses_mrope or pcp
            or runner.speculative_config is not None):
        why["pack"] = (
            "multihost=%r mesh_local=%s mrope=%s pcp=%s spec=%s" %
            (mhb, local, runner.uses_mrope, pcp, runner.speculative_config))
    if _gbm() is None:
        why["gbm"] = "grammar fan-out counters off (DP_SCHED_SKIP_GRAMMAR_FANOUT unset)"
    _ST["act"] = set(h for h in req if h not in why)
    _ST["why"] = dict((h, r) for h, r in why.items() if h in req)
    _log("config gate: requested=%s ACTIVE=%s dropped=%s | multihost=%r "
         "mesh_local=%s dp=%s check=%d every=%d" %
         (req, sorted(_ST["act"]), _ST["why"], mhb, local, runner.dp_size,
          _ST["check"], _ST["every"]))


def prep(runner):
    """Top of _prepare_inputs' device section, every step. True -> this step's puts go in ONE packed put."""
    if not _ST["on"] or "pack" not in _ST["act"]:
        return False
    # the stock non-mamba put branch must be the one taken
    kvc = runner.kv_cache_config
    if ((kvc.has_mamba_layers and getattr(
            runner.cache_config, "mamba_cache_mode", "none") != "align")
            or runner.uses_mrope or runner.pcp_preprocessor is not None):
        _off("pack", "mamba/mrope/pcp at runtime")
        return False
    return True


def put(runner, padded_num_reqs, req_indices_dp, dp_sh, positions, rdist, blob,
        md_sh):
    """ONE jax.device_put for the sampling metadata, positions, request_distribution and the metadata blob.

    The sampling host arrays are built exactly as TPUSupportedSamplingMetadata.from_input_batch builds them; a
    field whose bytes equal the last put of that (field, shape, dtype, sharding) reuses that device array. Every
    array keeps the sharding the stock code gives it. Returns (sampling_metadata, positions,
    request_distribution, blob) on device.
    """
    from tpu_inference.layers.jax.sample.sampling_metadata import (
        DEFAULT_SAMPLING_PARAMS, TPUSupportedSamplingMetadata)
    from tpu_inference.utils import device_array
    c, ib, mesh = _ST["c"], runner.input_batch, runner.mesh
    needs_lp = ib.max_num_logprobs > 0 if ib.max_num_logprobs else False
    items = [("dummy", np.zeros((1 if needs_lp else 2, ), dtype=np.int32),
              NamedSharding(mesh, PartitionSpec()))]
    greedy = ib.all_greedy
    if not greedy:
        ndp = len(req_indices_dp)
        assert padded_num_reqs % ndp == 0, (
            f"padded_num_reqs ({padded_num_reqs}) must be divisible by "
            f"dp_size ({ndp})")
        per = padded_num_reqs // ndp
        for name, cpu in (("temperature", ib.temperature_cpu),
                          ("top_k", ib.top_k_cpu), ("top_p", ib.top_p_cpu)):
            out = np.full((padded_num_reqs, ),
                          DEFAULT_SAMPLING_PARAMS[name],
                          dtype=cpu.dtype)
            for r in range(ndp):
                ix = req_indices_dp.get(r, [])
                if ix:
                    out[r * per:r * per + len(ix)] = cpu[ix]
            items.append((name, out, dp_sh))
    cache, keys, xs, shs, got = _ST["cache"], [], [], [], {}
    for name, x, sh in items:
        k, b = (name, x.shape, x.dtype.str, sh), x.tobytes()
        hit = cache.get(k)
        if hit is not None and hit[0] == b and not hit[1].is_deleted():
            got[name] = hit[1]
            c["hit"] += 1
        else:
            keys.append((name, k, b))
            xs.append(x)
            shs.append(sh)
            c["miss"] += 1
    xs += [positions, rdist, blob]
    shs += [dp_sh, md_sh, md_sh]
    ys = jax.device_put(xs, shs)
    c["puts"] += 1
    c["arrays"] += len(xs)
    for (name, k, b), y in zip(keys, ys):
        cache[k] = (b, y)
        got[name] = y
    if c["pk_ck"] < _ST["check"]:
        c["pk_ck"] += 1
        chk = items + [("positions", positions, dp_sh),
                       ("rdist", rdist, md_sh), ("blob", blob, md_sh)]
        dev = [got[n] for n, _, _ in items] + list(ys[-3:])
        bad = [
            n for (n, x, sh), y in zip(chk, dev)
            if y.sharding != sh or y.dtype != x.dtype or y.shape != x.shape
            or np.asarray(jax.device_get(y)).tobytes() != x.tobytes()
        ]
        if bad:
            _off(
                "pack",
                "device arrays %s differ from their host bytes/sharding" % bad)
            rd, bl = device_array(mesh, (rdist, blob), sharding=md_sh)
            return (TPUSupportedSamplingMetadata.from_input_batch(
                mesh,
                ib,
                padded_num_reqs,
                sharding=dp_sh,
                req_indices_dp=req_indices_dp),
                    device_array(mesh, positions, sharding=dp_sh), rd, bl)
        _log("pack check %d/%d PASS: %d arrays in 1 device_put (%d reused) "
             "byte- and sharding-equal to the host arrays; fields=%s" %
             (c["pk_ck"], _ST["check"], len(xs), len(items) - len(keys),
              [n for n, _, _ in items]))
    if greedy:
        smd = TPUSupportedSamplingMetadata(do_sampling=False,
                                           logprobs=needs_lp,
                                           _cache_collision_dummy=got["dummy"])
    else:
        smd = TPUSupportedSamplingMetadata(temperature=got["temperature"],
                                           top_p=got["top_p"],
                                           top_k=got["top_k"],
                                           _cache_collision_dummy=got["dummy"],
                                           do_sampling=True,
                                           logprobs=needs_lp)
    return smd, ys[-3], ys[-2], ys[-1]


def mds(n):
    """max_decode_steps as a device scalar. Inactive: the stock jnp.array call."""
    c = _ST["c"]
    if not _ST["on"] or "mds" not in _ST["act"]:
        return jnp.array(n, dtype=jnp.int32)
    try:
        from jax._src import mesh as _jm
        k = (int(n), _jm.get_concrete_mesh(),
             _jm.thread_resources.env.physical_mesh)
    except Exception as exc:
        _off("mds", "ambient mesh unreadable (%r)" % (exc, ))
        return jnp.array(n, dtype=jnp.int32)
    a = _ST["mdsc"].get(k)
    if a is None or a.is_deleted():
        a = _ST["mdsc"][k] = jnp.array(n, dtype=jnp.int32)
        c["mds_miss"] += 1
        return a
    c["mds_hit"] += 1
    if c["mds_ck"] < _ST["check"]:
        c["mds_ck"] += 1
        f = jnp.array(n, dtype=jnp.int32)
        if (f.sharding != a.sharding or f.dtype != a.dtype
                or f.weak_type != a.weak_type or int(f) != int(a)):
            _off(
                "mds", "cached %r/%s differs from fresh %r/%s" %
                (a, a.sharding, f, f.sharding))
            return f
        _log("mds check %d/%d PASS: cached max_decode_steps=%d equals a fresh "
             "jnp.array (dtype %s, sharding %s)" %
             (c["mds_ck"], _ST["check"], int(n), a.dtype, a.sharding))
    return a


def _counts():
    c, g = _ST["c"], _gbm()
    return ("pack puts=%d arrays=%d reuse=%d/%d | mds reuse=%d/%d | gbm "
            "skip/fanout=%s | ACTIVE=%s" %
            (c["puts"], c["arrays"], c["hit"], c["hit"] + c["miss"],
             c["mds_hit"], c["mds_hit"] + c["mds_miss"],
             "%d/%d" % tuple(g) if g else "-", sorted(_ST["act"])))


def note_chunk():
    """Once per continue-decode chunk: counters log line (first chunk, then every HOST_INPUT_FAST_PATH_LOG_EVERY)."""
    if not _ST["on"]:
        return
    _ST["chunks"] += 1
    k = _ST["chunks"]
    if k == 1:
        _log("first chunk: ACTIVE=%s dropped=%s | %s" %
             (sorted(_ST["act"]), _ST["why"], _counts()))
    if k % _ST["every"]:
        return
    _log("chunks %d-%d | %s" % (k - _ST["every"] + 1, k, _counts()))
    c, g = _ST["c"], _gbm()
    eng = {"pack": c["puts"], "mds": c["mds_hit"], "gbm": g[0] if g else 0}
    for h in sorted(_ST["act"] - _ST["inert"]):
        if not eng[h]:
            _ST["inert"].add(h)
            _log("INERT: part %s is active but never engaged in %d chunks" %
                 (h, k))
