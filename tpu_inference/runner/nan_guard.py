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
"""DEV ONLY: first-occurrence NaN tripwire for the accuracy-collapse hunt.

Enabled with ``NAN_GUARD=1``. After every model step it reduces, on device,
per-slot any-NaN over every mamba state tensor (conv windows + recurrent
state) and per-row any-NaN over the step's logits, then reports:

  * ``[nan-guard] TRIP`` on the FIRST clean->NaN transition of any pool slot
    (a poisoning event), with step index, layer name, tensor name, slot ids,
    and the batch's request->slot assignment at that step.
  * ``[nan-guard] TRIP`` on any NaN logits row (read-side hit), with row ids
    and the same batch snapshot.
  * A heartbeat line every ``NAN_GUARD_HEARTBEAT_EVERY`` (default 100) steps
    so a clean log proves the guard was alive, not skipped.

"Clean->NaN transition" rather than plain any-NaN makes this compose with
``MAMBA_POOL_NAN_CANARY=1`` (pools allocated as all-NaN): canary NaN in a
never-written slot is expected and not an event; a slot that was observed
clean and later reads NaN was poisoned at runtime. Never-written-NaN slot
counts appear in the heartbeat instead.

Attention KV pools are reduced to a per-layer any-NaN scalar every
``NAN_GUARD_KV_EVERY`` (default 20) steps -- enough to say which layer's KV
went NaN and when, without paying a full-pool read every step.

All reductions run under jit with replicated outputs, so on a multi-host pod
the transfer after the reduction is host-local. Every host runs the same
deterministic schedule, so the collectives this inserts stay in lockstep.

Not wired into the fused decode loop (``_execute_continue_decode``): the
failing config runs loop-off, and the loop body is a single jit the guard
cannot see into per step.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_KDA_TENSOR_NAMES = ("conv_q", "conv_k", "conv_v", "recurrent")
_GDN_TENSOR_NAMES = ("conv", "recurrent")


def _tensor_name(tuple_len: int, t: int, ndim: int) -> str:
    if tuple_len == 4:
        return _KDA_TENSOR_NAMES[t]
    if tuple_len == 2:
        return _GDN_TENSOR_NAMES[t]
    return f"t{t}_{'conv' if ndim == 3 else 'recurrent'}"


class NanGuard:

    def __init__(self, runner):
        self.runner = runner
        self.step = -1
        self.kv_every = int(os.environ.get("NAN_GUARD_KV_EVERY", "20"))
        self.heartbeat_every = int(
            os.environ.get("NAN_GUARD_HEARTBEAT_EVERY", "100"))
        # (cache_idx, tensor_idx) -> np.bool_[num_slots]
        self.seen_clean = {}
        self.reported = {}
        self.attn_reported = set()
        self.tripped = False
        self._names = None
        repl = NamedSharding(runner.mesh, PartitionSpec())
        self._mamba_jit = jax.jit(self._mamba_reduce, out_shardings=repl)
        self._attn_jit = jax.jit(self._attn_reduce, out_shardings=repl)

    def _layer_name(self, cache_idx: int) -> str:
        if self._names is None:
            rev = {}
            for name, idx in self.runner.layer_name_to_kvcache_index.items():
                rev.setdefault(idx, name)
            self._names = rev
        return self._names.get(cache_idx, f"cache_{cache_idx}")

    @staticmethod
    def _mamba_reduce(caches, logits, full_logits):
        out = {}
        for i, c in enumerate(caches):
            if not isinstance(c, tuple):
                continue
            for t, arr in enumerate(c):
                if jnp.issubdtype(arr.dtype, jnp.floating):
                    out[f"{i}:{t}"] = jnp.any(jnp.isnan(arr),
                                              axis=tuple(range(1, arr.ndim)))
        out["logits"] = jnp.any(jnp.isnan(logits), axis=-1)
        if full_logits is not None:
            out["full_logits"] = jnp.any(jnp.isnan(full_logits), axis=-1)
        return out

    @staticmethod
    def _attn_reduce(caches):
        out = {}
        for i, c in enumerate(caches):
            if not isinstance(c, tuple) and jnp.issubdtype(
                    c.dtype, jnp.floating):
                out[str(i)] = jnp.any(jnp.isnan(c))
        return out

    def _batch_snapshot(self) -> str:
        ib = self.runner.input_batch
        nreq = ib.num_reqs
        slots = getattr(ib, "mamba_state_indices_cpu", None)
        return (
            f"num_reqs={nreq} req_ids={ib.req_ids[:nreq]} "
            f"slots={slots[:nreq].tolist() if slots is not None else None}")

    def check(self, kv_caches, logits, full_logits=None) -> None:
        self.step += 1
        tuple_lens = {
            i: len(c)
            for i, c in enumerate(kv_caches) if isinstance(c, tuple)
        }
        res = jax.device_get(self._mamba_jit(kv_caches, logits, full_logits))

        events = []
        never_written_nan = 0
        for key, nan_arr in res.items():
            nan_arr = np.asarray(nan_arr)
            if key in ("logits", "full_logits"):
                rows = np.nonzero(nan_arr)[0]
                if rows.size:
                    events.append((key, None, rows))
                continue
            seen = self.seen_clean.get(key)
            if seen is None:
                seen = np.zeros(nan_arr.shape, dtype=bool)
                self.seen_clean[key] = seen
                self.reported[key] = np.zeros(nan_arr.shape, dtype=bool)
            reported = self.reported[key]
            new_bad = nan_arr & seen & ~reported
            never_written_nan += int((nan_arr & ~seen).sum())
            if new_bad.any():
                events.append(("pool", key, np.nonzero(new_bad)[0]))
                reported |= new_bad
            seen |= ~nan_arr

        for kind, key, ids in events:
            snap = self._batch_snapshot()
            if kind == "pool":
                i, t = (int(x) for x in key.split(":"))
                logger.error(
                    "[nan-guard] TRIP step=%d pool=%s tensor=%s "
                    "poisoned_slots=%s | %s", self.step, self._layer_name(i),
                    _tensor_name(tuple_lens[i], t, kv_caches[i][t].ndim),
                    ids.tolist(), snap)
            else:
                logger.error("[nan-guard] TRIP step=%d %s nan_rows=%s | %s",
                             self.step, kind, ids.tolist(), snap)
        if events and not self.tripped:
            self.tripped = True
            logger.error(
                "[nan-guard] FIRST TRIP was step=%d; all later trips log "
                "once per (pool, slot).", self.step)

        if self.step % self.kv_every == 0:
            attn = jax.device_get(self._attn_jit(kv_caches))
            for key, bad in attn.items():
                if bool(bad) and key not in self.attn_reported:
                    self.attn_reported.add(key)
                    logger.error(
                        "[nan-guard] TRIP step=%d attention kv pool=%s "
                        "contains NaN | %s", self.step,
                        self._layer_name(int(key)), self._batch_snapshot())

        if self.step % self.heartbeat_every == 0:
            poisoned = sum(int(r.sum()) for r in self.reported.values())
            logger.info(
                "[nan-guard] alive step=%d poisoned_slots=%d "
                "never_written_nan_slots=%d attn_pools_nan=%d", self.step,
                poisoned, never_written_nan, len(self.attn_reported))
