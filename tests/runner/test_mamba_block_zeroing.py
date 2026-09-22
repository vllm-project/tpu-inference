# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Per-allocation zeroing of mamba state slots.

A slot handed to a request still holds its previous owner's checkpoint; the
GDN kernel only writes one per forward pass, so the rest stay stale. These
cover the two halves: picking out which slots are genuinely new, and clearing
them on device.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.runner.kv_cache import (_bucket_num_rows, create_mamba_cache,
                                           zero_mamba_blocks)
from tpu_inference.runner.persistent_batch_manager import \
    PersistentBatchManager

BLOCK_SIZE = 256


def _manager(gid: int = 1) -> PersistentBatchManager:
    pbm = PersistentBatchManager.__new__(PersistentBatchManager)
    pbm.mamba_kv_cache_group_id = gid
    pbm.mamba_block_size = BLOCK_SIZE
    pbm.new_mamba_blocks = {}
    return pbm


# ------------------------------------------------------------ which slots --


def test_cached_request_records_every_appended_slot():
    pbm = _manager()
    # new_block_ids for a running request holds only what it just gained.
    pbm._record_new_mamba_blocks("r0", ([70, 71], [12, 13]))
    assert pbm.new_mamba_blocks == {"r0": [12, 13]}


def test_new_request_skips_its_prefix_cache_hits():
    """The leading blocks of a new request's list are the hit it is about to
    resume from -- zeroing those would destroy the very state it needs."""
    pbm = _manager()
    # 1536 computed tokens == 6 blocks: five null placeholders and the hit.
    block_ids = ([0] * 6 + [90, 91], [0, 0, 0, 0, 0, 55, 56, 57])
    pbm._record_new_mamba_blocks("r0",
                                 block_ids,
                                 num_computed_tokens=1536,
                                 is_full_list=True)
    assert pbm.new_mamba_blocks == {"r0": [56, 57]}


def test_cold_new_request_records_everything():
    pbm = _manager()
    pbm._record_new_mamba_blocks("r0", ([70], [20, 21, 22]),
                                 num_computed_tokens=0,
                                 is_full_list=True)
    assert pbm.new_mamba_blocks == {"r0": [20, 21, 22]}


def test_records_nothing_when_no_mamba_group_is_wired():
    pbm = _manager(gid=None)
    pbm.mamba_kv_cache_group_id = None
    pbm._record_new_mamba_blocks("r0", ([70], [20, 21]))
    assert pbm.new_mamba_blocks == {}


def test_group_id_past_the_end_is_ignored():
    pbm = _manager(gid=5)
    pbm._record_new_mamba_blocks("r0", ([70], [20]))
    assert pbm.new_mamba_blocks == {}


# ----------------------------------------------------------------- device --


def test_bucketing_stays_on_a_short_ladder():
    assert [_bucket_num_rows(n) for n in (1, 8, 9, 32, 33, 512)] == [
        8, 8, 32, 32, 128, 512
    ]
    # Past the ladder it rounds up to a multiple of the last rung.
    assert _bucket_num_rows(600) == 1024


def test_zeroing_clears_only_the_named_rows():
    devices = np.array(jax.devices())
    mesh = Mesh(devices.reshape(len(devices)), ("x", ))
    sharding = NamedSharding(mesh, P("x"))

    rows_total = 8 * len(devices)
    state = jax.device_put(
        np.full((rows_total, 4, 128), 3.0, dtype=np.float32), sharding)

    to_zero = np.array([1, 5, rows_total - 1], dtype=np.int32)
    state = zero_mamba_blocks(state, to_zero)

    host = np.asarray(jax.device_get(state))
    assert np.count_nonzero(host[to_zero]) == 0
    untouched = np.setdiff1d(np.arange(rows_total), to_zero)
    assert np.all(host[untouched] == 3.0)


def test_zeroing_an_empty_set_is_a_no_op():
    devices = np.array(jax.devices())
    mesh = Mesh(devices.reshape(len(devices)), ("x", ))
    sharding = NamedSharding(mesh, P("x"))
    state = create_mamba_cache((len(devices), 4), jnp.float32, sharding)
    same = zero_mamba_blocks(state, np.array([], dtype=np.int32))
    assert same is state


def test_zeroing_scrubs_a_recycled_slot():
    """The case the fix exists for: a slot carrying a previous request's
    state is handed to a new one and must not read back as that state."""
    devices = np.array(jax.devices())
    mesh = Mesh(devices.reshape(len(devices)), ("x", ))
    sharding = NamedSharding(mesh, P("x"))

    rows = 4 * len(devices)
    prev_owner_state = np.arange(rows * 16, dtype=np.float32).reshape(rows, 16)
    state = jax.device_put(prev_owner_state, sharding)

    recycled = np.array([2, 3], dtype=np.int32)
    state = zero_mamba_blocks(state, recycled)

    host = np.asarray(jax.device_get(state))
    assert np.all(host[recycled] == 0.0)
    assert np.all(host[0] == prev_owner_state[0])
