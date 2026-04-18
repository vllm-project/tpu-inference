# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import functools
import time
from typing import TYPE_CHECKING, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (KVConnectorRole,
                                                               SupportsHMA)
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

import tpu_inference.distributed.utils as dist_utils
from tpu_inference.distributed.tpu_connector import (
    LoadMeta, SendMeta, TPUConnector, TPUConnectorMetadata,
    TPUConnectorScheduler, TPUConnectorWorker, get_uuid, insert_kv_chunks)
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_runner import TPUModelRunner
from tpu_inference.utils import device_array

ReqId = str

logger = init_logger(__name__)


class TPUConnectorHMA(TPUConnector, SupportsHMA):
    """TPU connector supporting hybrid memory allocator."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 role: KVConnectorRole,
                 kv_cache_config: "KVCacheConfig | None" = None):
        assert vllm_config.kv_transfer_config is not None
        self._connector_metadata = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPUConnectorHMAScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = TPUConnectorHMAWorker(vllm_config)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(
            request, block_ids)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        raise AssertionError(
            "TPUConnectorHMA.request_finished was called, but this "
            "connector is SupportsHMA. vLLM should dispatch to "
            "`request_finished_all_groups`.")


class TPUConnectorHMAScheduler(TPUConnectorScheduler):

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.is_producer or not request.kv_transfer_params:
            return 0, False

        # No trim, no block-alignment rounding. D pulls every block
        # (including a partial last one) and runs zero local re-prefill.
        # Required for Mamba correctness. P's Mamba state is a recurrent
        # summary of all prompt tokens. If D re-prefilled any tail tokens
        # locally, they would be fed through the Mamba recurrence a
        # second time on D.
        count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
        if count > 0:
            return count, True
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.is_producer or not request.kv_transfer_params:
            return

        params = request.kv_transfer_params
        if num_external_tokens > 0:
            local_block_ids = list(blocks.get_block_ids())
            assert all(isinstance(g, list) for g in local_block_ids), (
                f"Expected list[list[int]] from blocks.get_block_ids() "
                f"in HMA mode; got {local_block_ids}")
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=local_block_ids,
                remote_block_ids=params["remote_block_ids"],
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        else:
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=None,
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        logger.info(f"TPUConnectorHMAScheduler update_state_after_alloc --> "
                    f"reqs_to_load={self.reqs_to_load}")

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        raise AssertionError(
            "TPUConnectorHMAScheduler.request_finished was called, but "
            "this scheduler only handles SupportsHMA requests via "
            "`request_finished_all_groups`.")

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.
        No-op for D workers.

        Args:
            request (Request): the request object.
            block_ids: The block IDs allocated for this request and need to be freed.
        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        if not self.is_producer:
            return False, None

        # Mark the request finished only if the prefill is done and generates 1 output token.
        # The request's max_tokens has been reset to 1, so it must be finished by length capped.
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        # No trim, no block-alignment rounding. D will pull every block
        # (including a partial last one) and runs zero local re-prefill.
        # Required for Mamba correctness. P's Mamba state is a recurrent
        # summary of all prompt tokens. If D re-prefilled any tail tokens
        # locally, they would be fed through the Mamba recurrence a
        # second time on D.
        computed_per_group: list[list[int]] = [
            list(group_ids) for group_ids in block_ids
        ]
        delay_free_blocks = any(len(g) > 0 for g in computed_per_group)

        if delay_free_blocks:
            uuid = get_uuid()
            expiration_time = time.perf_counter(
            ) + dist_utils.get_p2p_wait_pull_timeout()
            self.reqs_to_send[request.request_id] = SendMeta(
                uuid=uuid,
                local_block_ids=computed_per_group,
                expiration_time=expiration_time)
            kv_transfer_params = dict(uuid=uuid,
                                      remote_block_ids=computed_per_group,
                                      remote_host=self.kv_ip,
                                      remote_port=self.kv_port)
            logger.info(f"TPUConnectorHMAScheduler ----> generated "
                        f"num_prompt_tokens={len(request.prompt_token_ids)} | "
                        f"num_computed_tokens={request.num_computed_tokens} | "
                        f"reqs_to_send={self.reqs_to_send} | "
                        f"kv_transfer_params={kv_transfer_params}")
        else:
            kv_transfer_params = {}
        return delay_free_blocks, kv_transfer_params


class TPUConnectorHMAWorker(TPUConnectorWorker):

    def register_runner(self, runner: TPUModelRunner):
        self.node_id = runner.topology_order_id
        self.runner = runner
        self.mesh = runner.mesh

        # KV transfer sanity counters. P bumps on send queued; D bumps on
        # pull completed. A cumulative line is logged every N events.
        self.stats_num_sends = 0
        self.stats_bytes_sent = 0
        self.stats_num_pulls = 0
        self.stats_bytes_pulled = 0
        self.stats_log_interval = 32
        self.num_kv_groups = len(runner.kv_cache_config.kv_cache_groups)

        kv_caches = runner.kv_caches
        layer_to_group_id: list[int] = [0] * len(kv_caches)
        for group_id, group in enumerate(
                runner.kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                assert layer_name in runner.layer_name_to_kvcache_index, (
                    f"Layer '{layer_name}' is listed in kv_cache_group "
                    f"{group_id} but has no entry in "
                    f"runner.layer_name_to_kvcache_index.")
                layer = runner.layer_name_to_kvcache_index[layer_name]
                layer_to_group_id[layer] = group_id
        # kv cache group id for each layer
        self.layer_to_group_id = layer_to_group_id

        # Whether a kv cache group is a Mamba one.
        self.group_is_mamba: list[bool] = [
            isinstance(g.kv_cache_spec, MambaSpec)
            for g in runner.kv_cache_config.kv_cache_groups
        ]

        kv_array_to_group_id: list[int] = []
        layer_to_kv_array_range: list[tuple[int, int]] = []
        for layer, cache in enumerate(kv_caches):
            start = len(kv_array_to_group_id)
            group_id = self.layer_to_group_id[layer]
            if isinstance(cache, tuple):
                for _ in cache:
                    kv_array_to_group_id.append(group_id)
            else:
                kv_array_to_group_id.append(group_id)
            layer_to_kv_array_range.append((start, len(kv_array_to_group_id)))
        # Index of flattened kv cache jax array -> kv cache group id
        self.kv_array_to_group_id: list[int] = kv_array_to_group_id
        # layer -> [start_idx, end_idx) range in the flattened kv cache
        self.layer_to_kv_array_range: list[tuple[
            int, int]] = layer_to_kv_array_range
        self.num_kv_arrays: int = len(kv_array_to_group_id)

        flat_kv_caches = _flatten_kv_caches(kv_caches)
        self.kv_array_shapes: list[list[int]] = [
            list(a.shape) for a in flat_kv_caches
        ]
        self.kv_array_dtypes: list = [a.dtype for a in flat_kv_caches]
        self.kv_array_shardings: list = [a.sharding for a in flat_kv_caches]
        self.kv_array_host_shardings: list = [
            jax.sharding.NamedSharding(s.mesh,
                                       s.spec,
                                       memory_kind='pinned_host')
            for s in self.kv_array_shardings
        ]

        # Attention layers share a uniform sharding spec; pick the first
        # attention kv_array's spec for the legacy attention insert path.
        self.attn_sharding_spec = next(
            (self.kv_array_shardings[layer_to_kv_array_range[layer][0]].spec
             for layer in range(len(kv_caches))
             if not self.group_is_mamba[layer_to_group_id[layer]]), None)

        logger.info(f"TPUConnectorHMA Worker --> register_runner | "
                    f"node_id={self.node_id} | ip={self.host_ip} | "
                    f"num_kv_groups={self.num_kv_groups} | "
                    f"num_kv_arrays={self.num_kv_arrays} | "
                    f"kv_transfer_port={self.kv_transfer_port}")
        self._maybe_start_p2p_server()

        # TODO(wyzhang): support D2H host-pool for HMA. Currently not
        # supported (see the if-branch in `_prepare_kv_and_wait` — a
        # D2H-enabled send path raises there).
        self.host_kv_pool = None

    def process_send_load(self, metadata: TPUConnectorMetadata):
        # P
        reqs = metadata.reqs_to_send
        if reqs:
            assert self.is_producer
            logger.info(f"TPUConnectorHMA Worker {self.node_id} --> "
                        f"reqs_to_send={reqs}")
        for req_id, req_meta in reqs.items():
            self._prepare_kv_and_wait(req_id, req_meta)

        # D
        reqs = metadata.reqs_to_load
        if reqs:
            assert not self.is_producer
            logger.info(f"TPUConnectorHMA Worker {self.node_id} --> "
                        f"reqs_to_load={reqs}")
        for req_id, req_meta in reqs.items():
            if req_meta.remote_block_ids is not None:
                # Pull

                # Build one index array per group. JAX must see the same
                # device_array call across nodes for collective consistency.
                # TODO(xiang): pad block_ids to avoid recompilation
                indices_per_group = [
                    device_array(self.mesh, np.array(ids))
                    for ids in req_meta.local_block_ids
                ]
                conn = self._maybe_build_kv_connection(req_meta)
                self.reqs_pulling[req_id] = self.pull_executor.submit(
                    self._pull_kv, req_id, conn, req_meta, indices_per_group)
            else:
                # Insert
                if req_id in self.reqs_ready_to_insert:
                    kv, indices_per_group, block_ids_per_group = \
                        self.reqs_ready_to_insert.pop(req_id)
                    has_blocks = any(
                        len(ids) > 0 for ids in block_ids_per_group)
                    if has_blocks:
                        self.runner.kv_caches = _insert_kv_chunks_per_group(
                            self.runner.kv_caches,
                            kv,
                            block_ids_per_group,
                            self.layer_to_group_id,
                            self.layer_to_kv_array_range,
                            self.num_kv_groups,
                            self.mesh,
                            self.attn_sharding_spec,
                        )
                # Notify P so it can free the buffer.
                socket = self._maybe_build_notif_socket(req_meta)
                self._notify_pull_done(socket, req_id, req_meta.uuid)

    def _prepare_kv_and_wait(self, req_id: str, req_meta: SendMeta):
        local_block_ids = req_meta.local_block_ids  # list[list[int]]
        kv = _select_from_kv_caches_per_group(
            self.runner.kv_caches,
            local_block_ids,
            self.layer_to_group_id,
        )
        if dist_utils.get_enable_d2h_transfer() and not self.multi_host:
            # TODO(wyzhang): Generalize HostKVPool to per-flat-array shapes
            # so HMA can use D2H and relieve HBM pressure on P under high
            # concurrency. HostKVPool currently assumes uniform per-layer
            # shape (one `cache_inner_shape` + `num_layers` + `dtype`),
            # which doesn't hold for hybrid models — Mamba layers carry a
            # tuple of state arrays (SSM + conv) with different shapes, and
            # block semantics differ per group (1 state-block vs. many
            # token-blocks).
            raise AssertionError(
                "D2H host-pool transfer is not yet supported by "
                "TPUConnectorHMA. Set TPU_ENABLE_D2H_TRANSFER=false in the "
                "env (it defaults to true) to use the device-resident path. "
                "See TODO(wyzhang) in tpu_connector_hma.py.")
        else:
            buffer_idx = -1
            # NOTE(xiang): We need to manually store the kv because:
            # Although we can set use_raw_buffers=True to let kv be safely
            # destroyed after calling await_pull, it could be a stranding
            # buffer if D never pulls it. So we have to set
            # use_raw_buffers=False and store the kv, then the kv buffer
            # will be safely destroyed by either D notifying or expiration.
            self.reqs_wait_pull[req_id] = [
                kv, req_meta.expiration_time, buffer_idx
            ]
            self.kv_pull_uuid_to_req_id_map[req_meta.uuid] = req_id
            self.kv_transfer_server.await_pull(req_meta.uuid, kv)
            self.stats_num_sends += 1
            self.stats_bytes_sent += sum(k.nbytes for k in kv)
            if self.stats_num_sends % self.stats_log_interval == 0:
                logger.info(
                    f"TPUConnectorHMA Worker {self.node_id} --> stats | "
                    f"cumulative sends={self.stats_num_sends} "
                    f"bytes={self.stats_bytes_sent}")

    def _pull_kv(self, req_id: str, conn: Any, req_meta: LoadMeta,
                 indices_per_group: list[jax.Array]):
        local_block_ids = req_meta.local_block_ids  # list[list[int]]
        remote_block_ids = req_meta.remote_block_ids  # list[list[int]]
        assert len(local_block_ids) == len(remote_block_ids)
        for g, (lids, rids) in enumerate(zip(local_block_ids,
                                             remote_block_ids)):
            assert len(lids) == len(rids), (
                f"Group {g}: local blocks {len(lids)} != "
                f"remote blocks {len(rids)}")

        num_blocks_per_group = [len(ids) for ids in remote_block_ids]
        kv_spec = self._get_kv_spec_hybrid(num_blocks_per_group)
        logger.info(f"Worker {self.node_id} --> kv transfer | start pull "
                    f"req_id={req_id} | uuid={req_meta.uuid}")
        start_time = time.perf_counter()
        kv = conn.pull(req_meta.uuid, kv_spec)
        kv_size_mb = sum(k.nbytes for k in kv) / (1024 * 1024)
        end_time_0, end_time_1 = time.perf_counter(), None
        if dist_utils.get_enable_block_kv_transfer():
            while True:
                end_time_1 = time.perf_counter()
                if all(chunk.is_ready() for chunk in kv) or \
                        end_time_1 - end_time_0 > \
                        dist_utils.get_p2p_wait_pull_timeout():
                    break
                time.sleep(0.001)

        prepare_time_ms = (end_time_0 - start_time) * 1000
        pull_time_ms = ((end_time_1 - end_time_0) *
                        1000 if end_time_1 is not None else 0.0)
        logger.info(
            f"Worker {self.node_id} --> kv transfer | done pull "
            f"req_id={req_id} | uuid={req_meta.uuid} | "
            f"prepare time={prepare_time_ms:.2f}ms | "
            f"pull time={pull_time_ms:.2f}ms | size={kv_size_mb:.2f}MB")
        self.stats_num_pulls += 1
        self.stats_bytes_pulled += sum(k.nbytes for k in kv)
        if self.stats_num_pulls % self.stats_log_interval == 0:
            logger.info(
                f"TPUConnectorHMA Worker {self.node_id} --> stats | "
                f"cumulative pulls={self.stats_num_pulls} "
                f"bytes={self.stats_bytes_pulled}")
        return kv, indices_per_group, local_block_ids

    def _get_kv_spec_hybrid(
            self,
            num_blocks_per_group: list[int]) -> list[jax.ShapeDtypeStruct]:
        """Build the pull spec, one ShapeDtypeStruct per kv cache jax array"""
        specs = []
        for idx in range(self.num_kv_arrays):
            group_id = self.kv_array_to_group_id[idx]
            num_blocks = num_blocks_per_group[group_id]
            shape = copy.copy(self.kv_array_shapes[idx])
            assert num_blocks <= shape[0], (
                f"Requested {num_blocks} blocks but flat layer {idx} only has "
                f"{shape[0]}")
            shape[0] = num_blocks
            specs.append(
                jax.ShapeDtypeStruct(shape,
                                     self.kv_array_dtypes[idx],
                                     sharding=self.kv_array_shardings[idx]))
        return specs


def _flatten_kv_caches(kv_caches: list) -> list[jax.Array]:
    """Flatten kv caches by expanding tuple into individual arrays"""
    flat_kv_caches: list[jax.Array] = []
    for cache in kv_caches:
        if isinstance(cache, tuple):
            flat_kv_caches.extend(cache)
        else:
            flat_kv_caches.append(cache)
    return flat_kv_caches


def _select_from_kv_caches_per_group(
    kv_caches: list,
    block_ids_per_group: list[list[int]],
    layer_to_group_id: list[int],
) -> list[jax.Array]:
    """Read blocks specified by the per-kv-cache-group ids.

    Returns a flat list of arrays to be transfered. Mamba tuples (i.e. 
    Mamba block containing two arrays) will be expanded.
    """
    indices_per_group = [
        jnp.asarray(ids, dtype=jnp.int32) for ids in block_ids_per_group
    ]
    selected: list[jax.Array] = []
    for layer, cache in enumerate(kv_caches):
        group_id = layer_to_group_id[layer]
        indices = indices_per_group[group_id]
        if isinstance(cache, tuple):
            for state in cache:
                selected.append(state.at[indices].get())
        else:
            selected.append(cache.at[indices].get())
    return selected


@functools.partial(jax.jit, donate_argnums=(0, ))
def _mamba_scatter_set(state: jax.Array, block_ids: jax.Array,
                       new_slice: jax.Array) -> jax.Array:
    """In-place scatter into a Mamba state buffer.

    `donate_argnums=(0,)` lets XLA reuse `state`'s HBM buffer for the result
    instead of allocating a fresh full-sized array. Without donation this
    OOMs on large state pools when called per-layer."""
    return state.at[block_ids].set(new_slice)


def _insert_kv_chunks_per_group(
    kv_caches: list,
    kv_slices: list[jax.Array],
    block_ids_per_group: list[list[int]],
    layer_to_group_id: list[int],
    layer_to_kv_array_range: list[tuple[int, int]],
    num_groups: int,
    mesh: jax.sharding.Mesh,
    sharding_spec,
) -> list:
    """Write received KV slices back into kv_caches for each group."""
    updated: list = list(kv_caches)

    for group_id in range(num_groups):
        block_ids = block_ids_per_group[group_id]
        assert block_ids, (f"group {group_id} has empty block_ids in "
                           f"{block_ids_per_group}")

        mamba_layer_indices: list[int] = []
        mamba_kv_slice_ranges: list[tuple[int, int]] = []
        attn_layer_indices: list[int] = []
        attn_kv_slice_indices: list[int] = []

        for layer, cache in enumerate(kv_caches):
            if layer_to_group_id[layer] != group_id:
                continue
            start, end = layer_to_kv_array_range[layer]
            if isinstance(cache, tuple):
                mamba_layer_indices.append(layer)
                mamba_kv_slice_ranges.append((start, end))
            else:
                attn_layer_indices.append(layer)
                attn_kv_slice_indices.append(start)

        for layer, (start, end) in zip(mamba_layer_indices,
                                       mamba_kv_slice_ranges):
            new_states = []
            for state, kv_slice in zip(kv_caches[layer], kv_slices[start:end]):
                new_states.append(
                    _mamba_scatter_set(
                        state,
                        jnp.asarray(block_ids, dtype=jnp.int32),
                        kv_slice,
                    ))
            updated[layer] = tuple(new_states)

        if attn_layer_indices:
            attn_kv_caches = [kv_caches[i] for i in attn_layer_indices]
            attn_kv_slices = [kv_slices[s] for s in attn_kv_slice_indices]
            updated_attn = insert_kv_chunks(attn_kv_caches, attn_kv_slices,
                                            block_ids, mesh, sharding_spec)
            for layer, new_arr in zip(attn_layer_indices, updated_attn):
                updated[layer] = new_arr

    return updated
