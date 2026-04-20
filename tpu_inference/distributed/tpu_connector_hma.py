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
from tpu_inference.distributed.host_kv_pool_hma import HostKVPoolHMA
from tpu_inference.distributed.kv_transfer import copy_to_host
from tpu_inference.distributed.tpu_connector import (
    LoadMeta, SendMeta, TPUConnector, TPUConnectorMetadata,
    TPUConnectorScheduler, TPUConnectorWorker, get_uuid, insert_kv_chunks)
from tpu_inference.distributed.transfer_stats import TransferStats
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
            "Unexpected call to TPUConnectorHMA.request_finished. "
            "TPUConnectorHMA implements SupportsHMA and only expects "
            "`request_finished_all_groups` to be invoked.")


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
        load_meta = self.reqs_to_load[request.request_id]
        logger.info(
            f"TPUConnectorHMAScheduler D --> load queued | "
            f"req_id={request.request_id} | uuid={load_meta.uuid} | "
            f"remote_host={load_meta.remote_host} | "
            f"remote_port={load_meta.remote_port} | "
            f"pending_loads={len(self.reqs_to_load)}")


    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
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
            list(block_ids_one_group) for block_ids_one_group in block_ids
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
            logger.info(
                f"TPUConnectorHMAScheduler P --> send queued | "
                f"req_id={request.request_id} | uuid={uuid} | "
                f"num_prompt_tokens={len(request.prompt_token_ids)} | "
                f"num_computed_tokens={request.num_computed_tokens} | "
                f"pending_sends={len(self.reqs_to_send)}")
        else:
            kv_transfer_params = {}
        return delay_free_blocks, kv_transfer_params

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        raise AssertionError(
            "Unexpected call to TPUConnectorHMAScheduler.request_finished. "
            "This scheduler only expects `request_finished_all_groups` "
            "to be dispatched for SupportsHMA connectors.")    

class TPUConnectorHMAWorker(TPUConnectorWorker):

    def register_runner(self, runner: TPUModelRunner):
        self.node_id = runner.topology_order_id
        self.runner = runner
        self.mesh = runner.mesh
        self.stats = TransferStats(
            node_id=self.node_id,
            log_prefix=f"TPUConnectorHMA "
            f"{'P' if self.is_producer else 'D'}")

        self.num_groups = len(runner.kv_cache_config.kv_cache_groups)
        # Mapping of kv cache group id to whether it is mamba or full attn
        self.group_is_mamba: list[bool] = [
            isinstance(g.kv_cache_spec, MambaSpec)
            for g in runner.kv_cache_config.kv_cache_groups
        ]

        # Build layer id to kv cache group id mapping
        kv_caches = runner.kv_caches
        layer_to_group_id: list[int] = [0] * len(kv_caches)
        for group_id, group in enumerate(
                runner.kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                assert layer_name in runner.layer_name_to_kvcache_index, (
                    f"Layer '{layer_name}' is listed in kv_cache_group "
                    f"{group_id} but has no entry in "
                    f"runner.layer_name_to_kvcache_index.")
                layer_id = runner.layer_name_to_kvcache_index[layer_name]
                layer_to_group_id[layer_id] = group_id
        self.layer_to_group_id = layer_to_group_id


        # Flatten kv cache, since Mamba layer kv cache is a tuple
        # of (ssm, conv)
        leaves, treedef = jax.tree_util.tree_flatten(kv_caches)
        self.kv_cache_treedef = treedef
        self.num_kv_arrays: int = len(leaves)
        self.kv_array_shapes: list[list[int]] = [list(a.shape) for a in leaves]
        self.kv_array_dtypes: list = [a.dtype for a in leaves]
        self.kv_array_shardings: list = [a.sharding for a in leaves]
        self.kv_array_host_shardings: list = [
            jax.sharding.NamedSharding(s.mesh,
                                       s.spec,
                                       memory_kind='pinned_host')
            for s in self.kv_array_shardings
        ]

        # Build mapping from flat kv cache index to group id
        group_annot = [
            jax.tree_util.tree_map(lambda _, gid=layer_to_group_id[layer]: gid,
                                   cache)
            for layer, cache in enumerate(kv_caches)
        ]
        self.kv_array_to_group_id, _ = jax.tree_util.tree_flatten(group_annot)

        # Attention layers should share the same sharding spec.
        # Pick the first as attention sharding spec.
        attn_specs = [
            self.kv_array_shardings[i].spec
            for i in range(self.num_kv_arrays)
            if not self.group_is_mamba[self.kv_array_to_group_id[i]]
        ]
        assert all(s == attn_specs[0] for s in attn_specs), (
            f"Expected uniform sharding spec across attention kv arrays, "
            f"got {attn_specs}")
        self.attn_sharding_spec = attn_specs[0] if attn_specs else None

        # Build D2H host pool.
        self.host_kv_pool: Optional[HostKVPoolHMA] = None
        if (self.is_producer and dist_utils.get_enable_d2h_transfer()
                and not self.multi_host):
            max_blocks_per_group = self._compute_max_blocks_per_group()
            kv_array_inner_shapes = [
                tuple(s[1:]) for s in self.kv_array_shapes
            ]
            kv_array_max_blocks = [
                max_blocks_per_group[self.kv_array_to_group_id[i]]
                for i in range(self.num_kv_arrays)
            ]
            self.host_kv_pool = HostKVPoolHMA(
                pool_size=dist_utils.get_max_host_kv_buffer_size(),
                per_array_max_blocks=kv_array_max_blocks,
                per_array_inner_shape=kv_array_inner_shapes,
                per_array_dtype=self.kv_array_dtypes,
                per_array_host_sharding=self.kv_array_host_shardings,
            )
            
        self._maybe_start_p2p_server()    
        role = 'P' if self.is_producer else 'D'
        logger.info(
            f"TPUConnectorHMA Worker {self.node_id} {role} --> init | "
            f"ip={self.host_ip} | multi_host={self.multi_host} | "
            f"kv_transfer_port={self.kv_transfer_port} | "
            f"num_layers={len(kv_caches)} | "
            f"num_kv_groups={self.num_groups} | "
            f"group_is_mamba={self.group_is_mamba} | "
            f"layer_to_group_id={self.layer_to_group_id} | "
            f"num_kv_arrays={self.num_kv_arrays} | "
            f"host_kv_pool_enabled={self.host_kv_pool is not None}")

    def _compute_max_blocks_per_group(self) -> list[int]:
        """Compute max num of blocks per request per kv cache group."""
        block_size = self.vllm_config.cache_config.block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        attn_max_blocks = max_model_len // block_size
        return [(1 if is_mamba else attn_max_blocks)
                for is_mamba in self.group_is_mamba]

    def process_send_load(self, metadata: TPUConnectorMetadata):
        # P
        reqs = metadata.reqs_to_send
        if reqs:
            assert self.is_producer
            logger.info(
                f"TPUConnectorHMA Worker {self.node_id} P --> step send | "
                f"num_reqs={len(reqs)}")
        for req_id, req_meta in reqs.items():
            self._prepare_kv_and_wait(req_id, req_meta)

        # D
        reqs = metadata.reqs_to_load
        if reqs:
            assert not self.is_producer
            logger.info(
                f"TPUConnectorHMA Worker {self.node_id} D --> step load | "
                f"num_reqs={len(reqs)}")
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
                            self.kv_cache_treedef,
                            self.num_groups,
                            self.mesh,
                            self.attn_sharding_spec,
                        )
                # Notify P so it can free the buffer.
                socket = self._maybe_build_notif_socket(req_meta)
                self._notify_pull_done(socket, req_id, req_meta.uuid)

    def _prepare_kv_and_wait(self, req_id: str, req_meta: SendMeta):
        local_block_ids = req_meta.local_block_ids
        assert isinstance(local_block_ids, list) and all(
            isinstance(g, list) for g in local_block_ids), (
                f"Expected list[list[int]] (per-kv-cache-group) for "
                f"local_block_ids in HMA; got {local_block_ids!r}")

        kv = _select_from_kv_caches_per_group(
            self.runner.kv_caches,
            local_block_ids,
            self.layer_to_group_id,
        )
        if dist_utils.get_enable_d2h_transfer() and not self.multi_host:
            self.kv_d2h_executor.submit(self._async_d2h_and_transfer, req_id,
                                        req_meta, kv, local_block_ids)
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
            self.stats.increment_send(sum(k.nbytes for k in kv))

    def _async_d2h_and_transfer(self, req_id: str, req_meta: SendMeta,
                                kv_src: list[jax.Array],
                                local_block_ids: list[list[int]]):
        buffer_idx, dest_buffer = self.host_kv_pool.get_buffer()

        start_time = time.perf_counter()
        sliced_dest_buffer = []
        for idx, array in enumerate(dest_buffer):
            group_id = self.kv_array_to_group_id[idx]
            num_blocks = len(local_block_ids[group_id])
            sliced_dest_buffer.append(jax.lax.slice_in_dim(array, 0, num_blocks))
        end_slice_time = time.perf_counter()

        updated_dest_buffer = []
        for idx, (src, dest) in enumerate(zip(kv_src, sliced_dest_buffer)):
            updated_dest = copy_to_host(
                src=src,
                dest=dest,
                mesh=self.mesh,
                sharding_spec=self.kv_array_shardings[idx].spec)
            updated_dest_buffer.append(updated_dest)

        while True:
            end_copy_time = time.perf_counter()
            if all(chunk.is_ready() for chunk in updated_dest_buffer) or \
                    end_copy_time - end_slice_time > dist_utils.get_p2p_wait_pull_timeout():
                break
            time.sleep(0.001)

        self.reqs_wait_pull[req_id] = [
            dest_buffer, req_meta.expiration_time, buffer_idx
        ]
        self.kv_pull_uuid_to_req_id_map[req_meta.uuid] = req_id
        self.kv_transfer_server.await_pull(req_meta.uuid, updated_dest_buffer)

        total_bytes = sum(b.nbytes for b in updated_dest_buffer)
        logger.info(
            f"TPUConnectorHMA Worker {self.node_id} P --> send d2h done | "
            f"req_id={req_id} | uuid={req_meta.uuid} | "
            f"slice_ms={(end_slice_time-start_time)*1000:.2f} | "
            f"copy_ms={(end_copy_time-end_slice_time)*1000:.2f} | "
            f"bytes={total_bytes}")
        self.stats.increment_send(total_bytes)

    def _pull_kv(self, req_id: str, conn: Any, req_meta: LoadMeta,
                 indices_per_group: list[jax.Array]):
        local_block_ids = req_meta.local_block_ids
        remote_block_ids = req_meta.remote_block_ids
        for name, block_ids in (("local_block_ids", local_block_ids),
                                ("remote_block_ids", remote_block_ids)):
            assert isinstance(block_ids, list) and all(
                isinstance(g, list) for g in block_ids), (
                    f"Expected list[list[int]] (per-kv-cache-group) for "
                    f"{name} in HMA; got {block_ids}")
        assert len(local_block_ids) == len(remote_block_ids) and all(
            len(l) == len(r)
            for l, r in zip(local_block_ids, remote_block_ids)), (
                f"local/remote block-ids shape mismatch: "
                f"local={local_block_ids}, remote={remote_block_ids}")

        num_blocks_per_group = [len(ids) for ids in remote_block_ids]
        kv_spec = self._get_kv_spec_hybrid(num_blocks_per_group)
        start_time = time.perf_counter()
        kv = conn.pull(req_meta.uuid, kv_spec)
        kv_size_mb = sum(k.nbytes for k in kv) / (1024 * 1024)
        end_prep_time, end_pull_time = time.perf_counter(), None
        timed_out = False
        if dist_utils.get_enable_block_kv_transfer():
            timeout_s = dist_utils.get_p2p_wait_pull_timeout()
            while True:
                end_pull_time = time.perf_counter()
                if all(chunk.is_ready() for chunk in kv):
                    break
                if end_pull_time - end_prep_time > timeout_s:
                    timed_out = True
                    break
                time.sleep(0.001)

        prepare_time_ms = (end_prep_time - start_time) * 1000
        pull_time_ms = ((end_pull_time - end_prep_time) *
                        1000 if end_pull_time is not None else 0.0)
        total_bytes = sum(k.nbytes for k in kv)
        if timed_out:
            ready_flags = [bool(chunk.is_ready()) for chunk in kv]
            pending_idx = [i for i, r in enumerate(ready_flags) if not r]
            logger.error(
                f"TPUConnectorHMA Worker {self.node_id} D --> pull timeout | "
                f"req_id={req_id} | uuid={req_meta.uuid} | "
                f"timeout_s={timeout_s} | bytes={total_bytes} | "
                f"ready={sum(ready_flags)}/{len(ready_flags)} | "
                f"pending_idx={pending_idx[:20]}"
                f"{'...' if len(pending_idx) > 20 else ''}")
        else:
            logger.info(
                f"TPUConnectorHMA Worker {self.node_id} D --> pull done | "
                f"req_id={req_id} | uuid={req_meta.uuid} | "
                f"prepare_ms={prepare_time_ms:.2f} | "
                f"pull_ms={pull_time_ms:.2f} | bytes={total_bytes}")
        self.stats.increment_pull(total_bytes)
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

def _select_from_kv_caches_per_group(
    kv_caches: list,
    block_ids_per_group: list[list[int]],
    layer_to_group_id: list[int],
) -> list[jax.Array]:
    """Read blocks specified by `block_ids_per_group`.

    Returns a flat list of arrays to be transfered. Mamaba 
    layer kv cache is a tuple of arrays and will be flattened.
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
    kv_treedef,
    num_groups: int,
    mesh: jax.sharding.Mesh,
    sharding_spec,
) -> list:
    """Write received KV slices back into kv_caches for each group.

    `kv_slices` is a flat list of arrays (the pull result). It is
    reassembled via `kv_treedef` into the same pytree shape as
    `kv_caches` so per-layer access is single array (attention) or
    tuple (Mamba) — no manual flat-range bookkeeping.
    """
    sliced_tree = jax.tree_util.tree_unflatten(kv_treedef, kv_slices)
    updated: list = list(kv_caches)

    for group_id in range(num_groups):
        block_ids = block_ids_per_group[group_id]
        assert block_ids, (f"group {group_id} has empty block_ids in "
                           f"{block_ids_per_group}")

        mamba_layer_indices: list[int] = []
        attn_layer_indices: list[int] = []
        for layer, cache in enumerate(kv_caches):
            if layer_to_group_id[layer] != group_id:
                continue
            if isinstance(cache, tuple):
                mamba_layer_indices.append(layer)
            else:
                attn_layer_indices.append(layer)

        for layer in mamba_layer_indices:
            new_states = []
            for state, kv_slice in zip(kv_caches[layer], sliced_tree[layer]):
                new_states.append(
                    _mamba_scatter_set(
                        state,
                        jnp.asarray(block_ids, dtype=jnp.int32),
                        kv_slice,
                    ))
            updated[layer] = tuple(new_states)

        if attn_layer_indices:
            attn_kv_caches = [kv_caches[i] for i in attn_layer_indices]
            attn_kv_slices = [sliced_tree[i] for i in attn_layer_indices]
            updated_attn = insert_kv_chunks(attn_kv_caches, attn_kv_slices,
                                            block_ids, mesh, sharding_spec)
            for layer, new_arr in zip(attn_layer_indices, updated_attn):
                updated[layer] = new_arr

    return updated
