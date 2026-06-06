# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raiden-engine-backed TPU KV connector for PD-disaggregated serving.

This is a drop-in replacement for ``tpu_connector.TPUConnector`` that swaps the
``jax.experimental.transfer`` P2P server + the hand-rolled ZMQ side channel for
the standalone Raiden ``RaidenTransferEngine``
(``api.jax.raiden_transfer_engine``).

Enable it by pointing vLLM's kv-transfer-config at this module, e.g.::

    --kv-transfer-config '{"kv_connector":"TPUConnector",
        "kv_connector_module_path":"tpu_inference.distributed.tpu_connector_raiden",
        "kv_role":"kv_producer"}'

and make the Raiden export importable (PYTHONPATH must contain the
``raiden_oss_0531_3`` checkout so ``api`` / ``frameworks`` resolve).

Design (see RAIDEN_ENGINE_REPLACEMENT_PLAN.md):

* The **scheduler** half is reused verbatim from ``tpu_connector`` -- it still
  mints the uuid, computes the rounded ``computed_block_ids``, and advertises
  ``{uuid, remote_block_ids, remote_host, remote_port}`` where ``remote_port`` is
  ``TPU_KV_TRANSFER_PORT``. The Raiden engine binds its control socket to that
  same fixed port, so no scheduler change is needed (Phase 2).
* The **worker** half is reimplemented on top of the engine:
    - producer  : ``engine.register_send(req_id, uuid, block_ids)``
    - consumer  : ``engine.submit_load(req_id, uuid, endpoint, remote, local)``
                  (non-blocking; does pull -> H2H -> reorder -> H2D -> ack
                  internally)
    - both      : ``engine.poll_finished() -> (done_sending, done_recving,
                  failed_recving)``
  The engine owns gather/scatter, host staging, and the done-notification
  control plane, so ``select_from_kv_caches`` / ``insert_kv_chunks`` /
  ``HostKVPool`` and the ZMQ listener are all gone.
"""

import os
from typing import Any, Optional

# IMPORTANT: import the Raiden engine .so BEFORE any tpu_inference imports.
# tpu_inference.runner.tpu_runner pulls in torch_xla/libtpu; if that loads
# before the engine .so, dlopen of _raiden_transfer_engine.so aborts with a
# tcmalloc "free invalid pointer" (two copies of the XLA/PjRt runtime). Loading
# the engine first lets them coexist. See RAIDEN_ENGINE_REPLACEMENT_PLAN.md.
try:
    from api.jax.raiden_transfer_engine import RaidenTransferEngine
    _RAIDEN_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pylint: disable=broad-except
    RaidenTransferEngine = None  # type: ignore[assignment]
    _RAIDEN_IMPORT_ERROR = exc

import jax
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics, KVConnectorStats, PromMetric, PromMetricT)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

import tpu_inference.distributed.utils as dist_utils
from tpu_inference import envs
# Reuse the scheduler half and the shared metadata/dataclasses unchanged.
from tpu_inference.distributed.tpu_connector import (LoadMeta,
                                                     TPUConnectorMetadata,
                                                     TPUConnectorScheduler)
from tpu_inference.distributed.tpu_connector_stats import (
    TpuKVConnectorPromMetrics, TpuKVConnectorStats)
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_runner import TPUModelRunner

ReqId = str

logger = init_logger(__name__)


def _get_num_slots() -> int:
    """Host staging slots = max concurrent in-flight transfers.

    Each slot reserves ``max_blocks`` blocks of pinned host memory, so total host
    staging is ``num_slots * max_blocks`` blocks. Tune via ``RAIDEN_NUM_SLOTS``.
    """
    return int(os.getenv("RAIDEN_NUM_SLOTS", "16"))


def _get_unsafe_skip_buffer_lock() -> bool:
    val = os.getenv("RAIDEN_UNSAFE_SKIP_BUFFER_LOCK", "false").lower()
    return val in ("1", "true", "yes")


class TPUConnector(KVConnectorBase_V1):
    """KVConnector entry point; same name as the JAX-transfer connector so the
    ``kv_connector`` config value is unchanged -- only ``kv_connector_module_path``
    differs."""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole,
                 kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._connector_metadata = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPUConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = RaidenTPUConnectorWorker(vllm_config)

    ############################################################
    # Scheduler Side Methods (delegated to the reused scheduler)
    ############################################################
    def get_num_new_matched_tokens(
            self, request: "Any",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Any", blocks: "Any",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> TPUConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta()

    def request_finished(
        self,
        request: "Any",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def get_finished_count(self) -> int:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_finished_count()

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: list[jax.Array]):
        # We read runner.kv_caches directly in register_runner instead; the ref
        # is reassigned (in place, via donation) during model forward and the
        # engine tracks the live physical buffer (see plan, Blocker B).
        pass

    def register_runner(self, runner: TPUModelRunner) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_runner(runner)

    def start_load_kv(self, _, **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, TPUConnectorMetadata)
        self.connector_worker.process_send_load(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, *args, **kwargs) -> None:
        pass

    def wait_for_save(self):
        pass

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
            cls,
            data: dict[str, Any] | None = None) -> KVConnectorStats | None:
        return (TpuKVConnectorStats(
            data=data) if data is not None else TpuKVConnectorStats())

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return TpuKVConnectorPromMetrics(vllm_config, metric_types, labelnames,
                                         per_engine_labelvalues)


class RaidenTPUConnectorWorker:
    """Worker half backed by ``RaidenTransferEngine``."""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer

        self.runner: TPUModelRunner = None
        self.node_id = 0
        self.multi_host = envs.TPU_MULTIHOST_BACKEND == "ray"

        self.host_ip = dist_utils.get_host_ip()
        # Bind the engine control socket to the SAME port the scheduler
        # advertises as remote_port, so the cross-process scheduler/worker
        # boundary needs no extra handshake (Phase 2).
        self.kv_transfer_port = int(dist_utils.get_kv_transfer_port())

        self.engine: Optional["RaidenTransferEngine"] = None
        # Consumer-side: req_ids for which we already issued a real submit_load,
        # so the scheduler's later remote_block_ids=None step is a no-op.
        self.submitted: set[ReqId] = set()

        logger.info(
            "RaidenTPUConnector Worker --> init | is_producer=%s | ip=%s | "
            "kv_transfer_port=%s | num_slots=%s", self.is_producer,
            self.host_ip, self.kv_transfer_port, _get_num_slots())

    def register_runner(self, runner: TPUModelRunner):
        if RaidenTransferEngine is None:
            raise ImportError(
                "RaidenTransferEngine is not importable. Add the raiden export "
                "(e.g. raiden_oss_0531_3) to PYTHONPATH so 'api.jax."
                "raiden_transfer_engine' resolves. Original error: "
                f"{_RAIDEN_IMPORT_ERROR}")

        self.runner = runner
        self.node_id = runner.topology_order_id

        kv_caches = runner.kv_caches
        block_size = self.vllm_config.cache_config.block_size
        max_blocks = self.vllm_config.model_config.max_model_len // block_size

        self.engine = RaidenTransferEngine(
            kv_caches=kv_caches,
            local_control_port=self.kv_transfer_port,
            max_blocks=max_blocks,
            num_slots=_get_num_slots(),
            timeout_s=float(dist_utils.get_p2p_wait_pull_timeout()),
            unsafe_skip_buffer_lock=_get_unsafe_skip_buffer_lock(),
        )
        logger.info(
            "RaidenTPUConnector Worker %s --> engine ready | "
            "control_port=%s | data_port=%s | max_blocks=%s", self.node_id,
            getattr(self.engine, "local_control_port", None),
            getattr(self.engine, "local_data_port", None), max_blocks)

    def _remote_endpoint(self, req_meta: LoadMeta) -> str:
        host = req_meta.remote_host
        port = req_meta.remote_port
        if isinstance(host, list):
            assert isinstance(port, list) and len(host) == len(port)
            return f"{host[self.node_id]}:{port[self.node_id]}"
        return f"{host}:{port}"

    def process_send_load(self, metadata: TPUConnectorMetadata):
        """Called before model forward each step."""
        # Producer: register the prefilled blocks so D can pull them.
        for req_id, req_meta in metadata.reqs_to_send.items():
            assert self.is_producer
            self.engine.register_send(req_id, req_meta.uuid,
                                      req_meta.local_block_ids)

        # Consumer: pull (or release-only ack).
        for req_id, req_meta in metadata.reqs_to_load.items():
            assert not self.is_producer
            remote_endpoint = self._remote_endpoint(req_meta)
            if req_meta.remote_block_ids is not None:
                if req_id in self.submitted:
                    continue
                self.submitted.add(req_id)
                self.engine.submit_load(
                    req_id=req_id,
                    uuid=req_meta.uuid,
                    remote_endpoint=remote_endpoint,
                    remote_block_ids=req_meta.remote_block_ids,
                    local_block_ids=req_meta.local_block_ids,
                )
            else:
                # remote_block_ids is None means either:
                #   (a) the async pull already finished -> the engine's
                #       submit_load already wrote KV and acked P; nothing to do.
                #   (b) full local prefix-cache hit, no pull was ever issued ->
                #       P is still holding blocks waiting; send a release-only
                #       ack (submit_load with empty blocks -> AckRemote).
                if req_id in self.submitted:
                    self.submitted.discard(req_id)
                    continue
                self.engine.submit_load(
                    req_id=req_id,
                    uuid=req_meta.uuid,
                    remote_endpoint=remote_endpoint,
                    remote_block_ids=[],
                    local_block_ids=[],
                )

    def get_finished(self) -> tuple[set[str], set[str]]:
        if self.engine is None:
            return set(), set()
        done_sending, done_recving, failed_recving = self.engine.poll_finished(
        )
        if failed_recving:
            logger.warning(
                "RaidenTPUConnector Worker %s --> failed_recving=%s",
                self.node_id, failed_recving)
        if done_sending:
            logger.info("RaidenTPUConnector Worker %s --> done_sending=%s",
                        self.node_id, done_sending)
        if done_recving:
            logger.info("RaidenTPUConnector Worker %s --> done_recving=%s",
                        self.node_id, done_recving)
        # Treat failed receives as "done" so the scheduler doesn't hang waiting;
        # the request will fall back to local recompute / error handling.
        return set(done_sending), set(done_recving) | set(failed_recving)

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        # The engine emits RAIDEN_TIMING logs directly; no aggregated stats yet.
        return None
