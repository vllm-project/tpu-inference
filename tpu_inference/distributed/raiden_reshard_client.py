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
"""Client for the reshard control plane's request-block registration.

Raiden serves the reshard control plane -- the work-unit directory, the
request-block registry, the plan builder and receiver arming -- from its C++
reshard store. Two of the three commands this connector needs have Python
clients in ``tpu_sync.rpc.raiden_controller``: ``register_work_unit`` and
``coordinate_transfer``. Request-block registration does not, and its Python
client is a torch extension binding, so the producer's byte-span declarations
are encoded here instead.

The encoding is not a reimplementation of anything: ``ControllerRequest`` with
``COMMAND_REGISTER_REQUEST_BLOCKS`` over a 4-byte big-endian length frame is
the same envelope the other two commands use, on the same port, so this shares
``connect_socket`` with them and inherits its address parsing, IPv6 handling
and connect retry.
"""

import typing
from typing import Any

_IMPORT_ERROR = None
try:
    from tpu_sync.rpc import controller_service_pb2, raiden_service_pb2
    from tpu_sync.rpc.raiden_controller import connect_socket
except Exception as _exc:  # pylint: disable=broad-except
    controller_service_pb2 = None
    raiden_service_pb2 = None
    connect_socket = None
    _IMPORT_ERROR = _exc

# Matches the connect and read timeout the controller client uses for the
# coordinate call, which blocks for the whole transfer.
_DEFAULT_TIMEOUT_S = 300.0


class ReshardRequestBlockClient:
    """Registers a producer rank's byte-span declarations for one request.

    One instance is cheap and holds no connection; each call opens and closes
    its own socket, which is what the surrounding controller client does.
    """

    def __init__(self,
                 controller_address: str,
                 timeout_s: float = _DEFAULT_TIMEOUT_S):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "tpu_sync.rpc is unavailable, so the reshard control plane "
                f"cannot be reached: {_IMPORT_ERROR}")
        self._address = controller_address
        self._timeout_s = timeout_s

    def register_request_blocks(
        self,
        req_id: str,
        uuid: int,
        unit: Any,
        block_ids: typing.Sequence[int],
        pool_spans: typing.Sequence[Any] = (),
    ) -> None:
        """Declares which of this rank's bytes land where, for one request.

        Args:
            req_id: Request identity the plan is keyed by, together with
                ``uuid``. One destination rank per ``req_id``.
            uuid: Plan identity. A source keys its active plan by uuid, so
                each destination rank of a fan-out needs a distinct one.
            unit: The registering producer rank's ``RaidenId``.
            block_ids: The producer blocks holding this request's KV.
            pool_spans: Per-tag span entries from
                ``kv_pool_layout.build_request_span_entries``. Empty for a
                rank that owns none of this destination's heads -- the
                controller still requires it to register.
        """
        block_req = controller_service_pb2.RegisterRequestBlocksRequest(
            req_id=req_id,
            uuid=uuid,
            unit=raiden_service_pb2.RaidenIdProto(
                job_name=unit.job_name,
                job_replica_id=unit.job_replica_id,
                data_name=unit.data_name,
                data_replica_idx=unit.data_replica_idx,
            ),
            block_ids=list(block_ids),
        )
        for entry in pool_spans:
            entry_proto = block_req.pool_spans.add(
                tag=str(entry.tag),
                block_ids=[int(block_id) for block_id in entry.block_ids],
                declared_bytes=int(entry.declared_bytes),
                dst_space_version=int(getattr(entry, "dst_space_version", 0)),
            )
            for span in entry.spans:
                entry_proto.spans.add(
                    src_block_ordinal=int(span.src_block_ordinal),
                    src_offset_bytes=int(span.src_offset_bytes),
                    dst_block_index=int(span.dst_block_index),
                    dst_offset_bytes=int(span.dst_offset_bytes),
                    size_bytes=int(span.size_bytes),
                    src_stride_bytes=int(span.src_stride_bytes),
                    dst_stride_bytes=int(span.dst_stride_bytes),
                    count=int(span.count),
                )
        req = controller_service_pb2.ControllerRequest(
            command=controller_service_pb2.ControllerRequest.
            COMMAND_REGISTER_REQUEST_BLOCKS,
            register_request_blocks_request=block_req,
        )
        self._send(req)

    def _send(self, req: Any) -> Any:
        sock = connect_socket(self._address, timeout=self._timeout_s)
        try:
            payload = req.SerializeToString()
            sock.sendall(len(payload).to_bytes(4, "big") + payload)
            resp_len = int.from_bytes(_recv_exactly(sock, 4), "big")
            resp = controller_service_pb2.ControllerResponse()
            resp.ParseFromString(_recv_exactly(sock, resp_len))
            if not resp.success:
                raise RuntimeError(
                    "Reshard control plane rejected register_request_blocks: "
                    f"{resp.message}")
            return resp
        finally:
            sock.close()


def _recv_exactly(sock, count: int) -> bytes:
    buf = b""
    while len(buf) < count:
        chunk = sock.recv(count - len(buf))
        if not chunk:
            raise RuntimeError(
                "Reshard control plane closed the connection after "
                f"{len(buf)} of {count} bytes")
        buf += chunk
    return buf
