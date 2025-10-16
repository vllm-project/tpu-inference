from typing import Any, Optional

import jax
from jax.experimental import transfer

BASE_JAX_PORT = 5000


class GroupCoordinator:
    """
    """
    rank_in_group: int
    world_size: int
    transfer_server: Optional[Any]
    connection: Optional[Any]

    def __init__(self, rank_in_group: int, world_size: int):
        self.rank_in_group = rank_in_group
        self.world_size = world_size
        self.transfer_server = None
        self.connection = None

    def send_tensor_dict(self, uuid: int, tensor_dict: dict[str, jax.Array]):
        self.transfer_server.await_pull(uuid, tensor_dict)

    def recv_tensor_dict(self, uuid: int,
                         tensor_spec: dict[str, jax.ShapeDtypeStruct]):
        return self.connection.pull(uuid, tensor_spec)

    @property
    def is_first_rank(self):
        return self.rank_in_group == 0

    @property
    def is_last_rank(self):
        return self.rank_in_group == self.world_size - 1


def init_pp_distributed_environment(ip: str, rank: int, world_size: int,
                                    device: Any, need_pp: bool):
    global _PP
    _PP = GroupCoordinator(rank, world_size)
    if need_pp:
        port_number = BASE_JAX_PORT + rank
        server_address = f"{ip}:{port_number}"
        transfer_server = transfer.start_transfer_server(
            device.client, server_address, [f"{ip}:0", f"{ip}:0"])
        _PP.transfer_server = transfer_server


def connect(prev_ip: str, prev_rank: int):
    prev_port_number = BASE_JAX_PORT + prev_rank
    connection = _PP.transfer_server.connect(f'{prev_ip}:{prev_port_number}')
    _PP.connection = connection


def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, (
        "pipeline model parallel group is not initialized")
    return _PP
