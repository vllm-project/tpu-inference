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
"""

This script simulates the "mult-host disagg KV transfer" on a single host VM.

For a TPU VM which has 8 chips, we split the chips into 2 groups,
each group takes 4 chips. The script simulates 8 hosts using the 2 groups:

- Group-1: The prefill worker running on host-0,1,2,3 with 4 chips.
- Group-2: The decode worker running on host-4,5,6,7 with another 4 chips.

Each worker runs 4 processes on its hosts, only 1 chip is visibile to
each process.
- The prefill worker creates a KV array sharded on 4 chips, each prefill process
  launches a P2P transfer server, then fetches the KV shard on its chip and
  waits for the data pulling.
- Each decode worker process also launches a P2P transfer server, builds a connection
  with its corresponding prefill server, pulls the KV shard from the prefill server,
  then makes the full KV.
"""

import glob
import multiprocessing
import os

import jax
import jax.numpy as jnp
import requests
from jax.experimental.transfer import start_transfer_server
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import SingleDeviceSharding

GCE_TPU_ACCELERATOR_ENDPOINT = (
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/")
GCE_TPU_HEADERS = {"Metadata-Flavor": "Google"}

TPU_CHIPS_PER_PROCESS_BOUNDS = "TPU_CHIPS_PER_PROCESS_BOUNDS"
TPU_PROCESS_BOUNDS = "TPU_PROCESS_BOUNDS"
TPU_VISIBLE_CHIPS = "TPU_VISIBLE_CHIPS"
CLOUD_TPU_TASK_ID = "CLOUD_TPU_TASK_ID"
TPU_PROCESS_ADDRESSES = "TPU_PROCESS_ADDRESSES"
TPU_PROCESS_PORT = "TPU_PROCESS_PORT"


def get_num_chips() -> int:
    accel_files = glob.glob("/dev/accel*")
    if accel_files:
        return len(accel_files)
    try:
        vfio_entries = os.listdir("/dev/vfio")
        numeric_entries = [
            int(entry) for entry in vfio_entries if entry.isdigit()
        ]
        return len(numeric_entries)
    except FileNotFoundError:
        return 0


def get_tpu_metadata(key: str = "") -> str:
    try:
        accelerator_type_request = requests.get(
            os.path.join(GCE_TPU_ACCELERATOR_ENDPOINT, key),
            headers=GCE_TPU_HEADERS,
        )
        if (accelerator_type_request.status_code == 200
                and accelerator_type_request.text):
            return accelerator_type_request.text
        else:
            print("Unable to poll TPU GCE Metadata. Got "
                  f"status code: {accelerator_type_request.status_code} and "
                  f"content: {accelerator_type_request.text}")
    except requests.RequestException as e:
        print("Unable to poll the TPU GCE Metadata: %s", e)
    return None


def get_uuid(pid: int) -> int:
    return 1189 + pid


def get_mesh() -> Mesh:
    sharding_size = jax.device_count()
    return jax.make_mesh(
        (sharding_size, ),
        ("model", ),
        axis_types=(jax.sharding.AxisType.Auto, ) * len(("model", )),
    )


def get_kv_shape(mesh: Mesh, per_shard: bool = False) -> tuple[int, ...]:
    num_blocks = 10
    block_size = 32
    num_kv_heads = 8
    head_dim = 128

    if per_shard:
        shard_size = mesh.shape["model"]
        num_kv_heads = num_kv_heads // shard_size

    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa
    shape = rpa.get_kv_cache_shape(num_blocks, block_size, num_kv_heads,
                                   head_dim, jnp.bfloat16)
    return shape


def get_kv_spec(mesh: Mesh) -> jax.ShapeDtypeStruct:
    shape = get_kv_shape(mesh)
    dtype = jnp.bfloat16
    sharding = NamedSharding(mesh, P(None, None, "model"))
    num_layers = 4
    return [jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)] * num_layers


# Hard code the shard spec, it could have been calculated.
def get_kv_shard_spec(mesh: Mesh) -> jax.ShapeDtypeStruct:
    shape = get_kv_shape(mesh, per_shard=True)
    dtype = jnp.bfloat16
    sharding = SingleDeviceSharding(jax.local_devices()[0])
    num_layers = 4
    return [jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)] * num_layers


def get_mean_std(x: list[jax.Array]) -> tuple[float, float]:
    mean = jax.tree.reduce(jnp.add, jax.tree.map(jnp.mean, x)).tolist()
    std = jax.tree.reduce(jnp.add, jax.tree.map(jnp.std, x)).tolist()
    return mean, std


def prefill_worker(pid: int, squeue: multiprocessing.Queue):

    def log(s):
        print(f"Prefill-{pid} --> {s}")

    def _get_port(pid: int):
        ports = [8476, 8477, 8478, 8479]
        return f"{ports[pid]}"

    def _get_p2p_server_addr(pid: int):
        ports = [8576, 8577, 8578, 8579]
        return f"127.0.0.1:{ports[pid]}"

    def _get_ip(pid: int):
        port = _get_port(pid)
        return f"127.0.0.1:{port}"

    def _get_ips():
        ips = [_get_ip(i) for i in range(4)]
        return ",".join(ips)

    log("Start")
    # Slice 4 chips, and assign one chip to one process while keeping
    # all 4 chips visible to each process.
    # NOTE: The process_bounds must be "2,2,1", "1,4,1" will error!
    os.environ[TPU_CHIPS_PER_PROCESS_BOUNDS] = "1,1,1"
    os.environ[TPU_PROCESS_BOUNDS] = "2,2,1"
    os.environ[TPU_VISIBLE_CHIPS] = f"{pid}"
    os.environ[CLOUD_TPU_TASK_ID] = f"{pid}"
    os.environ[TPU_PROCESS_ADDRESSES] = _get_ips()
    os.environ[TPU_PROCESS_PORT] = _get_port(pid)

    mesh = get_mesh()
    log(f"local={jax.local_device_count()} | global={jax.device_count()} | mesh={mesh}"
        )

    kv_spec = get_kv_spec(mesh)
    key = jax.random.PRNGKey(0)

    def _create_layer_kv(spec):
        return jax.device_put(
            jax.random.uniform(key, shape=spec.shape, dtype=spec.dtype),
            spec.sharding)

    kv = jax.tree.map(_create_layer_kv, kv_spec)
    mean, std = get_mean_std(kv)
    log(f"kv | shape={len(kv)} * {kv[0].shape} | sharding={kv[0].sharding} | mean={mean} | std={std}"
        )

    # Fetch the shard on this chip only.
    # NOTE: index must be 0, because each process only sees its local data.
    kv_shard = [_kv.addressable_data(0) for _kv in kv]

    s = start_transfer_server(
        jax.local_devices()[0].client,
        _get_p2p_server_addr(pid),
        ['127.0.0.1:0'],
    )
    server_addr = s.address()
    log(f"Launched server on {server_addr}")

    uuid = get_uuid(pid)
    s.await_pull(uuid, kv_shard)
    log("Awaiting pull...")

    # Wait until kv pulled by D
    assert squeue.get() == 1
    log("Done")


def decode_worker(pid: int, squeue: multiprocessing.Queue):

    def log(s):
        print(f"Decode-{pid} --> {s}")

    def _get_port(pid: int):
        ports = [8486, 8487, 8488, 8489]
        return f"{ports[pid]}"

    def _get_p2p_server_addr(pid: int):
        ports = [8586, 8587, 8588, 8589]
        return f"127.0.0.1:{ports[pid]}"

    def _get_prefill_p2p_server_addr(pid: int):
        ports = [8576, 8577, 8578, 8579]
        return f"127.0.0.1:{ports[pid]}"

    def _get_ip(pid: int):
        port = _get_port(pid)
        return f"127.0.0.1:{port}"

    def _get_ips():
        ips = [_get_ip(i) for i in range(4)]
        return ",".join(ips)

    log("Start")
    os.environ[TPU_CHIPS_PER_PROCESS_BOUNDS] = "1,1,1"
    os.environ[TPU_PROCESS_BOUNDS] = "2,2,1"
    os.environ[TPU_VISIBLE_CHIPS] = f"{pid + 4}"
    os.environ[CLOUD_TPU_TASK_ID] = f"{pid}"
    os.environ[TPU_PROCESS_ADDRESSES] = _get_ips()
    os.environ[TPU_PROCESS_PORT] = _get_port(pid)

    mesh = get_mesh()
    log(f"local={jax.local_device_count()} | global={jax.device_count()} | mesh={mesh}"
        )

    s = start_transfer_server(
        jax.local_devices()[0].client,
        _get_p2p_server_addr(pid),
        ['127.0.0.1:0'],
    )
    server_addr = s.address()
    log(f"Launched server on {server_addr}")

    prefill_addr = _get_prefill_p2p_server_addr(pid)
    conn = s.connect(prefill_addr)
    log(f"Created connection with {prefill_addr}")

    log("Pulling...")
    uuid = get_uuid(pid)
    kv_shard_spec = get_kv_shard_spec(mesh)
    kv_shard = conn.pull(uuid, kv_shard_spec)

    kv_spec = get_kv_spec(mesh)
    kv = []
    for spec, shard in zip(kv_spec, kv_shard):
        kv.append(
            jax.make_array_from_process_local_data(spec.sharding, shard,
                                                   spec.shape))

    mean, std = get_mean_std(kv)
    log(f"kv | shape={len(kv)} * {kv[0].shape} | sharding={kv[0].sharding} | mean={mean} | std={std}"
        )
    squeue.put(1)
    log("Done")


def main():
    tpu_type = get_tpu_metadata("accelerator-type")
    instance_id = get_tpu_metadata("instance-id")
    worker_id = get_tpu_metadata("agent-worker-number")
    _ = get_tpu_metadata("tpu-env")
    print(
        f"TPU_type={tpu_type} | instance_id={instance_id} | worker_id={worker_id}"
    )
    assert tpu_type == "v6e-8"

    prefills = []
    decodes = []
    for i in range(4):
        squeue = multiprocessing.Queue()
        # NOTE: Must be "fork" otherwise will be jax coredump during start_transfer_server
        prefill = multiprocessing.get_context("fork").Process(
            target=prefill_worker, args=(
                i,
                squeue,
            ))
        decode = multiprocessing.get_context("fork").Process(
            target=decode_worker, args=(
                i,
                squeue,
            ))
        prefills.append(prefill)
        decodes.append(decode)

    [prefill.start() for prefill in prefills]
    [decode.start() for decode in decodes]

    [decode.join() for decode in decodes]
    [prefill.join() for prefill in prefills]


if __name__ == "__main__":
    main()
