"""

This script simulates the mult-host disagg KV transfer on a single host VM.

For a TPU VM which has 8 chips, it splits the chips into 2 groups, each group takes 4 chips.

TODO: add more explanations
"""

import argparse
import glob
import multiprocessing
import os
from typing import List

import jax
import jax.numpy as jnp
import requests
from jax.experimental.transfer import start_transfer_server
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

GCE_TPU_ACCELERATOR_ENDPOINT = (
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/")
GCE_TPU_HEADERS = {"Metadata-Flavor": "Google"}

TPU_CHIPS_PER_PROCESS_BOUNDS = "TPU_CHIPS_PER_PROCESS_BOUNDS"
TPU_PROCESS_BOUNDS = "TPU_PROCESS_BOUNDS"
TPU_VISIBLE_CHIPS = "TPU_VISIBLE_CHIPS"

LOCAL_ADDR = '0.0.0.0:0'


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


def get_uuid() -> int:
    return 1189


def get_mesh() -> Mesh:
    sharding_size = jax.device_count()
    return jax.make_mesh((sharding_size, ), ("model"))


def get_kv_spec(mesh: Mesh) -> List[int]:
    # (num_blocks, block_size, num_kv_heads, head_dim)
    shape = (100, 32, 8, 128)
    dtype = jnp.bfloat16
    sharding = NamedSharding(mesh, P(None, None, "model", None))
    return jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)


def get_global_kv(mesh: Mesh) -> jax.Array:
    spec = get_kv_spec(mesh)
    global_kv = jax.device_put(jnp.ones(spec.shape, dtype=spec.dtype),
                               spec.sharding)
    return global_kv


def prefill_worker(num_procs: int, squeue: multiprocessing.Queue):

    def log(s):
        print(f"Prefill --> {s}")

    log("start")
    # Slice 4 chips and make them only visible to this process
    os.environ[TPU_CHIPS_PER_PROCESS_BOUNDS] = "1,4,1"
    os.environ[TPU_PROCESS_BOUNDS] = f"{num_procs},1,1"
    os.environ[TPU_VISIBLE_CHIPS] = "0,1,2,3"

    mesh = get_mesh()
    log(f"local={jax.local_device_count()} | global={jax.device_count()} | mesh={mesh}"
        )

    kv = get_global_kv(mesh)
    log(kv.shape)
    log(kv.sharding)

    uuid = get_uuid()
    s = start_transfer_server(
        jax.devices()[0].client,
        '0.0.0.0:7080',
        ['0.0.0.0:0'],
    )
    server_addr = s.address()
    log(f"Launched server on {server_addr}")
    squeue.put(s.address())

    log("Awaiting pull...")
    s.await_pull(uuid, kv)

    log("done")


def decode_worker(num_procs: int, squeue: multiprocessing.Queue):

    def log(s):
        print(f"Decode --> {s}")

    log("start")
    # Slice 4 chips and make them only visible to this process
    os.environ[TPU_CHIPS_PER_PROCESS_BOUNDS] = "1,4,1"
    os.environ[TPU_VISIBLE_CHIPS] = f"{num_procs},1,1"
    os.environ[TPU_VISIBLE_CHIPS] = "4,5,6,7"

    mesh = get_mesh()
    log(f"local={jax.local_device_count()} | global={jax.device_count()} | mesh={mesh}"
        )

    kv_spec = get_kv_spec(mesh)

    uuid = get_uuid()
    s = start_transfer_server(
        jax.devices()[0].client,
        '0.0.0.0:7081',
        ['0.0.0.0:0'],
    )
    server_addr = s.address()
    log(f"Launched server on {server_addr}")

    prefill_addr = squeue.get()
    conn = s.connect(prefill_addr)
    log(f"Created connection with {prefill_addr}")

    log("Pulling...")
    kv = conn.pull(uuid, kv_spec)
    log(kv.shape)
    log(kv.sharding)

    log("done")


def main():
    parser = argparse.ArgumentParser(
        description="A simple command-line argument parser example.", )
    parser.add_argument("--num_procs_per_worker",
                        type=int,
                        required=False,
                        default=1)
    args = parser.parse_args()

    tpu_type = get_tpu_metadata("accelerator-type")
    instance_id = get_tpu_metadata("instance-id")
    worker_id = get_tpu_metadata("agent-worker-number")
    print(
        f"TPU_type={tpu_type} | instance_id={instance_id} | worker_id={worker_id}"
    )
    assert tpu_type == "v6e-8"

    print("================ TPU environments ================")
    print(get_tpu_metadata("tpu-env"))
    print("================ TPU environments ================")

    num_proces_per_worker = args.num_procs_per_worker
    squeue = multiprocessing.Queue()
    # NOTE: Must be "fork" otherwise will be jax coredump during start_transfer_server
    prefill = multiprocessing.get_context("fork").Process(
        target=prefill_worker, args=(num_proces_per_worker, squeue))
    decode = multiprocessing.get_context("fork").Process(
        target=decode_worker, args=(num_proces_per_worker, squeue))

    prefill.start()
    decode.start()

    prefill.join()
    decode.join()


if __name__ == "__main__":
    main()
