"""

This script simulates the "mult-host disagg KV transfer" on a single host VM.

For a TPU VM which has 8 chips, we split the chips into 2 groups,
each group takes 4 chips. The script simulates 2 hosts using the 2 groups:

- Group-1: The prefill worker running on host-1 with 4 chips.
- Group-2: The decode worker running on host-2 with another 4 chips.

Each worker runs one process on its host, again only 4 chips are visibile to
the process.
- The prefill worker creates a KV array sharded on 4 chips, launches a P2P
  transfer server, then waits for the data pulling.
- The decode worker also launches a P2P transfer server, builds a connection
  with the prefill's server, then pulls the KV array from it.
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

GCE_TPU_ACCELERATOR_ENDPOINT = (
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/")
GCE_TPU_HEADERS = {"Metadata-Flavor": "Google"}

TPU_CHIPS_PER_PROCESS_BOUNDS = "TPU_CHIPS_PER_PROCESS_BOUNDS"
TPU_PROCESS_BOUNDS = "TPU_PROCESS_BOUNDS"
TPU_VISIBLE_CHIPS = "TPU_VISIBLE_CHIPS"


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
    return jax.make_mesh(
        (sharding_size, ),
        ("model", ),
        axis_types=(jax.sharding.AxisType.Auto, ) * len(("model", )),
    )


def get_kv_spec(mesh: Mesh) -> list[int]:
    num_blocks = 10
    block_size = 32
    num_kv_heads = 8
    head_dim = 128

    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa
    shape = rpa.get_kv_cache_shape(num_blocks, block_size, num_kv_heads,
                                   head_dim, jnp.bfloat16)
    dtype = jnp.bfloat16
    sharding = NamedSharding(mesh, P(None, None, "model"))
    num_layers = 4
    return [jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)] * num_layers


def get_mean_std(x: list[jax.Array]) -> tuple[float, float]:
    mean = jax.tree.reduce(jnp.add, jax.tree.map(jnp.mean, x)).tolist()
    std = jax.tree.reduce(jnp.add, jax.tree.map(jnp.std, x)).tolist()
    return mean, std


def prefill_worker(squeue: multiprocessing.Queue):

    def log(s):
        print(f"Prefill --> {s}")

    log("Start")
    # Slice 4 chips and make them only visible to this process
    # NOTE: We don't use McJAX to initialize, because wer are actually
    #       simulating multi-host on single-host, need to split the chips manually.
    # NOTE: The chips must be bounds="1,4,1", visible="0,1,2,3".
    #       But the physical connection is:
    #       0, 2, 4, 6
    #       1, 3, 5, 7
    #       bounds="2,2,1", visible="0,1,2,3", error!
    #       bounds="1,4,1", visible="0,2,4,6", error!
    os.environ[TPU_CHIPS_PER_PROCESS_BOUNDS] = "1,4,1"
    os.environ[TPU_PROCESS_BOUNDS] = "1,1,1"
    os.environ[TPU_VISIBLE_CHIPS] = "0,1,2,3"

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

    s = start_transfer_server(
        jax.local_devices()[0].client,
        '127.0.0.1:7080',
        ['127.0.0.1:0'],
        use_raw_buffers=False,
    )
    log(f"Launched server on {s.address()}")

    uuid = get_uuid()
    s.await_pull(uuid, kv)
    log("Awaiting pull...")

    # If set use_raw_buffers=True, kv can be safely deleted here
    # jax.tree.map(lambda x: x.delete(), kv); del kv

    # Wait until kv pulled by D
    assert squeue.get() == 1
    log("Done")


def decode_worker(squeue: multiprocessing.Queue):

    def log(s):
        print(f"Decode --> {s}")

    log("Start")
    os.environ[TPU_CHIPS_PER_PROCESS_BOUNDS] = "1,4,1"
    os.environ[TPU_PROCESS_BOUNDS] = "1,1,1"
    os.environ[TPU_VISIBLE_CHIPS] = "4,5,6,7"

    mesh = get_mesh()
    log(f"local={jax.local_device_count()} | global={jax.device_count()} | mesh={mesh}"
        )

    kv_spec = get_kv_spec(mesh)

    s = start_transfer_server(
        jax.local_devices()[0].client,
        '127.0.0.1:7081',
        ['127.0.0.1:0'],
    )
    server_addr = s.address()
    log(f"Launched server on {server_addr}")

    prefill_addr = "127.0.0.1:7080"
    conn = s.connect(prefill_addr)
    log(f"Created connection with {prefill_addr}")

    log("Pulling...")
    uuid = get_uuid()
    kv = conn.pull(uuid, kv_spec)

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

    # Use this queue to send pulling complete signal.
    squeue = multiprocessing.Queue()
    # NOTE: Must be "fork" otherwise will be jax coredump during start_transfer_server
    prefill = multiprocessing.get_context("fork").Process(
        target=prefill_worker, args=(squeue, ))
    decode = multiprocessing.get_context("fork").Process(target=decode_worker,
                                                         args=(squeue, ))

    prefill.start()
    decode.start()

    decode.join()
    prefill.join()


if __name__ == "__main__":
    main()
