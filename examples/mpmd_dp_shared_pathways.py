"""MPMD data-parallel on Pathways via the Shared Pathways Service (multi-tenancy).

Two independent JAX clients, each in its own *spawned* process, connect to the
same remote Pathways resource manager (RM). The RM hands each client its own
non-overlapping TPU subslice. Each client then runs a data-parallel workload
sharded across the devices of its own subslice.

Contrast with examples/mpmd_dp.py:
  - mpmd_dp.py uses ONE client and manually splits jax.devices()[:8] / [8:].
    It also forks AFTER initializing JAX, which crashes the children.
  - Here each process is a SEPARATE client. `jax.devices()` inside a
    connect() context returns only that client's subslice. JAX is initialized
    *inside* connect() (never in the parent), and `spawn` gives each child a
    clean interpreter -- so there is no fork-after-JAX-init corruption.

Requires:  pathwaysutils >= 0.1.6   (pip install pathways-utils==0.1.6)

Run:
  python examples/mpmd_dp_shared_pathways.py \
    --tpu_type=<tpuv7x:NxM> --tpu_count=1
  (cluster/project/region/bucket/RM-address default to the values below.)
  
  
Before re-running, make sure the RM exists:

kubectl get pods | grep pathways-cluster

If nothing's there, deploy it first (kubectl apply -f scripts/pathways/pathways_rm_shared.yaml) — the script's _validate_tpu_supported patch won't help if there's no RM to
connect to.

Then:

python examples/mpmd_dp_shared_pathways.py

"""

import multiprocessing
import os
import random
import re
import string
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from absl import app, flags
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from pathwaysutils.experimental.shared_pathways_service import isc_pathways


def _run_dp_workload(name: str, config: dict) -> None:
    """One JAX client: connect to the RM, get a subslice, run a DP workload."""
    # isc_pathways defaults the proxy k8s Job name to "isc-proxy-$USER-xxxxx".
    # On GCP VMs $USER looks like "wenxindong_google_com"; underscores are
    # illegal in Kubernetes resource names (RFC 1123). Pass a sanitized, unique
    # name explicitly so the proxy Job applies cleanly.
    user = re.sub(r"[^a-z0-9-]", "-", os.environ.get("USER", "user").lower())
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits,
                                    k=5))
    proxy_job_name = f"isc-proxy-{user}-{suffix}"
    print(f"[{name}] proxy job name: {proxy_job_name}", flush=True)

    with isc_pathways.connect(
            cluster=config["cluster"],
            project=config["project"],
            region=config["region"],
            gcs_bucket=config["gcs_bucket"],
            pathways_service=config["pathways_service"],
            expected_tpu_instances={config["tpu_type"]: config["tpu_count"]},
            proxy_job_name=proxy_job_name,
            proxy_server_image=config["proxy_server_image"],
            proxy_options=isc_pathways.ProxyOptions(
                use_insecure_credentials=False),
    ):
        # Inside connect(): jax.devices() returns ONLY this client's subslice.
        devices = jax.devices()
        print(f"[{name}] got {len(devices)} devices: "
              f"{[d.id for d in devices]}", flush=True)

        # Data-parallel mesh over this client's own subslice.
        mesh = Mesh(devices, ('dp', ))
        sharding = NamedSharding(mesh, P('dp'))

        @jax.jit
        def workload(x):
            for _ in range(50):
                x = jnp.sin(x) @ jnp.cos(x).T
            return x

        n = len(devices) * 128
        x = jax.device_put(jnp.ones((n, n)), sharding)
        y = jax.block_until_ready(workload(x))
        print(f"[{name}] done -- result sharding {y.sharding}", flush=True)


FLAGS = flags.FLAGS
flags.DEFINE_string("cluster", "wenxindong-pw-tpu7x-16", "GKE cluster name.")
flags.DEFINE_string("project", "cloud-tpu-inference-test", "GCP project ID.")
flags.DEFINE_string("region", "us-central1", "GCP region.")
flags.DEFINE_string("gcs_bucket", "gs://wenxindong-multipod-dev", "GCS bucket.")
flags.DEFINE_string("pathways_service",
                    "pathways-cluster-pathways-head-0-0.pathways-cluster:29001",
                    "Pathways resource manager address.")
flags.DEFINE_string("tpu_type", "tpu7x:2x2x1", "TPU machine type:topology. "
                    "2x2x1 = a 4-chip subslice of the 2x2x2 physical slice.")
flags.DEFINE_integer("tpu_count", 1, "TPU slices per client.")
flags.DEFINE_string(
    "proxy_server_image",
    "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest",
    "Proxy server image (must match the client's jax version).")


def main(argv: Sequence[str]) -> None:
    config = {
        "cluster": FLAGS.cluster,
        "project": FLAGS.project,
        "region": FLAGS.region,
        "gcs_bucket": FLAGS.gcs_bucket,
        "pathways_service": FLAGS.pathways_service,
        "tpu_type": FLAGS.tpu_type,
        "tpu_count": FLAGS.tpu_count,
        "proxy_server_image": FLAGS.proxy_server_image,
    }
    procs = [
        multiprocessing.Process(target=_run_dp_workload,
                                args=(f"client-{i}", config))
        for i in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("exit codes:", [p.exitcode for p in procs])


if __name__ == "__main__":
    try:
        # spawn (not fork): each child is a clean interpreter and inits JAX
        # only inside isc_pathways.connect(). Avoids fork-after-JAX-init crash.
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # already set (e.g. in a spawn child)
    app.run(main)
