# Copyright 2026 Google LLC
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
"""Read-only multi-host TPU slice probe.

Reports what a Buildkite agent on a multi-host queue (tpu_v7x_16_queue,
tpu_v7x_32_queue) can see about its slice: which VM of the slice it is, how many
peers the slice has, and how many TPU chips are attached to this host.

Deliberately does NOT import jax / torch_tpu and never opens a TPU device: on a
multi-host slice, initializing the runtime on one VM blocks until every peer VM
joins, which would hang the job. Chip counts here come from /dev and GCE
metadata only. The in-container device count (which does initialize the
runtime) is a separate step that runs after the Ray cluster is up.

Always exits 0 for the informational sections; it exits non-zero only if the
host does not look like a multi-host TPU VM at all, since that means the queue
is not what the pipeline assumed.
"""

import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys

METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"


def _run(cmd, timeout=60):
    """Run a command, returning (rc, stdout, stderr) and never raising."""
    try:
        p = subprocess.run(cmd,
                           capture_output=True,
                           text=True,
                           timeout=timeout,
                           check=False)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{' '.join(cmd)}: timed out after {timeout}s"


def _metadata(path):
    rc, out, _ = _run([
        "curl", "-s", "--max-time", "10", "-H", "Metadata-Flavor: Google",
        f"{METADATA_ROOT}/{path}"
    ])
    return out if rc == 0 else ""


def section(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)), flush=True)


def probe_identity():
    section("host identity")
    hostname = socket.gethostname()
    print(f"hostname            : {hostname}")
    # TPU VM hostnames are t1v-n-<slice-id>-w-<worker-index>.
    m = re.search(r"-w-(\d+)$", hostname)
    worker_index = int(m.group(1)) if m else None
    print(f"worker index        : "
          f"{worker_index if worker_index is not None else 'unknown'}"
          f"{' (Ray head)' if worker_index == 0 else ''}")
    print(
        f"buildkite agent     : {os.environ.get('BUILDKITE_AGENT_NAME', '-')}")
    print(
        f"buildkite queue     : {os.environ.get('BUILDKITE_AGENT_META_DATA_QUEUE', '-')}"
    )
    return worker_index


def probe_metadata():
    section("GCE / TPU metadata")
    zone = _metadata("instance/zone").split("/")[-1]
    tpu_name = _metadata("instance/description")
    accel = _metadata("instance/attributes/accelerator-type")
    worker_id = _metadata("instance/attributes/agent-worker-number")
    tpu_env_raw = _metadata("instance/attributes/tpu-env")
    print(f"zone                : {zone or '-'}")
    print(f"slice (description) : {tpu_name or '-'}")
    print(f"accelerator-type    : {accel or '-'}")
    print(f"agent-worker-number : {worker_id or '-'}")

    # tpu-env is a YAML-ish "KEY: 'value'" block. Only the topology keys matter
    # for sizing the slice; the rest is agent-image plumbing, so print those
    # explicitly and keep the full block out of the log.
    tpu_env = {}
    for line in tpu_env_raw.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            tpu_env[k.strip()] = v.strip().strip("'")
    for key in ("TYPE", "ACCELERATOR_TYPE", "TOPOLOGY", "WORKER_ID",
                "CHIPS_PER_HOST_BOUNDS", "HOST_BOUNDS",
                "TPU_CHIPS_PER_PROCESS_BOUNDS", "TPU_PROCESS_BOUNDS", "WRAP"):
        if key in tpu_env:
            print(f"tpu-env {key:<24}: {tpu_env[key]}")

    for var in ("TPU_ACCELERATOR_TYPE", "TPU_WORKER_ID",
                "TPU_WORKER_HOSTNAMES", "TPU_CHIPS_PER_HOST_BOUNDS",
                "TPU_HOST_BOUNDS", "TPU_TOPOLOGY", "TPU_SKIP_MDS_QUERY"):
        if var in os.environ:
            print(f"env {var:<20}: {os.environ[var]}")
    return zone, tpu_name, tpu_env


def probe_local_chips():
    section("local TPU chips (no runtime init)")
    # /dev/vfio holds one numbered group per passed-through TPU device, plus the
    # 'vfio' control node and the 'devices' directory -- neither is a chip.
    vfio_devs = sorted(d for d in glob.glob("/dev/vfio/*")
                       if os.path.basename(d).isdigit())
    accel = sorted(glob.glob("/dev/accel*"))
    print(f"/dev/vfio groups    : {len(vfio_devs)} -> "
          f"{[os.path.basename(d) for d in vfio_devs] or '-'}")
    print(f"/dev/accel entries  : {len(accel)} -> {accel or '-'}")

    # TPU functions all sit under Google's PCI vendor id 1ae0. Count by device
    # id so an unrelated Google device (gVNIC, PCI bridges) is not miscounted.
    rc, out, _ = _run(["lspci", "-nn"])
    pci_ids = {}
    if rc == 0:
        for ln in out.splitlines():
            m = re.search(r"\[1ae0:([0-9a-f]{4})\]", ln)
            if m:
                pci_ids.setdefault(m.group(1), []).append(ln)
        print("Google PCI functions by device id:")
        for dev_id, lines in sorted(pci_ids.items()):
            print(f"    1ae0:{dev_id}  x{len(lines)}   e.g. {lines[0]}")
    else:
        print("lspci unavailable")

    chips = len(vfio_devs) or len(accel)
    print(f"=> TPU devices on this host: {chips}")
    return chips


def probe_slice(zone, tpu_name):
    section("slice peers (gcloud)")
    if not shutil.which("gcloud"):
        print("gcloud not available; cannot enumerate peers")
        return []
    if not (zone and tpu_name):
        print("zone/slice name unknown; cannot enumerate peers")
        return []

    rc, out, err = _run([
        "gcloud", "compute", "tpus", "tpu-vm", "describe", tpu_name, "--zone",
        zone, "--format", "json"
    ],
                        timeout=120)
    if rc != 0:
        print(f"gcloud describe failed (rc={rc}): {err or out}")
        return []

    try:
        d = json.loads(out)
    except json.JSONDecodeError as e:
        print(f"could not parse gcloud json: {e}")
        return []

    endpoints = d.get("networkEndpoints", []) or []
    ips = [e.get("ipAddress", "") for e in endpoints]
    print(f"acceleratorType     : {d.get('acceleratorType', '-')}")
    print(f"runtimeVersion      : {d.get('runtimeVersion', '-')}")
    print(f"state               : {d.get('state', '-')}")
    print(f"hosts in slice      : {len(ips)}")
    for i, ip in enumerate(ips):
        role = "head" if i == 0 else f"worker {i}"
        print(f"    [{i}] {ip:<16} ({role})")
    return ips


def probe_peer_reachability(ips):
    section("peer reachability (ssh)")
    if len(ips) < 2:
        print("fewer than 2 hosts; nothing to check")
        return
    ssh_opts = [
        "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o",
        "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10"
    ]
    key = os.path.expanduser("~/.ssh/id_rsa")
    if os.path.exists(key):
        ssh_opts += ["-i", key]
    else:
        print("note: ~/.ssh/id_rsa missing; run_multihost.sh generates one")
    user = os.environ.get("SSH_USER") or os.environ.get("USER") or "root"
    for ip in ips[1:]:
        rc, out, err = _run([
            "ssh", *ssh_opts, f"{user}@{ip}",
            "hostname; ls /dev/vfio 2>/dev/null | grep -vc '^vfio$' || true"
        ],
                            timeout=45)
        if rc == 0:
            lines = out.splitlines()
            host = lines[0] if lines else "?"
            chips = lines[1] if len(lines) > 1 else "?"
            print(f"    {ip:<16} OK   host={host} chips={chips}")
        else:
            print(
                f"    {ip:<16} FAIL rc={rc} {err.splitlines()[-1] if err else ''}"
            )


def probe_runtime_tools():
    section("host tooling")
    for tool in ("docker", "gcloud", "ssh", "python3"):
        path = shutil.which(tool)
        print(f"{tool:<8}: {path or 'MISSING'}")
    rc, out, _ = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
    print(f"docker server: {out if rc == 0 else 'unavailable'}")


def main():
    worker_index = probe_identity()
    zone, tpu_name, tpu_env = probe_metadata()
    local_chips = probe_local_chips()
    ips = probe_slice(zone, tpu_name)
    probe_peer_reachability(ips)
    probe_runtime_tools()

    section("verdict")
    hosts = len(ips)
    if hosts and local_chips:
        print(f"slice: {hosts} host(s) x {local_chips} TPU devices per host "
              f"= {hosts * local_chips} devices total")
    if tpu_env.get("TOPOLOGY"):
        print(f"slice topology (from tpu-env): {tpu_env['TOPOLOGY']}, "
              f"host bounds {tpu_env.get('HOST_BOUNDS', '?')}, "
              f"chips per host {tpu_env.get('CHIPS_PER_HOST_BOUNDS', '?')}")
    if worker_index not in (0, None):
        print(f"WARNING: agent is on worker {worker_index}, not worker 0; "
              "the Ray head must run on worker 0.")
    if hosts < 2:
        print(f"ERROR: this agent does not see a multi-host slice (hosts="
              f"{hosts}). A multi-host test cannot run here -- check that the "
              f"step targets a multi-host queue.")
        return 1
    print("OK: multi-host slice visible from this agent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
