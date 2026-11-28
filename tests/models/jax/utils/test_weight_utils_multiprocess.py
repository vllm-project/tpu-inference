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
"""Weight staging when several processes share one jax.distributed domain.

Each process runs its own engine on its own local devices (a host-local
mesh), while the CPU client spans every process. Weights are staged on a CPU
device before they are placed on the mesh, so that device has to belong to
the loading process. This test starts two CPU processes in one
jax.distributed domain and, in each, loads a tiny param through the real
weight-load path onto a mesh built from that process's local devices.
"""

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time

import pytest

_NPROC = 2
_LOCAL_DEVICES = 2
_TIMEOUT_S = 300
# Attempts to form the domain; another process can take the coordinator port.
_ATTEMPTS = 3


def _free_port() -> int:
    # The coordinator listens on the wildcard address, so probe that.
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _local_only(arr, process_index: int) -> bool:
    return arr.is_fully_addressable and all(d.process_index == process_index
                                            for d in arr.sharding.device_set)


def _error(e: BaseException) -> str:
    msgs = []
    while e is not None and len(msgs) < 4:
        msgs.append(f"{type(e).__name__}: {str(e)[:300]}")
        e = e.__cause__ or e.__context__
    return " <- ".join(msgs)


def _worker(args) -> dict:
    import jax

    res = {"pid": args.pid}
    try:
        jax.distributed.initialize(f"127.0.0.1:{args.port}",
                                   num_processes=args.nproc,
                                   process_id=args.pid,
                                   initialization_timeout=60)
    except Exception as e:  # pylint: disable=broad-except
        res["init_error"] = f"{type(e).__name__}: {e}"
        print(f"jax.distributed.initialize failed: {res['init_error']}")
        return res
    open(os.path.join(args.out, f"p{args.pid}.up"), "w").close()

    import jax.numpy as jnp
    import numpy as np
    import torch
    from flax import nnx
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from safetensors.numpy import save_file

    from tpu_inference.layers.common import utils as common_utils
    from tpu_inference.models.jax.utils import weight_utils

    me = jax.process_index()
    res["process_count"] = jax.process_count()
    res["cpu_mesh_devices"] = [[d.id, d.process_index]
                               for d in common_utils.cpu_mesh().devices.flat]
    # A host-local mesh: this process's devices only.
    devs = jax.local_devices()
    mesh = Mesh(np.array(devs).reshape(1, len(devs)), ("data", "model"))

    hidden, vocab = 8, 8 * len(devs)
    torch_weight = torch.arange(vocab * hidden,
                                dtype=torch.float32).reshape(vocab, hidden)
    want = np.ascontiguousarray(torch_weight.numpy().T)
    for backend in ("", "spmd"):
        if backend:
            os.environ["TPU_MULTIHOST_BACKEND"] = backend
        else:
            os.environ.pop("TPU_MULTIHOST_BACKEND", None)
        try:
            init = jax.device_put(jnp.zeros((hidden, vocab), jnp.float32),
                                  NamedSharding(mesh, P()))
            param = nnx.Param(init, out_sharding=P(None, "model"), mesh=mesh)
            weight_utils.load_nnx_param_from_reshaped_torch(
                param, torch_weight, param_name="lm_head.weight")
            val = param.get_value()
            res[f"load[{backend}]"] = _local_only(val, me) and np.array_equal(
                np.asarray(val), want)
        except Exception as e:  # pylint: disable=broad-except
            res[f"load[{backend}]"] = _error(e)

    # The safetensors file generator stages flax tensors on a CPU device too.
    path = os.path.join(args.out, f"w{args.pid}.safetensors")
    save_file({"w": want}, path)
    try:
        ((_, arr), ) = weight_utils.model_weights_single_file_generator(
            path, framework="flax")
        res["safetensors_flax"] = _local_only(arr, me) and np.array_equal(
            np.asarray(arr), want)
    except Exception as e:  # pylint: disable=broad-except
        res["safetensors_flax"] = _error(e)
    return res


def _start_and_wait(out_dir, env) -> None:
    port = _free_port()
    procs = []
    try:
        try:
            for pid in range(_NPROC):
                cmd = [sys.executable, os.path.abspath(__file__)]
                cmd += ["--pid", str(pid), "--nproc", str(_NPROC)]
                cmd += ["--port", str(port), "--out", str(out_dir)]
                with open(out_dir / f"p{pid}.log", "w") as log:
                    procs.append(
                        subprocess.Popen(cmd,
                                         env=env,
                                         cwd=out_dir,
                                         stdout=log,
                                         stderr=subprocess.STDOUT))
        except OSError as e:
            pytest.skip(f"cannot start {_NPROC} worker processes: {e}")
        deadline = time.monotonic() + _TIMEOUT_S
        while time.monotonic() < deadline:
            codes = [p.poll() for p in procs]
            if all(c is not None for c in codes):
                break
            # A worker that exited without joining leaves the others waiting.
            if any(c is not None and not (out_dir / f"p{i}.up").exists()
                   for i, c in enumerate(codes)):
                break
            time.sleep(0.2)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()


def _run_workers(tmp_path) -> list[dict]:
    spec = importlib.util.find_spec("tpu_inference")
    pkg_root = os.path.dirname(os.path.dirname(spec.origin))
    env = {
        k: v
        for k, v in os.environ.items()
        if k.lower() not in ("http_proxy", "https_proxy", "no_proxy",
                             "tpu_multihost_backend")
    }
    env.update(
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"--xla_force_host_platform_device_count={_LOCAL_DEVICES}",
        PYTHONPATH=os.pathsep.join(p for p in (pkg_root, env.get("PYTHONPATH"))
                                   if p))
    for attempt in range(_ATTEMPTS):
        out_dir = tmp_path / f"attempt{attempt}"
        out_dir.mkdir()
        _start_and_wait(out_dir, env)
        logs = {
            pid: (out_dir / f"p{pid}.log").read_text()[-3000:]
            for pid in range(_NPROC)
        }
        if all((out_dir / f"p{pid}.up").exists() for pid in range(_NPROC)):
            break
    else:
        pytest.skip(f"{_NPROC} processes could not form a jax.distributed "
                    f"domain in {_ATTEMPTS} attempts:\n" +
                    "\n".join(logs.values()))

    results = []
    for pid in range(_NPROC):
        path = out_dir / f"p{pid}.json"
        if not path.exists():
            pytest.fail(f"process {pid} wrote no result:\n{logs[pid]}")
        r = json.loads(path.read_text())
        if r["process_count"] != _NPROC:
            pytest.skip(f"expected {_NPROC} processes in the domain, got "
                        f"{r['process_count']}")
        results.append(r)
    return results


def test_weight_staging_is_process_local(tmp_path):
    failures = []
    for r in _run_workers(tmp_path):
        pid = r["pid"]
        if any(proc != pid for _, proc in r["cpu_mesh_devices"]):
            failures.append(f"process {pid}: cpu_mesh() is on another "
                            f"process's device {r['cpu_mesh_devices']}")
        for key in ("load[]", "load[spmd]", "safetensors_flax"):
            if r[key] is not True:
                failures.append(f"process {pid}: {key} -> {r[key]}")
    assert not failures, "\n".join(failures)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    res = _worker(args)
    with open(os.path.join(args.out, f"p{args.pid}.json"), "w") as f:
        json.dump(res, f)
    # Process 0 hosts the coordination service: keep it up until every peer
    # has written its result.
    if args.pid == 0 and "init_error" not in res:
        deadline = time.monotonic() + _TIMEOUT_S
        while time.monotonic() < deadline and not all(
                os.path.exists(os.path.join(args.out, f"p{i}.json"))
                for i in range(args.nproc)):
            time.sleep(0.2)
    sys.stdout.flush()
    # Skip interpreter teardown: the distributed client can block at exit
    # once a peer process is gone.
    os._exit(0)


if __name__ == "__main__":
    _main()
