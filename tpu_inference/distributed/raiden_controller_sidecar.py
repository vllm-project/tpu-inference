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
"""Raiden reshard controller sidecar for heterogeneous P/D KV transfer.

The controller is the only component that sees both sides' geometry. It
validates the layout fingerprints, plans the byte-span reshard, arms the
destination workers (``PoolReshardRegisterRecv``) and fires the sources
(``PoolReshardPush``) over their control-plane listener sockets. It moves no
data itself, so one small process serves a whole P/D pair.

It runs beside the prefill engine rather than inside it: the engine process
owns TPU chips and is restarted far more often than the control plane, and
prefill and decode may sit on different hosts, where an in-process controller
would have no natural home.

Raiden serves this plane from ``reshard_sidecar``, a standalone binary in the
Raiden tree, so this module supervises that binary rather than hosting the
plane itself: it resolves the binary, waits for the ready-file handshake, and
republishes the bound port for the launcher. ``build.sh`` does not build
``reshard_sidecar`` among its default targets and no wheel ships it, so a
Raiden source checkout built with
``//tpu_sync/kv_cache/reshard:reshard_sidecar`` is a prerequisite; point
``TPU_KV_RESHARD_SIDECAR`` at the binary if it is not discoverable.

Run it directly::

    python -m tpu_inference.distributed.raiden_controller_sidecar --port 9700

Both engines then point at it with ``TPU_KV_CONTROLLER_ADDRESS=<host>:9700``.
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Optional

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_BINARY_NAME = "reshard_sidecar"
_BINARY_ENV = "TPU_KV_RESHARD_SIDECAR"
# Path of the binary inside a Bazel-built Raiden checkout, relative to the
# repository root.
_BAZEL_RELPATH = os.path.join("bazel-bin", "tpu_sync", "kv_cache", "reshard",
                              _BINARY_NAME)


def find_sidecar_binary(explicit: Optional[str] = None) -> str:
    """Resolves the reshard sidecar binary.

    Args:
        explicit: A caller-supplied path, which wins if given.

    Returns:
        Path to an executable ``reshard_sidecar``.

    Raises:
        FileNotFoundError: If no candidate is executable, listing what was
            tried so the fix is obvious.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get(_BINARY_ENV):
        candidates.append(os.environ[_BINARY_ENV])
    on_path = shutil.which(_BINARY_NAME)
    if on_path:
        candidates.append(on_path)
    try:
        import tpu_sync

        # Raiden ships as a namespace package, so __file__ is None and the
        # search roots are the __path__ entries.
        for entry in list(getattr(tpu_sync, "__path__", ())):
            repo_root = os.path.dirname(entry)
            candidates.append(os.path.join(repo_root, _BAZEL_RELPATH))
    except Exception:  # pylint: disable=broad-except
        pass

    for candidate in candidates:
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        f"No executable {_BINARY_NAME} found. Build it with "
        f"'bazel build //tpu_sync/kv_cache/reshard:{_BINARY_NAME}' in a "
        f"Raiden checkout, or set {_BINARY_ENV}. Tried: "
        f"{candidates or '(no candidates)'}")


def primary_host_address() -> str:
    """Returns the routable local address the peers should dial.

    A connected UDP socket picks the interface the kernel would route out of
    without sending anything, which beats ``gethostbyname(gethostname())`` on
    hosts whose hostname resolves to loopback.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 53))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def start_controller(port: int,
                     advertise_host: Optional[str] = None,
                     ready_file: Optional[str] = None,
                     binary: Optional[str] = None,
                     request_registry_ttl_s: float = 600.0,
                     timeout_s: float = 60.0):
    """Starts the reshard sidecar and waits for it to bind.

    Args:
        port: TCP port to bind. 0 selects an ephemeral port, which the
            returned ready record reports under ``port``.
        advertise_host: Host published to peers; defaults to the routable
            local address.
        ready_file: Path for the ready handshake; defaults to a temp file.
        binary: Explicit path to ``reshard_sidecar``.
        request_registry_ttl_s: Lifetime of an unclaimed request-block
            registration.
        timeout_s: How long to wait for the ready handshake.

    Returns:
        ``(process, ready)`` -- the running :class:`subprocess.Popen` and the
        parsed ready record, whose ``port`` and ``address`` are authoritative.

    Raises:
        RuntimeError: If the binary exits or fails to publish in time.
    """
    binary_path = find_sidecar_binary(binary)
    host = advertise_host or primary_host_address()
    owns_ready_file = ready_file is None
    if owns_ready_file:
        handle, ready_file = tempfile.mkstemp(prefix="raiden_reshard_",
                                              suffix=".ready")
        os.close(handle)
    # The launcher polls for non-empty content, so a stale file from a crashed
    # run would be read as this run's handshake.
    if os.path.exists(ready_file):
        os.unlink(ready_file)

    argv = [
        binary_path,
        f"--port={port}",
        f"--advertise-host={host}",
        f"--ready-file={ready_file}",
        f"--request-registry-ttl-s={request_registry_ttl_s}",
    ]
    logger.info(f"Starting reshard sidecar: {' '.join(argv)}")
    process = subprocess.Popen(argv)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Reshard sidecar exited with code {process.returncode} "
                "before publishing its ready file")
        try:
            with open(ready_file, "r") as f:
                line = f.read().strip()
            if line:
                ready = json.loads(line)
                logger.info(
                    f"Raiden reshard sidecar listening on {ready['address']}")
                if owns_ready_file:
                    os.unlink(ready_file)
                return process, ready
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.1)

    process.terminate()
    raise RuntimeError(
        f"Reshard sidecar did not publish {ready_file} within {timeout_s}s")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",
                        type=int,
                        default=9700,
                        help="TCP port to bind (0 for an ephemeral port).")
    parser.add_argument("--advertise-host",
                        default=None,
                        help="Host published to peers (default: the routable "
                        "local address).")
    parser.add_argument("--ready-file",
                        default=None,
                        help="Path for the ready handshake (default: a temp "
                        "file).")
    parser.add_argument("--binary",
                        default=None,
                        help=f"Path to {_BINARY_NAME} (default: "
                        f"${_BINARY_ENV}, $PATH, then the Bazel output of an "
                        "importable Raiden checkout).")
    parser.add_argument("--request-registry-ttl-s",
                        type=float,
                        default=600.0,
                        help="Lifetime of an unclaimed request-block "
                        "registration.")
    args = parser.parse_args(argv)

    process, ready = start_controller(
        args.port,
        advertise_host=args.advertise_host,
        ready_file=args.ready_file,
        binary=args.binary,
        request_registry_ttl_s=args.request_registry_ttl_s)
    # Emitted on stdout so a launcher script can capture the port when --port 0
    # was used.
    print(f"RAIDEN_CONTROLLER_PORT={ready['port']}", flush=True)

    def forward(signum, _frame):
        if process.poll() is None:
            process.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, forward)
    returncode = process.wait()
    logger.info("Raiden reshard sidecar stopped")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
