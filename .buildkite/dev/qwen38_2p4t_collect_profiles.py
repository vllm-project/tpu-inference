"""Copy the phased profiler's files from every other Ray node into this one.

The profiler runs in each host's Ray worker and writes its trace under that
host's PHASED_PROFILING_DIR. On the kube slice only the head pod uploads its
artifacts, and the other pods are deleted with their files. Run on the head
after the server has stopped, while the Ray cluster is still up, this pulls
each worker host's profile files over Ray, one 128 MiB chunk at a time (a
host's xplane is ~0.7 GB), into the same relative paths under the head's
directory.

    python3 qwen38_2p4t_collect_profiles.py <profiling dir> [--timeout S]

Trace files are named after their pod, so hosts never collide. A file whose
path already exists here is skipped if its content is identical (size and
SHA-1), and otherwise written with the source host's name appended.
Diagnostics only: never exits non-zero.
"""
import argparse
import hashlib
import os
import socket
import sys
import time

CHUNK = 128 << 20


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("profiling_dir")
    p.add_argument("--dest", default=None,
                   help="write here instead of profiling_dir (testing)")
    p.add_argument("--address", default="auto")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--include-head", action="store_true",
                   help="also copy this node's own files (testing)")
    a = p.parse_args()
    dest_root = a.dest or a.profiling_dir
    deadline = time.monotonic() + a.timeout

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=a.address, logging_level="error", log_to_driver=False)
    head = ray.get_runtime_context().get_node_id()
    nodes = [n for n in ray.nodes() if n["Alive"]
             and (a.include_head or n["NodeID"] != head)]
    print(f"[collect] {len(nodes)} node(s) to collect from")

    def sha1(path):
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()

    @ray.remote(num_cpus=0)
    def list_files(root):
        out = []
        for d, _, files in os.walk(root):
            for f in files:
                path = os.path.join(d, f)
                out.append((os.path.relpath(path, root), os.path.getsize(path),
                            sha1(path)))
        return socket.gethostname(), out

    @ray.remote(num_cpus=0)
    def read_chunk(path, offset):
        with open(path, "rb") as fh:
            fh.seek(offset)
            return fh.read(CHUNK)

    total = 0
    for n in nodes:
        pin = NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)
        try:
            host, files = ray.get(
                list_files.options(scheduling_strategy=pin).remote(a.profiling_dir),
                timeout=max(1.0, deadline - time.monotonic()))
        except Exception as e:  # noqa: BLE001
            print(f"[collect] {n['NodeManagerAddress']}: listing failed: {e!r}")
            continue
        got = 0
        for rel, size, digest in files:
            base = os.path.join(dest_root, rel)
            out = next((c for c in (base, f"{base}.{host}")
                        if not os.path.exists(c)
                        or (os.path.getsize(c) == size and sha1(c) == digest)),
                       None)
            if out is None:
                print(f"[collect] {host}: {rel}: both {base} and its .{host} copy "
                      "exist with other content; skipped")
                continue
            if os.path.exists(out):
                continue                      # identical copy already here
            os.makedirs(os.path.dirname(out), exist_ok=True)
            try:
                with open(out, "wb") as fh:
                    for offset in range(0, size, CHUNK):
                        fh.write(ray.get(
                            read_chunk.options(scheduling_strategy=pin).remote(
                                os.path.join(a.profiling_dir, rel), offset),
                            timeout=max(1.0, deadline - time.monotonic())))
            except Exception as e:  # noqa: BLE001
                print(f"[collect] {host}: {rel}: {e!r}")
                os.remove(out)                # never leave a partial copy
                continue
            got += size
        total += got
        print(f"[collect] {host} ({n['NodeManagerAddress']}): {len(files)} file(s), "
              f"{got / 2**20:.1f} MiB copied")
    print(f"[collect] done: {total / 2**20:.1f} MiB into {dest_root}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print(f"[collect] failed: {e!r}")
        sys.exit(0)
