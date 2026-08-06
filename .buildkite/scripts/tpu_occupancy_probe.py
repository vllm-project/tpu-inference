#!/usr/bin/env python3
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
"""Read-only probe: is anything holding this host's TPU chips?

Strictly observational. It never imports jax/libtpu, never opens a TPU
device, and never signals a process, so it cannot disturb a workload that
is running or in the middle of starting up. Everything below is a read of
/dev, /proc, or the output of ps/who/ss.

Exits 0 regardless of the verdict -- the verdict is the log output, not
the exit status, so a busy TPU does not read as a build failure.
"""

import glob
import os
import subprocess
import sys

# Devices a TPU workload holds open while it owns the chips.
DEVICE_GLOBS = ("/dev/vfio/*", "/dev/accel*")

# Ports worth reporting: 8470-8479 are the TPU worker ports, 8000 is the
# vllm serve default. A service that is still starting up usually shows a
# listener before it shows sustained HBM use.
PORTS_OF_INTEREST = {8000, 8470, 8471, 8472, 8473, 8474, 8475, 8476, 8477}


def run(cmd, timeout=20):
    """Run a read-only command, returning stdout ('' on any failure)."""
    try:
        proc = subprocess.run(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL,
                              text=True,
                              timeout=timeout)
        return proc.stdout if proc.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def section(title):
    print(f"\n--- {title} ---", flush=True)


def list_devices():
    """TPU device nodes present on this host."""
    found = []
    for pattern in DEVICE_GLOBS:
        found.extend(sorted(glob.glob(pattern)))
    return found


def proc_field(pid, name):
    """One field from /proc/<pid>/status, or '' if unreadable."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(name + ":"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return ""
    return ""


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return " ".join(f.read().decode("utf-8",
                                            "replace").split("\0")).strip()
    except OSError:
        return ""


def scan_proc_for_holders(devices):
    """Find PIDs with an open fd on any TPU device, by reading /proc.

    Pure filesystem reads. Processes owned by other users are only visible
    when running as root; unreadable entries are counted and reported so a
    partial scan is never mistaken for an empty one.
    """
    holders = {}
    unreadable = 0
    device_set = set(devices)
    for pid_dir in glob.glob("/proc/[0-9]*"):
        pid = os.path.basename(pid_dir)
        fd_dir = f"{pid_dir}/fd"
        try:
            fds = os.listdir(fd_dir)
        except PermissionError:
            unreadable += 1
            continue
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if target in device_set or target.startswith("/dev/vfio/") \
                    or target.startswith("/dev/accel"):
                holders.setdefault(pid, set()).add(target)
    return holders, unreadable


def report_holders(holders):
    for pid in sorted(holders, key=int):
        user = proc_field(pid, "Uid").split()[0] if proc_field(pid,
                                                               "Uid") else "?"
        name = proc_field(pid, "Name")
        devs = ", ".join(sorted(holders[pid]))
        print(f"  pid={pid} uid={user} name={name}")
        print(f"    devices: {devs}")
        print(f"    cmdline: {cmdline(pid)[:200]}")


def main():
    print("=" * 70)
    print(f"TPU occupancy probe (read-only) on {os.uname().nodename}")
    print("=" * 70)

    section("TPU device nodes")
    devices = list_devices()
    if devices:
        for d in devices:
            print(f"  {d}")
    else:
        print("  none found (no /dev/vfio/* or /dev/accel*)")

    section("Processes holding a TPU device (/proc fd scan)")
    holders, unreadable = scan_proc_for_holders(devices)
    if holders:
        report_holders(holders)
    else:
        print("  none visible")
    if unreadable:
        print(f"  NOTE: {unreadable} process(es) not readable by uid "
              f"{os.geteuid()} -- scan is partial, see lsof below")

    # Cross-check with lsof under sudo -n (never prompts). This catches
    # holders the /proc scan could not see when not running as root.
    section("Cross-check: lsof on TPU devices")
    lsof_out = run(["sudo", "-n", "lsof"] + list(DEVICE_GLOBS))
    if not lsof_out:
        lsof_out = run(["lsof"] + list(DEVICE_GLOBS))
    print(lsof_out.rstrip() if lsof_out.strip(
    ) else "  no output (either nothing holds the devices, or lsof "
          "is unavailable / not permitted)")

    section("Accelerator-related processes (ps)")
    ps_out = run(
        ["ps", "-eo", "pid,user,etime,pcpu,pmem,args", "--sort=-pcpu"])
    matched = [
        line for line in ps_out.splitlines()
        if any(k in line.lower()
               for k in ("vllm", "jax", "libtpu", "tpu_inference",
                         "vllm_torchtpu",
                         "python")) and "tpu_occupancy_probe" not in line
    ]
    if matched:
        for line in matched[:15]:
            print(f"  {line[:200]}")
    else:
        print("  none")

    section("libtpu lockfile")
    lock = "/tmp/libtpu_lockfile"
    print(f"  {lock}: {'present' if os.path.exists(lock) else 'absent'}")

    section("Listening ports of interest")
    ss_out = run(["ss", "-lntp"])
    hits = [
        line for line in ss_out.splitlines()
        if any(f":{p} " in line or line.rstrip().endswith(f":{p}")
               for p in PORTS_OF_INTEREST)
    ]
    if hits:
        for line in hits[:15]:
            print(f"  {line[:200]}")
    else:
        print("  none of "
              f"{sorted(PORTS_OF_INTEREST)} are listening")

    section("Logged-in users / recent logins")
    who_out = run(["who"])
    print(
        who_out.rstrip() if who_out.strip() else "  (no interactive sessions)")
    last_out = run(["last", "-n", "8"])
    if last_out.strip():
        print("  recent logins:")
        for line in last_out.splitlines()[:8]:
            print(f"    {line[:120]}")

    # Verdict. Deliberately conservative: anything holding a device, or a
    # partial scan with a live-looking listener, reports as in use.
    section("VERDICT")
    lsof_has_holder = bool([
        ln for ln in lsof_out.splitlines()
        if ln.strip() and not ln.startswith("COMMAND")
    ])
    if holders or lsof_has_holder:
        print("  IN USE -- a process currently holds a TPU device. "
              "Do not reset or reclaim this pod.")
    elif not devices:
        print("  UNKNOWN -- no TPU device nodes visible from this context.")
    else:
        print("  NO DEVICE HOLDER FOUND -- no process holds a TPU device "
              "right now.")
        if unreadable:
            print("  Caveat: the /proc scan was partial and lsof added "
                  "nothing, so a holder owned by another user may have "
                  "been missed.")
    print("\nProbe complete. Nothing was started, stopped, or signalled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
