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
"""Round-trip wake-up latency between two threads and between two processes.

A sleeping thread woken by another pays the CPU's idle-exit cost. On a host
booted with idle=poll that cost is near zero; on one whose idle CPUs halt it
is a VM exit and re-entry. The engine's per-step handoffs pay it every step.
"""
import os
import statistics
import threading
import time

N = 20000


def report(name, samples):
    s = sorted(samples)
    pct = lambda p: s[int(p * (len(s) - 1))] * 1e6
    print(f"{name:28s} p50 {pct(0.5):7.1f} us  p90 {pct(0.9):7.1f} us  "
          f"p99 {pct(0.99):7.1f} us  mean {statistics.mean(s) * 1e6:7.1f} us")


def threads():
    ping, pong = threading.Semaphore(0), threading.Semaphore(0)

    def peer():
        for _ in range(N):
            ping.acquire()
            pong.release()

    t = threading.Thread(target=peer)
    t.start()
    out = []
    for _ in range(N):
        t0 = time.perf_counter()
        ping.release()
        pong.acquire()
        out.append(time.perf_counter() - t0)
    t.join()
    report("thread semaphore round trip", out)


def processes():
    a_r, a_w = os.pipe()
    b_r, b_w = os.pipe()
    if os.fork() == 0:
        for _ in range(N):
            os.read(a_r, 1)
            os.write(b_w, b"x")
        os._exit(0)
    out = []
    for _ in range(N):
        t0 = time.perf_counter()
        os.write(a_w, b"x")
        os.read(b_r, 1)
        out.append(time.perf_counter() - t0)
    os.wait()
    report("process pipe round trip", out)


def sleep_wake():
    out = []
    for _ in range(2000):
        t0 = time.perf_counter()
        time.sleep(0.0001)
        out.append(time.perf_counter() - t0 - 0.0001)
    report("sleep(100us) overshoot", out)


if __name__ == "__main__":
    threads()
    processes()
    sleep_wake()
