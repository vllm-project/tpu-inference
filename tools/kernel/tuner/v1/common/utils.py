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

import os
import re
import time

# from tensorflow.core.profiler.protobuf import xplane_pb2
from tensorflow.tsl.profiler.protobuf import xplane_pb2


def find_events_by_pattern(pb_file_path, event_pattern_string):
    # 1. Compile the regex pattern (case-insensitive by default for convenience)
    # for example: r"(jit_ragged_paged_attention\()"
    pattern = re.compile(event_pattern_string, re.IGNORECASE)

    if os.path.isdir(pb_file_path):
        # walk the directory to find the most recent .xplane.pb file
        pb_files = []
        for root, dirs, files in os.walk(pb_file_path):
            for file in files:
                if file.endswith(".xplane.pb"):
                    pb_files.append(os.path.join(root, file))
        if not pb_files:
            raise FileNotFoundError(
                f"No .xplane.pb files found in directory: {pb_file_path}")
        assert len(
            pb_files
        ) == 1, f"Multiple .xplane.pb files found in directory: {pb_file_path}. Please specify the file path directly."
        pb_file_path = pb_files[0]

        # # If a directory is provided, find the most recent .xplane.pb file in it
        # pb_files = [f for f in os.listdir(pb_file_path) if f.endswith(".xplane.pb")]
        # if not pb_files:
        #     raise FileNotFoundError(f"No .xplane.pb files found in directory: {pb_file_path}")
        # pb_files.sort(key=lambda f: os.path.getmtime(os.path.join(pb_file_path, f)), reverse=True)
        # pb_file_path = os.path.join(pb_file_path, pb_files[0])
        # print(f"Found .xplane.pb file: {pb_file_path}")

    # 2. Parse the binary file
    xspace = xplane_pb2.XSpace()
    with open(pb_file_path, "rb") as f:
        xspace.ParseFromString(f.read())

    matching_events = []

    # 3. Traverse the hierarchy
    for plane in xspace.planes:
        # Build the metadata lookup dictionary
        event_names = {
            meta_id: meta.name
            for meta_id, meta in plane.event_metadata.items()
        }

        for line in plane.lines:
            line_name = line.name if line.name else f"ID: {line.id}"

            for event in line.events:
                name = event_names.get(event.metadata_id, "Unknown Event")

                # 4. Check if the event name matches our regex pattern
                if pattern.search(name):
                    duration_ms = event.duration_ps / 1e9
                    start_ms = event.offset_ps / 1e9

                    matching_events.append({
                        "plane": plane.name,
                        "line": line_name,
                        "name": name,
                        "start_ms": start_ms,
                        "duration_ms": duration_ms
                    })
    average_duration_ms = (sum(e["duration_ms"] for e in matching_events) /
                           len(matching_events)) if matching_events else 0

    return matching_events, average_duration_ms


if __name__ == "__main__":
    file_path = "/mnt/disks/persist/batched_rpa_kernel_tuning/tmp/batched_rpa_run/"

    # Example 1: Match any event containing "dot" or "matmul"
    start_time = time.perf_counter()
    search_pattern = r"(jit_ragged_paged_attention\()"

    print(f"Searching for pattern: '{search_pattern}'\n")
    matching_events, average_duration_ms = find_events_by_pattern(
        file_path, search_pattern)
    end_time = time.perf_counter()
    print(f"Search completed in {end_time - start_time:.4f} seconds.\n")

    if not matching_events:
        print("No matching events found.")
    else:
        # Sort results by duration (longest taking events first)
        matching_events.sort(key=lambda x: x["duration_ms"], reverse=True)
        print(
            f"Average duration of matching events: {average_duration_ms:.4f} ms"
        )
        print(f"Found {len(matching_events)} matching events. Top 15 longest:")
        print(
            f"{'DURATION (ms)':<15} | {'START (ms)':<15} | {'PLANE / LINE':<30} | {'EVENT NAME'}"
        )
        print("-" * 80)

        for e in matching_events:
            plane_line = f"{e['plane'][:10]} / {e['line'][:15]}"
            print(
                f"{e['duration_ms']:<15.4f} | {e['start_ms']:<15.4f} | {plane_line:<30} | {e['name']}"
            )
