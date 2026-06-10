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

from tensorflow.tsl.profiler.protobuf import xplane_pb2


def find_events_by_pattern(pb_file_path, event_pattern_string):
    # 1. Compile the regex pattern (case-insensitive by default for convenience)
    # for example: r"(jit_ragged_paged_attention\()"
    pattern = re.compile(event_pattern_string, re.IGNORECASE)

    if os.path.isdir(pb_file_path):
        # walk the directory to find the .xplane.pb and expect only one file
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
                    duration_us = event.duration_ps // 1_000_000
                    start_us = event.offset_ps // 1_000_000

                    matching_events.append({
                        "plane": plane.name,
                        "line": line_name,
                        "name": name,
                        "start_us": start_us,
                        "duration_us": duration_us
                    })
    average_duration_us = (
        sum(e["duration_us"] for e in matching_events) //
        len(matching_events)) if matching_events else 0xFFFFFFFF

    return matching_events, average_duration_us
