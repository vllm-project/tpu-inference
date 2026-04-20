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
import socket
import time


def get_host_ip():
    is_docker = os.path.exists('/.dockerenv')
    if not is_docker:
        # Not running in Docker, return local IP
        return socket.gethostbyname(socket.gethostname())
    try:
        host_ip = socket.gethostbyname('host.docker.internal')
        return host_ip
    except socket.gaierror:
        print("Could not resolve host.docker.internal")
        return None


def get_timestamp_sec():
    return int(time.time())
