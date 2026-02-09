#!/bin/bash
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

set -euo pipefail

docker_root=$(docker info -f '{{.DockerRootDir}}')
if [ -z "$docker_root" ]; then
  echo "Failed to determine Docker root directory."
  exit 1
fi
echo "Docker root directory: $docker_root"
# Check disk usage of the filesystem where Docker's root directory is located
disk_usage=$(df "$docker_root" | tail -1 | awk '{print $5}' | sed 's/%//')
# Define the threshold
threshold=70
if [ "$disk_usage" -gt "$threshold" ]; then
  echo "Disk usage($disk_usage) is above $threshold%. Cleaning up Docker images and volumes..."
  # Remove dangling images (those that are not tagged and not used by any container)
  docker image prune -f
  # Remove unused volumes / force the system prune for old images as well.
  docker volume prune -f && docker system prune --force --filter "until=12h" --all
  echo "Docker images and volumes cleanup completed."
else
  echo "Disk usage($disk_usage%) is below $threshold%. No cleanup needed."
fi
