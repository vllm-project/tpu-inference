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
"""Helper functions for building the tpu-inference release wheel.

Used by build_tpu_inference_wheel.sh.
"""

import os
import sys
import urllib.error
import urllib.request

from packaging.utils import parse_wheel_filename
from packaging.version import InvalidVersion, Version


def validate_version(raw_version: str) -> None:
    """Print the normalized version; exit 1 if PyPI would not accept it."""
    try:
        version = Version(raw_version)
    except InvalidVersion:
        sys.exit(f"ERROR: '{raw_version}' is not a valid PEP 440 version.")
    # PyPI rejects local versions (e.g. 0.31.0+abc).
    if version.local is not None:
        sys.exit(
            f"ERROR: '{raw_version}' has a local part (+...); PyPI rejects it."
        )
    print(version)


def verify_built_wheel(wheel_path: str, expected_version: str) -> None:
    """Verify the built wheel's filename version and tags."""
    name, version, _, tags = parse_wheel_filename(os.path.basename(wheel_path))
    if str(version) != expected_version:
        sys.exit(
            f"ERROR: wheel version {version} != release version {expected_version}"
        )
    if {str(t) for t in tags} != {"py3-none-any"}:
        print(
            f"WARNING: wheel is tagged {sorted(map(str, tags))}, not py3-none-any.",
            file=sys.stderr,
        )
    print(f"Wheel: {name} {version}")


def check_not_published(index_base: str, project: str, version: str) -> None:
    """Exit 1 if project==version is already on the index (PyPI never allows
    reusing a version)."""
    url = f"{index_base}/pypi/{project}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30):
            pass
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"{project} {version} is not on {index_base} yet.")
            return
        sys.exit(f"ERROR: could not check {url} (HTTP {e.code}).")
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: could not reach {url}: {e.reason}")
    sys.exit(f"ERROR: {project} {version} already exists on {index_base}. "
             "PyPI versions can't be reused: use a new release branch "
             "(e.g. releases/v0.31.1) or a new RELEASE_VERSION.")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: wheel_utils.py <command> [args...]")
    cmd = sys.argv[1]
    args = sys.argv[2:]

    commands = {
        "validate_version": validate_version,
        "verify_built_wheel": verify_built_wheel,
        "check_not_published": check_not_published,
    }
    if cmd not in commands:
        sys.exit(f"ERROR: unknown command '{cmd}'.")
    commands[cmd](*args)


if __name__ == "__main__":
    main()
