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
"""Helper functions for building and publishing the release wheels.

build_tpu_inference_wheel.sh and build_vllm_tpu_wheel.sh use the build
checks. publish_wheel.sh uses the publish checks for tpu-inference and the
combined vllm-tpu (wheel and sdist).
"""

import base64
import json
import os
import sys
import urllib.error
import urllib.request

from packaging.utils import (canonicalize_name, parse_sdist_filename,
                             parse_wheel_filename)
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


def verify_publish_dist(dist_path: str, expected_name: str,
                        expected_version: str) -> None:
    """Verify a wheel or sdist is the expected distribution and release
    version."""
    filename = os.path.basename(dist_path)
    if filename.endswith(".whl"):
        name, version, _, _ = parse_wheel_filename(filename)
    else:
        name, version = parse_sdist_filename(filename)
    if name != canonicalize_name(expected_name):
        sys.exit(f"ERROR: {filename} is '{name}', expected '{expected_name}'.")
    if version != Version(expected_version):
        sys.exit(f"ERROR: {filename} version {version} "
                 f"!= release version {expected_version}.")
    print(f"Verified {filename}")


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


def fetch_oidc_audience(index_base: str) -> None:
    """Query the PyPI OIDC audience endpoint and print the audience string."""
    with urllib.request.urlopen(f"{index_base}/_/oidc/audience",
                                timeout=30) as r:
        audience = json.load(r).get("audience")
    if not audience:
        sys.exit(
            "ERROR: Failed to parse audience from OIDC audience endpoint.")
    print(audience)


def verify_gcp_oidc_token() -> None:
    """Read OIDC JWT from stdin, verify Google Cloud issuer,
    and log structural claims."""
    token = sys.stdin.read().strip()
    payload = token.split(".")[1]
    claims = json.loads(
        base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    if claims.get("iss") != "https://accounts.google.com":
        sys.exit(
            f"ERROR: identity token is from {claims.get('iss')!r}, not Google "
            "Cloud. Run this step on a GCE VM agent with a service account.")
    print("Verified GCP OIDC Token Structural Claims (redacted identity):")
    print(
        json.dumps(
            {
                "iss": claims.get("iss"),
                "aud": claims.get("aud"),
                "email_verified": claims.get("email_verified"),
                "has_email_claim": bool(claims.get("email")),
                "has_sub_claim": bool(claims.get("sub")),
            },
            indent=2,
        ))


def mint_pypi_token(index_base: str, project: str) -> None:
    """Read OIDC JWT from stdin and exchange it for a short-lived PyPI API
    token."""
    oidc_token = sys.stdin.read().strip()
    req = urllib.request.Request(
        f"{index_base}/_/oidc/mint-token",
        data=json.dumps({
            "token": oidc_token
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            err_json = json.loads(raw)
            sanitized = json.dumps(
                {
                    "message": err_json.get("message"),
                    "errors": err_json.get("errors")
                },
                indent=2,
            )
        except Exception:
            sanitized = raw
        print(f"Mint-token returned HTTP {e.code}:\n{sanitized}",
              file=sys.stderr)
        print(
            "\n================================================================"
            "======\n"
            f"NOTE: GCP OIDC token generation succeeded, but {index_base} "
            f"rejected\nthe publisher lookup (HTTP {e.code}).\n"
            f"Ensure a Trusted/Pending Publisher is registered on "
            f"{index_base}/manage/account/publishing/\n"
            f"for PyPI Project Name {project!r} with this VM's GCP Service "
            "Account.\n"
            "================================================================"
            "======",
            file=sys.stderr,
        )
        sys.exit(1)

    token = body.get("token")
    if not token:
        sys.exit(
            "ERROR: HTTP 200 from mint-token but no '.token' field found.")
    print(token)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: wheel_utils.py <command> [args...]")
    cmd = sys.argv[1]
    args = sys.argv[2:]

    commands = {
        "validate_version": validate_version,
        "verify_built_wheel": verify_built_wheel,
        "verify_publish_dist": verify_publish_dist,
        "check_not_published": check_not_published,
        "fetch_oidc_audience": fetch_oidc_audience,
        "verify_gcp_oidc_token": verify_gcp_oidc_token,
        "mint_pypi_token": mint_pypi_token,
    }
    if cmd not in commands:
        sys.exit(f"ERROR: unknown command '{cmd}'.")
    commands[cmd](*args)


if __name__ == "__main__":
    main()
