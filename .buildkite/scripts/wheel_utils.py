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
checks. serve_wheels.sh uses the serve checks, run by the Python of the venv
the wheels are installed into. publish_wheel.sh uses the publish checks for
tpu-inference and the combined vllm-tpu (wheel and sdist).
"""

import base64
import importlib.metadata
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from typing import Optional

from packaging.utils import (canonicalize_name, parse_sdist_filename,
                             parse_wheel_filename)
from packaging.version import Version


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


def verify_installed_wheels(version: str, venv: str) -> None:
    """Verify the venv has vllm-tpu and tpu-inference <version>, imports them
    from itself and runs vLLM on a TPU. Run by the venv's own Python."""
    venv = os.path.realpath(venv)
    for dist in ("vllm-tpu", "tpu-inference"):
        installed = importlib.metadata.version(dist)
        if installed != version:
            sys.exit(f"ERROR: {dist} {installed} is installed, not {version}.")

    # Imported here: only the venv the wheels went into has these.
    import jax
    import vllm
    from vllm.platforms import current_platform

    import tpu_inference

    for module in (vllm, tpu_inference):
        path = os.path.realpath(module.__file__)
        if not path.startswith(venv + os.sep):
            sys.exit(f"ERROR: {module.__name__} is imported from {path}, "
                     "not from the venv.")
    devices = jax.devices()
    if not devices or devices[0].platform != "tpu":
        sys.exit(f"ERROR: JAX sees no TPU: {devices}")
    if not current_platform.is_tpu():
        sys.exit(f"ERROR: vLLM picked the {current_platform.device_name} "
                 "platform, not TPU.")
    print(f"vllm-tpu {version} (vLLM {vllm.__version__}), "
          f"tpu-inference {version}")
    print(f"{current_platform.get_device_name()}: "
          f"{len(devices)} x {devices[0].device_kind}")


def _call_server(base_url: str, path: str, body: Optional[dict] = None):
    """GET <path> from the server, or POST <body> to it, and return the JSON
    it answers."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base_url + path,
                                 data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"ERROR: {path} answered {e.code}: "
                 f"{e.read().decode(errors='replace')}")


def verify_server_answers(base_url: str, model: str, out_file: str) -> None:
    """Verify the server at <base_url> serves <model> and answers a
    completion and a chat, and write the answers to <out_file>."""
    served = [m["id"] for m in _call_server(base_url, "/v1/models")["data"]]
    if model not in served:
        sys.exit(f"ERROR: /v1/models lists {served}, not {model}.")

    prompt = "The capital of France is"
    text = _call_server(base_url, "/v1/completions", {
        "model": model,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0,
    })["choices"][0]["text"]

    question = "Say hello in one short sentence."
    message = _call_server(
        base_url, "/v1/chat/completions", {
            "model": model,
            "messages": [{
                "role": "user",
                "content": question
            }],
            "max_tokens": 32,
            "temperature": 0,
        })["choices"][0]["message"]
    # With a reasoning parser on, the model's thinking lands in
    # reasoning_content.
    reply = message.get("content") or message.get("reasoning_content") or ""

    lines = [
        f"/v1/completions       {prompt!r} -> {text!r}",
        f"/v1/chat/completions  {question!r} -> {reply!r}",
    ]
    print("\n".join(lines))
    for name, answer in (("completion", text), ("chat reply", reply)):
        if not answer.strip():
            sys.exit(f"ERROR: the {name} is empty.")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")


def verify_bench_result(result_file: str, out_file: str) -> None:
    """Verify every request of a `vllm bench serve --save-result` run
    completed, and write a one-line summary to <out_file>. vllm bench serve
    itself only warns about failed requests, even when all of them fail, and
    still exits 0."""
    with open(result_file) as f:
        result = json.load(f)
    completed, failed = result["completed"], result["failed"]
    wanted = result["num_prompts"]
    if failed or completed != wanted:
        sys.exit(f"ERROR: vllm bench serve: {completed} of {wanted} requests "
                 f"completed, {failed} failed; the errors are above.")
    line = (f"vllm bench serve      {completed}/{wanted} requests, "
            f"{result['output_throughput']:.1f} output tok/s, "
            f"mean TTFT {result['mean_ttft_ms']:.1f} ms, "
            f"mean TPOT {result['mean_tpot_ms']:.1f} ms")
    print(line)
    with open(out_file, "w") as f:
        f.write(line + "\n")


def free_port() -> None:
    """Print a TCP port on 127.0.0.1 that nothing listens on right now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        print(s.getsockname()[1])


def verify_publish_dist(dist_path: str, project: str,
                        expected_version: str) -> None:
    """Verify a wheel or sdist matches the target
    PyPI project name and release version."""
    filename = os.path.basename(dist_path)
    if filename.endswith(".whl"):
        name, version, _, _ = parse_wheel_filename(filename)
    else:
        name, version = parse_sdist_filename(filename)
    if name != canonicalize_name(project):
        sys.exit(
            f"ERROR: {filename} is '{name}' but PUBLISH_PYPI is '{project}'.")
    if version != Version(expected_version):
        sys.exit(f"ERROR: {filename} version {version} "
                 f"!= release version {expected_version}.")
    print(f"Publishing {filename}")


def fetch_oidc_audience(index_base: str) -> None:
    """Query the PyPI OIDC audience endpoint and print the audience string."""
    with urllib.request.urlopen(f"{index_base}/_/oidc/audience",
                                timeout=30) as r:
        audience = json.load(r).get("audience")
    if not audience:
        sys.exit(
            "ERROR: Failed to parse audience from OIDC audience endpoint.")
    print(audience)


def verify_gcp_oidc_token(expected_aud: str) -> None:
    """Read OIDC JWT from stdin, verify it is from Google Cloud and for
    expected_aud, and log structural claims."""
    token = sys.stdin.read().strip()
    payload = token.split(".")[1]
    claims = json.loads(
        base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    if claims.get("iss") != "https://accounts.google.com":
        sys.exit(
            f"ERROR: identity token is from {claims.get('iss')!r}, not Google "
            "Cloud. Run this step on a GCE VM agent with a service account.")
    if claims.get("aud") != expected_aud:
        sys.exit(f"ERROR: identity token audience is {claims.get('aud')!r}, "
                 f"expected {expected_aud!r}.")
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
        "verify_built_wheel": verify_built_wheel,
        "verify_installed_wheels": verify_installed_wheels,
        "verify_server_answers": verify_server_answers,
        "verify_bench_result": verify_bench_result,
        "free_port": free_port,
        "verify_publish_dist": verify_publish_dist,
        "fetch_oidc_audience": fetch_oidc_audience,
        "verify_gcp_oidc_token": verify_gcp_oidc_token,
        "mint_pypi_token": mint_pypi_token,
    }
    if cmd not in commands:
        sys.exit(f"ERROR: unknown command '{cmd}'.")
    commands[cmd](*args)


if __name__ == "__main__":
    main()
