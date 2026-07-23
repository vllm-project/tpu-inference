#!/usr/bin/env python3
"""Create, wait for, or delete a per-job Kubernetes cache PVC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import ssl
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

SERVICE_ACCOUNT_ROOT = Path("/var/run/secrets/kubernetes.io/serviceaccount")
DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")


def validate_name(value: str, label: str) -> str:
    if not DNS_LABEL.fullmatch(value):
        raise ValueError(
            f"{label} must be a lowercase Kubernetes DNS label of at most 63 characters")
    return value


def build_claim(
    *,
    name: str,
    source_pvc: str,
    storage_class: str,
    size: str,
    build_id: str,
    zone: str | None,
) -> dict[str, Any]:
    validate_name(name, "claim name")
    validate_name(source_pvc, "source PVC")
    labels = {
        "app.kubernetes.io/managed-by": "buildkite",
        "tpu-inference.dev/cache-role": "job-clone",
    }
    annotations = {
        "tpu-inference.dev/source-pvc": source_pvc,
        "tpu-inference.dev/build-id": build_id,
    }
    if zone:
        annotations["tpu-inference.dev/expected-zone"] = zone
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": name,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "storageClassName": storage_class,
            "dataSource": {
                "apiGroup": "",
                "kind": "PersistentVolumeClaim",
                "name": source_pvc,
            },
            "resources": {
                "requests": {
                    "storage": size,
                },
            },
        },
    }


class KubernetesClient:

    def __init__(self, namespace: str | None = None):
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        if not host:
            raise ValueError("KUBERNETES_SERVICE_HOST is not set; run inside a pod")
        self.base_url = f"https://{host}:{port}/api/v1"
        self.namespace = namespace or (
            SERVICE_ACCOUNT_ROOT / "namespace").read_text().strip()
        validate_name(self.namespace, "namespace")
        self.token = (SERVICE_ACCOUNT_ROOT / "token").read_text().strip()
        context = ssl.create_default_context(
            cafile=str(SERVICE_ACCOUNT_ROOT / "ca.crt"))
        self.context = context

    @property
    def claims_url(self) -> str:
        return f"{self.base_url}/namespaces/{quote(self.namespace)}/persistentvolumeclaims"

    def request(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        data = json.dumps(body).encode() if body is not None else None
        request = Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, context=self.context, timeout=30) as response:
                payload = json.loads(response.read() or b"{}")
                return response.status, payload
        except HTTPError as error:
            payload = json.loads(error.read() or b"{}")
            return error.code, payload
        except URLError as error:
            raise RuntimeError(f"Kubernetes API request failed: {error}") from error

    def get_claim(self, name: str) -> tuple[int, dict[str, Any]]:
        validate_name(name, "claim name")
        return self.request("GET", f"{self.claims_url}/{quote(name)}")

    def create_claim(self, claim: dict[str, Any]) -> None:
        status, payload = self.request("POST", self.claims_url, claim)
        if status in (200, 201):
            return
        if status == 409:
            existing_status, existing = self.get_claim(claim["metadata"]["name"])
            if existing_status != 200:
                raise RuntimeError(
                    f"PVC exists but cannot be read (HTTP {existing_status}): {existing}")
            expected_source = claim["metadata"]["annotations"][
                "tpu-inference.dev/source-pvc"]
            existing_source = existing.get("metadata", {}).get(
                "annotations", {}).get("tpu-inference.dev/source-pvc")
            if existing_source == expected_source:
                return
            raise RuntimeError(
                f"PVC already exists with a different source: {existing_source!r}")
        raise RuntimeError(f"PVC creation failed (HTTP {status}): {payload}")

    def wait_bound(self, name: str, timeout_seconds: int) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status, claim = self.get_claim(name)
            if status == 200 and claim.get("status", {}).get("phase") == "Bound":
                return claim
            if status not in (200, 404):
                raise RuntimeError(f"PVC read failed (HTTP {status}): {claim}")
            time.sleep(2)
        raise TimeoutError(f"PVC {name!r} did not bind within {timeout_seconds}s")

    def delete_claim(self, name: str) -> None:
        validate_name(name, "claim name")
        status, payload = self.request(
            "DELETE",
            f"{self.claims_url}/{quote(name)}",
            {"apiVersion": "v1", "kind": "DeleteOptions"},
        )
        if status not in (200, 202, 404):
            raise RuntimeError(f"PVC deletion failed (HTTP {status}): {payload}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace")
    subparsers = parser.add_subparsers(dest="action", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--source-pvc", required=True)
    create.add_argument("--storage-class", default="premium-rwo")
    create.add_argument("--size", default="500Gi")
    create.add_argument("--zone")
    create.add_argument("--build-id", default=os.environ.get("BUILDKITE_BUILD_ID", "local"))
    create.add_argument("--timeout-seconds", type=int, default=600)
    create.add_argument(
        "--wait-bound",
        action="store_true",
        help="wait for Bound (do not use with WaitForFirstConsumer storage classes)",
    )
    create.add_argument("--dry-run", action="store_true")

    delete = subparsers.add_parser("delete")
    delete.add_argument("--name", required=True)

    args = parser.parse_args()
    try:
        if args.action == "create":
            claim = build_claim(
                name=args.name,
                source_pvc=args.source_pvc,
                storage_class=args.storage_class,
                size=args.size,
                build_id=args.build_id,
                zone=args.zone,
            )
            if args.dry_run:
                print(json.dumps(claim, indent=2, sort_keys=True))
                return 0
            client = KubernetesClient(args.namespace)
            client.create_claim(claim)
            if args.wait_bound:
                bound = client.wait_bound(args.name, args.timeout_seconds)
                volume_name = bound.get("spec", {}).get("volumeName", "unknown")
                print(f"PVC {args.name} is Bound to {volume_name}")
            else:
                print(
                    f"PVC {args.name} accepted; its first consumer will trigger binding")
        else:
            client = KubernetesClient(args.namespace)
            client.delete_claim(args.name)
            print(f"PVC {args.name} deletion requested")
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"PVC operation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
