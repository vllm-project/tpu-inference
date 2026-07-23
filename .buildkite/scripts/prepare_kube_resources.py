#!/usr/bin/env python3
"""Prepare declared CI models, datasets, and cache overlays on a CPU pod."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Iterable

DEFAULT_REGISTRY = Path(".buildkite/kubernetes/resource_registry.json")


def load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text())
    if not isinstance(registry, dict):
        raise ValueError("resource registry must be a JSON object")
    for section in ("models", "datasets"):
        if not isinstance(registry.get(section), dict):
            raise ValueError(f"resource registry requires an object named {section!r}")
    return registry


def _safe_destination(cache_root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe cache destination: {relative!r}")
    destination = (cache_root / Path(*path.parts)).resolve()
    try:
        destination.relative_to(cache_root.resolve())
    except ValueError as error:
        raise ValueError(f"cache destination escapes cache root: {relative!r}") from error
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_model(name: str, spec: dict[str, Any], cache_root: Path) -> dict[str, str]:
    repo_id = spec.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(f"model {name!r} requires a non-empty repo_id")
    revision = spec.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise ValueError(f"model {name!r} revision must be a string")

    from huggingface_hub import snapshot_download

    hf_home = cache_root / "hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=hf_home,
            token=os.environ.get("HF_TOKEN"),
        ))
    return {
        "name": name,
        "repo_id": repo_id,
        "requested_revision": revision or "main",
        "snapshot_path": str(snapshot_path),
        "resolved_revision": snapshot_path.name,
    }


def prepare_dataset(name: str, spec: dict[str, Any], cache_root: Path) -> dict[str, str]:
    relative = spec.get("destination")
    expected_sha = spec.get("sha256")
    prepare_script = spec.get("prepare_script")
    if not all(isinstance(value, str) and value
               for value in (relative, expected_sha, prepare_script)):
        raise ValueError(
            f"dataset {name!r} requires destination, sha256, and prepare_script")
    destination = _safe_destination(cache_root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.is_file() or _sha256(destination) != expected_sha:
        if destination.exists():
            destination.unlink()
        subprocess.run(
            [prepare_script, str(destination), expected_sha],
            check=True,
        )

    actual_sha = _sha256(destination) if destination.is_file() else "missing"
    if actual_sha != expected_sha:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"dataset {name!r} checksum mismatch: expected {expected_sha}, got {actual_sha}")
    return {
        "name": name,
        "path": str(destination),
        "sha256": actual_sha,
    }


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"GCS prefix must start with gs://: {uri!r}")
    bucket, separator, prefix = uri[5:].partition("/")
    if not bucket or not separator or not prefix.strip("/"):
        raise ValueError(f"GCS prefix requires bucket and object prefix: {uri!r}")
    return bucket, prefix.strip("/") + "/"


def hydrate_gcs_prefix(
    uri: str,
    destination: Path,
    *,
    storage_client: Any | None = None,
) -> dict[str, int]:
    """Download missing objects from a GCS prefix without replacing local files."""
    if storage_client is None:
        from google.cloud import storage
        storage_client = storage.Client()
    bucket_name, prefix = parse_gcs_uri(uri)
    bucket = storage_client.bucket(bucket_name)
    downloaded = skipped = 0
    for blob in storage_client.list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(prefix):]
        target = _safe_destination(destination, relative)
        if target.exists():
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        try:
            blob.download_to_filename(temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        downloaded += 1
    return {"downloaded": downloaded, "skipped": skipped}


def snapshot_cache_files(cache_path: Path) -> list[str]:
    if not cache_path.exists():
        return []
    return sorted(
        str(path.relative_to(cache_path)) for path in cache_path.rglob("*")
        if path.is_file() and not path.is_symlink())


def prepare_resources(
    registry: dict[str, Any],
    models: Iterable[str],
    datasets: Iterable[str],
    cache_read_prefixes: Iterable[str],
    cache_root: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {"models": [], "datasets": [], "cache_reads": []}
    for name in models:
        if name not in registry["models"]:
            raise ValueError(f"model is not registered for CI preparation: {name}")
        result["models"].append(
            prepare_model(name, registry["models"][name], cache_root))
    for name in datasets:
        if name not in registry["datasets"]:
            raise ValueError(f"dataset is not registered for CI preparation: {name}")
        result["datasets"].append(
            prepare_dataset(name, registry["datasets"][name], cache_root))

    compilation_cache = cache_root / "tpu_jax_cache"
    compilation_cache.mkdir(parents=True, exist_ok=True)
    for prefix in cache_read_prefixes:
        counts = hydrate_gcs_prefix(prefix, compilation_cache)
        result["cache_reads"].append({"prefix": prefix, **counts})
    result["baseline_cache_files"] = snapshot_cache_files(compilation_cache)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--cache-root", type=Path, default=Path("/cache"))
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--cache-read-prefix", action="append", default=[])
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("/cache/.buildkite/prepared-resources.json"),
    )
    args = parser.parse_args()

    try:
        registry = load_registry(args.registry)
        result = prepare_resources(
            registry,
            args.model,
            args.dataset,
            args.cache_read_prefix,
            args.cache_root,
        )
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"resource preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
