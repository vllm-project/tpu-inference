#!/usr/bin/env python3
"""Publish newly compiled JAX/XLA cache entries from a Kubernetes TPU job to GCS.

The golden PVC gives every job a warm read-only base. Nothing carries the other
way: entries compiled during a build die with the job's ephemeral PVC, so a
shape the golden does not cover is recompiled on every single build forever.

This closes that loop for the production environment:

    snapshot   record what the golden clone already contained
    publish    upload only what this job added, and only if it passed

Uploads use ``if_generation_match=0``. JAX cache entries are content-addressed,
so two jobs that compile the same shape produce the same object name; the
precondition makes the loser a no-op instead of a racing overwrite. That is the
same property the bare-metal ``gcloud storage rsync --no-clobber`` relies on.

The Kubernetes test image is built with ``BM_INFRA=false`` and therefore has no
gcloud CLI, so transfers use google-cloud-storage from requirements.txt.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any

# Cache trees are tens of thousands of small objects; serial uploads would add
# minutes to every job. The work is network-bound, so threads are enough.
DEFAULT_WORKERS = 32


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"GCS prefix must start with gs://: {uri!r}")
    bucket, separator, prefix = uri[5:].partition("/")
    if not bucket or not separator or not prefix.strip("/"):
        raise ValueError(f"GCS prefix requires bucket and object prefix: {uri!r}")
    return bucket, prefix.strip("/") + "/"


def snapshot_cache_files(cache_path: Path) -> list[str]:
    if not cache_path.exists():
        return []
    return sorted(
        str(path.relative_to(cache_path)) for path in cache_path.rglob("*")
        if path.is_file() and not path.is_symlink())


def validate_cache_layout(cache_path: Path) -> None:
    """Reject a golden that was restored one directory too deep.

    The bare-metal cache lives under gs://.../jax<VER>_tpu<GEN>/. Copying that
    directory itself, rather than its contents, produces
    /cache/tpu_jax_cache/jax0.11.0_tputpu6e/... which JAX silently ignores: the
    build looks warm, every lookup misses, and the only symptom is a slow job.
    """
    if not cache_path.exists():
        return
    nested = sorted(path.name for path in cache_path.iterdir()
                    if path.is_dir() and path.name.startswith("jax") and "_tpu" in path.name)
    if nested:
        raise ValueError(
            f"compilation cache has an extra namespace directory under {cache_path}: "
            f"{', '.join(nested)}; copy that directory's contents directly into "
            "the configured cache path")


def new_cache_files(cache_path: Path, manifest: dict[str, Any]) -> list[Path]:
    baseline = manifest.get("baseline_cache_files")
    if not isinstance(baseline, list) or not all(isinstance(e, str) for e in baseline):
        raise ValueError("manifest has no valid baseline_cache_files list")
    known = set(baseline)
    return [
        cache_path / relative for relative in snapshot_cache_files(cache_path)
        if relative not in known
    ]


def publish(
    cache_path: Path,
    manifest: dict[str, Any],
    write_prefix: str | None,
    *,
    storage_client: Any | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    files = new_cache_files(cache_path, manifest)
    result: dict[str, Any] = {
        "new_files": len(files),
        "uploaded": 0,
        "already_exists": 0,
        "write_prefix": write_prefix,
    }
    if not write_prefix:
        result["dry_run"] = True
        return result
    if storage_client is None:
        from google.cloud import storage
        storage_client = storage.Client()
    bucket_name, prefix = parse_gcs_uri(write_prefix)
    bucket = storage_client.bucket(bucket_name)

    def upload(path: Path) -> bool:
        blob = bucket.blob(prefix + path.relative_to(cache_path).as_posix())
        try:
            blob.upload_from_filename(path, if_generation_match=0)
            return True
        except Exception as error:
            # 412 means another job already published this content-addressed
            # entry. Checked via getattr so google.api_core is not needed in
            # dry-run or unit-test environments.
            if getattr(error, "code", None) == 412:
                return False
            raise

    if workers <= 1 or len(files) <= 1:
        outcomes = [upload(path) for path in files]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(files))) as pool:
            outcomes = list(pool.map(upload, files))

    result["uploaded"] = sum(1 for created in outcomes if created)
    result["already_exists"] = sum(1 for created in outcomes if not created)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["snapshot", "publish"])
    parser.add_argument("--cache-path", type=Path, default=Path("/cache/tpu_jax_cache"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("/cache/.buildkite/jax-cache-manifest.json"))
    parser.add_argument("--write-prefix")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    try:
        if args.mode == "snapshot":
            args.cache_path.mkdir(parents=True, exist_ok=True)
            validate_cache_layout(args.cache_path)
            baseline = snapshot_cache_files(args.cache_path)
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(
                json.dumps({"baseline_cache_files": baseline}, indent=2) + "\n")
            result = {"baseline_cache_file_count": len(baseline)}
        else:
            manifest = json.loads(args.manifest.read_text())
            result = publish(args.cache_path, manifest, args.write_prefix,
                             workers=args.workers)
    except (OSError, ValueError) as error:
        print(f"cache {args.mode} failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
