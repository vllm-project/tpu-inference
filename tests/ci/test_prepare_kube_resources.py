from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.cpu_safe

_SCRIPT = (Path(__file__).resolve().parents[2] / ".buildkite" / "scripts" /
           "prepare_kube_resources.py")
_SPEC = importlib.util.spec_from_file_location("prepare_kube_resources", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_load_registry_requires_sections(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"models": {}}))
    with pytest.raises(ValueError, match="datasets"):
        _MODULE.load_registry(registry)


@pytest.mark.parametrize("relative", ["/absolute", "../escape", "a/../../escape"])
def test_safe_destination_rejects_escape(tmp_path, relative):
    with pytest.raises(ValueError, match="unsafe|escapes"):
        _MODULE._safe_destination(tmp_path, relative)


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("gs://bucket/cache/global", ("bucket", "cache/global/")),
        ("gs://bucket/cache/pr/123/", ("bucket", "cache/pr/123/")),
    ],
)
def test_parse_gcs_uri(uri, expected):
    assert _MODULE.parse_gcs_uri(uri) == expected


class _FakeBlob:

    def __init__(self, name, content):
        self.name = name
        self.content = content

    def download_to_filename(self, path):
        Path(path).write_bytes(self.content)


class _FakeStorageClient:

    def __init__(self, blobs):
        self.blobs = blobs

    def bucket(self, name):
        return name

    def list_blobs(self, bucket, prefix):
        assert bucket == "bucket"
        return [blob for blob in self.blobs if blob.name.startswith(prefix)]


def test_hydrate_gcs_prefix_merges_without_overwrite(tmp_path):
    existing = tmp_path / "existing"
    existing.write_bytes(b"local")
    client = _FakeStorageClient([
        _FakeBlob("cache/global/existing", b"remote"),
        _FakeBlob("cache/global/new/entry", b"new"),
    ])

    counts = _MODULE.hydrate_gcs_prefix(
        "gs://bucket/cache/global", tmp_path, storage_client=client)

    assert counts == {"downloaded": 1, "skipped": 1}
    assert existing.read_bytes() == b"local"
    assert (tmp_path / "new" / "entry").read_bytes() == b"new"


def test_prepare_dataset_reuses_verified_file(tmp_path, monkeypatch):
    content = b"verified dataset"
    destination = tmp_path / "datasets" / "sample.bin"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(content)
    spec = {
        "destination": "datasets/sample.bin",
        "prepare_script": "unused",
        "sha256": _MODULE.hashlib.sha256(content).hexdigest(),
    }
    monkeypatch.setattr(_MODULE.subprocess, "run",
                        lambda *args, **kwargs: pytest.fail(
                            "verified datasets must not run the downloader"))

    result = _MODULE.prepare_dataset("sample", spec, tmp_path)

    assert result["path"] == str(destination)
    assert result["sha256"] == spec["sha256"]
