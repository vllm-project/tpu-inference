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


def test_cache_layout_rejects_extra_bare_metal_namespace(tmp_path):
    (tmp_path / "jax0.10.2_tputpu6e").mkdir()

    with pytest.raises(ValueError, match="extra namespace"):
        _MODULE.validate_compilation_cache_layout(tmp_path)


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


class _FakeUploadBlob:

    def __init__(self, name, uploads):
        self.name = name
        self.uploads = uploads

    def upload_from_filename(self, path, if_generation_match):
        assert if_generation_match == 0
        self.uploads.append((self.name, Path(path).read_bytes()))


class _FakeBucket:

    def __init__(self, uploads):
        self.uploads = uploads

    def blob(self, name):
        return _FakeUploadBlob(name, self.uploads)


class _FakeUploadClient:

    def __init__(self):
        self.uploads = []

    def bucket(self, name):
        assert name == "bucket"
        return _FakeBucket(self.uploads)


def test_publish_uploads_only_files_created_after_prepare(tmp_path):
    (tmp_path / "old").write_bytes(b"old")
    (tmp_path / "new").write_bytes(b"new")
    client = _FakeUploadClient()

    result = _MODULE.publish_new_cache_files(
        tmp_path,
        {"baseline_cache_files": ["old"]},
        "gs://bucket/cache/pr/123",
        storage_client=client,
    )

    assert result["new_files"] == 1
    assert result["uploaded"] == 1
    assert client.uploads == [("cache/pr/123/new", b"new")]


def test_publish_without_prefix_is_a_dry_run(tmp_path):
    (tmp_path / "new").write_bytes(b"new")

    result = _MODULE.publish_new_cache_files(
        tmp_path, {"baseline_cache_files": []}, None)

    assert result == {
        "new_files": 1,
        "uploaded": 0,
        "already_exists": 0,
        "write_prefix": None,
        "dry_run": True,
    }
