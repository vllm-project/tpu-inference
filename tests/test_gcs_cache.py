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

"""Tests for GCS JAX compilation cache utilities in tpu_inference."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock

import types
import pytest

from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

class _MockLoader(Loader):
    def create_module(self, spec):
        m = mock.MagicMock()
        m.__name__ = spec.name
        m.__path__ = []
        m.__loader__ = self
        m.__spec__ = spec
        if spec.name in ("vllm.v1.worker.worker_base", "vllm.worker.worker_base"):
            m.WorkerBase = type("WorkerBase", (), {})
            m.CompilationTimes = type("CompilationTimes", (), {})
        return m

    def exec_module(self, module):
        pass

class _MockFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if (
            fullname in ("vllm", "torch", "torchax")
            or fullname.startswith(("vllm.", "torch.", "torchax."))
        ):
            return ModuleSpec(fullname, _MockLoader(), is_package=True)
        return None

sys.meta_path.insert(0, _MockFinder())

from tpu_inference import gcs_cache


class TestGcsCache:

    def test_parse_gcs_uri(self):
        bucket, prefix = gcs_cache._parse_gcs_uri("gs://my-bucket/path/to/cache")
        assert bucket == "my-bucket"
        assert prefix == "path/to/cache/"

        bucket, prefix = gcs_cache._parse_gcs_uri("gs://my-bucket")
        assert bucket == "my-bucket"
        assert prefix == ""

        bucket, prefix = gcs_cache._parse_gcs_uri("gs://my-bucket/single-dir/")
        assert bucket == "my-bucket"
        assert prefix == "single-dir/"

        with pytest.raises(ValueError):
            gcs_cache._parse_gcs_uri("https://not-gcs.com")

        with pytest.raises(ValueError):
            gcs_cache._parse_gcs_uri("/local/path")

    def test_ensure_jax_cache_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "test_cache"
            res = gcs_cache.ensure_jax_cache_env(target_dir)
            assert res == target_dir
            assert target_dir.is_dir()
            assert os.environ.get("JAX_COMPILATION_CACHE_DIR") == str(target_dir)
            assert os.environ.get("VLLM_XLA_CACHE_PATH") == str(target_dir)

    def test_ensure_jax_cache_env_updates_jax_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "test_cache"
            mock_jax = mock.MagicMock()
            with mock.patch.dict(sys.modules, {"jax": mock_jax}):
                res = gcs_cache.ensure_jax_cache_env(target_dir)
                assert res == target_dir
                mock_jax.config.update.assert_called_once_with(
                    "jax_compilation_cache_dir", str(target_dir)
                )

    @mock.patch.object(gcs_cache, "download_cache", return_value=True)
    def test_restore_jax_cache_with_explicit_uri(self, mock_download):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            success = gcs_cache.restore_jax_cache(
                gcs_uri="gs://bucket/test_restore_explicit",
                local_dir=cache_dir,
                force=True,
            )
            assert success is True
            mock_download.assert_called_once_with(
                cache_dir, "gs://bucket/test_restore_explicit", max_workers=8
            )

    @mock.patch.object(gcs_cache, "download_cache", return_value=True)
    def test_restore_jax_cache_from_env(self, mock_download):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            with mock.patch.dict(
                os.environ, {"JAX_CACHE_GCS_DIR": "gs://bucket/test_restore_env"}
            ):
                success = gcs_cache.restore_jax_cache(
                    local_dir=cache_dir, force=True
                )
                assert success is True
                mock_download.assert_called_once_with(
                    cache_dir, "gs://bucket/test_restore_env", max_workers=8
                )

    def test_restore_jax_cache_no_uri(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            with mock.patch.dict(
                os.environ,
                {
                    "JAX_CACHE_GCS_DIR": "",
                    "ROLLOUT_JAX_CACHE_GCS_DIR": "",
                    "VLLM_JAX_CACHE_GCS_DIR": "",
                },
            ):
                success = gcs_cache.restore_jax_cache(
                    gcs_uri="", local_dir=cache_dir
                )
                assert success is False

    def test_save_jax_cache_disabled(self):
        with mock.patch.dict(os.environ, {"SAVE_JAX_CACHE": "false"}):
            success = gcs_cache.save_jax_cache(gcs_uri="gs://bucket/test")
            assert success is False

    @mock.patch.object(gcs_cache, "upload_cache", return_value=True)
    def test_save_jax_cache_success(self, mock_upload):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            with mock.patch.dict(os.environ, {"SAVE_JAX_CACHE": "true"}):
                success = gcs_cache.save_jax_cache(
                    gcs_uri="gs://bucket/saved_cache",
                    local_dir=cache_dir,
                )
                assert success is True
                mock_upload.assert_called_once_with(
                    cache_dir, "gs://bucket/saved_cache", max_workers=8
                )

    def test_download_cache_transfer_manager_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            mock_client = mock.MagicMock()
            mock_bucket = mock.MagicMock()
            mock_client.bucket.return_value = mock_bucket

            blob1 = mock.MagicMock()
            blob1.name = "cache/obj1"
            blob2 = mock.MagicMock()
            blob2.name = "cache/obj2"
            mock_client.list_blobs.return_value = [blob1, blob2]

            with mock.patch("google.cloud.storage.Client", return_value=mock_client), \
                 mock.patch("google.cloud.storage.transfer_manager.download_many_to_path", return_value=[None, None]) as mock_tm:
                success = gcs_cache.download_cache(cache_dir, "gs://bucket/cache")
                assert success is True
                mock_tm.assert_called_once_with(
                    mock_bucket,
                    ["obj1", "obj2"],
                    destination_directory=str(cache_dir),
                    blob_name_prefix="cache/",
                    max_workers=8,
                )

    def test_download_cache_empty_bucket(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            mock_client = mock.MagicMock()
            mock_client.list_blobs.return_value = []
            with mock.patch("google.cloud.storage.Client", return_value=mock_client), \
                 mock.patch("google.cloud.storage.transfer_manager.download_many_to_path") as mock_tm:
                success = gcs_cache.download_cache(cache_dir, "gs://bucket/empty")
                assert success is True
                mock_tm.assert_not_called()

    @mock.patch("shutil.which")
    @mock.patch("subprocess.run")
    def test_download_cache_fallback_gsutil(self, mock_run, mock_which):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            mock_which.side_effect = lambda cmd: "/usr/bin/gsutil" if cmd == "gsutil" else None
            mock_run.return_value = mock.MagicMock(returncode=0)

            with mock.patch.dict(
                sys.modules,
                {
                    "google.cloud": None,
                    "google.cloud.storage": None,
                },
            ):
                success = gcs_cache.download_cache(cache_dir, "gs://bucket/cache")
                assert success is True
                mock_run.assert_called_once_with(
                    ["gsutil", "-m", "rsync", "-r", "gs://bucket/cache", str(cache_dir)],
                    check=False,
                )

    @mock.patch("shutil.which")
    @mock.patch("subprocess.run")
    def test_download_cache_fallback_gcloud(self, mock_run, mock_which):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            mock_which.side_effect = lambda cmd: "/usr/bin/gcloud" if cmd == "gcloud" else None
            mock_run.return_value = mock.MagicMock(returncode=0)

            with mock.patch.dict(
                sys.modules,
                {
                    "google.cloud": None,
                    "google.cloud.storage": None,
                },
            ):
                success = gcs_cache.download_cache(cache_dir, "gs://bucket/cache")
                assert success is True
                mock_run.assert_called_once_with(
                    ["gcloud", "storage", "rsync", "-r", "gs://bucket/cache", str(cache_dir)],
                    check=False,
                )

    def test_upload_cache_transfer_manager_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "obj1").write_text("dummy")

            mock_client = mock.MagicMock()
            mock_bucket = mock.MagicMock()
            mock_client.bucket.return_value = mock_bucket
            with mock.patch("google.cloud.storage.Client", return_value=mock_client), \
                 mock.patch("google.cloud.storage.transfer_manager.upload_many_from_filenames", return_value=[None]) as mock_tm:
                success = gcs_cache.upload_cache(cache_dir, "gs://bucket/cache")
                assert success is True
                mock_tm.assert_called_once_with(
                    mock_bucket,
                    ["obj1"],
                    source_directory=str(cache_dir),
                    blob_name_prefix="cache/",
                    skip_if_exists=True,
                    max_workers=8,
                )

    def test_upload_cache_transfer_manager_skips_precondition_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "obj1").write_text("dummy")

            mock_client = mock.MagicMock()
            mock_bucket = mock.MagicMock()
            mock_client.bucket.return_value = mock_bucket

            # PreconditionFailed exception (code 412) simulates already exists
            class FakePreconditionFailed(Exception):
                code = 412

            with mock.patch("google.cloud.storage.Client", return_value=mock_client), \
                 mock.patch("google.cloud.storage.transfer_manager.upload_many_from_filenames", return_value=[FakePreconditionFailed()]):
                success = gcs_cache.upload_cache(cache_dir, "gs://bucket/cache")
                assert success is True

    def test_upload_cache_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Empty cache dir returns True with warning without attempting network call
            success = gcs_cache.upload_cache(cache_dir, "gs://bucket/cache")
            assert success is True

    @mock.patch("shutil.which")
    @mock.patch("subprocess.run")
    def test_upload_cache_fallback_gsutil(self, mock_run, mock_which):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "obj1").write_text("data")

            mock_which.side_effect = lambda cmd: "/usr/bin/gsutil" if cmd == "gsutil" else None
            mock_run.return_value = mock.MagicMock(returncode=0)

            with mock.patch.dict(
                sys.modules,
                {
                    "google.cloud": None,
                    "google.cloud.storage": None,
                },
            ):
                success = gcs_cache.upload_cache(cache_dir, "gs://bucket/cache")
                assert success is True
                mock_run.assert_called_once_with(
                    ["gsutil", "-m", "rsync", "-r", str(cache_dir), "gs://bucket/cache"],
                    check=False,
                )

    @mock.patch("shutil.which")
    @mock.patch("subprocess.run")
    def test_upload_cache_fallback_gcloud(self, mock_run, mock_which):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "obj1").write_text("data")

            mock_which.side_effect = lambda cmd: "/usr/bin/gcloud" if cmd == "gcloud" else None
            mock_run.return_value = mock.MagicMock(returncode=0)

            with mock.patch.dict(
                sys.modules,
                {
                    "google.cloud": None,
                    "google.cloud.storage": None,
                },
            ):
                success = gcs_cache.upload_cache(cache_dir, "gs://bucket/cache")
                assert success is True
                mock_run.assert_called_once_with(
                    ["gcloud", "storage", "rsync", "-r", str(cache_dir), "gs://bucket/cache"],
                    check=False,
                )

    def test_cli_main_download(self):
        with mock.patch.object(gcs_cache, "download_cache", return_value=True) as mock_dl:
            with mock.patch("sys.exit") as mock_exit:
                gcs_cache.main(["download", "/tmp/local_cache", "gs://bucket/cache"])
                mock_exit.assert_called_once_with(0)
                mock_dl.assert_called_once_with(
                    "/tmp/local_cache", "gs://bucket/cache", max_workers=8
                )

    def test_cli_main_upload(self):
        with mock.patch.object(gcs_cache, "upload_cache", return_value=True) as mock_ul:
            with mock.patch("sys.exit") as mock_exit:
                gcs_cache.main(["upload", "/tmp/local_cache", "gs://bucket/cache"])
                mock_exit.assert_called_once_with(0)
                mock_ul.assert_called_once_with(
                    "/tmp/local_cache", "gs://bucket/cache", max_workers=8
                )

    def test_register_auto_save(self):
        with mock.patch("atexit.register") as mock_atexit:
            gcs_cache._AUTO_SAVE_REGISTERED = False
            gcs_cache.register_auto_save(gcs_uri="gs://bucket/cache")
            mock_atexit.assert_called_once_with(
                gcs_cache.save_jax_cache, gcs_uri="gs://bucket/cache", local_dir=None
            )
            # Idempotent registration
            gcs_cache.register_auto_save(gcs_uri="gs://bucket/cache")
            assert mock_atexit.call_count == 1



