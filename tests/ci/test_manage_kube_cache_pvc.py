from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.cpu_safe

_SCRIPT = (Path(__file__).resolve().parents[2] / ".buildkite" / "scripts" /
           "manage_kube_cache_pvc.py")
_SPEC = importlib.util.spec_from_file_location("manage_kube_cache_pvc", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize(
    "name",
    ["UPPER", "-leading", "trailing-", "has_underscore", "x" * 64],
)
def test_validate_name_rejects_invalid_dns_labels(name):
    with pytest.raises(ValueError, match="DNS label"):
        _MODULE.validate_name(name, "test name")


def test_build_claim_clones_golden_and_records_locality():
    claim = _MODULE.build_claim(
        name="bk-37-mlperf",
        source_pvc="tpu-cache-golden-pvc",
        storage_class="premium-rwo",
        size="500Gi",
        build_id="build-uuid",
        zone="southamerica-west1-a",
    )

    assert claim["spec"]["dataSource"] == {
        "apiGroup": "",
        "kind": "PersistentVolumeClaim",
        "name": "tpu-cache-golden-pvc",
    }
    assert claim["spec"]["resources"]["requests"]["storage"] == "500Gi"
    assert claim["metadata"]["annotations"] == {
        "tpu-inference.dev/source-pvc": "tpu-cache-golden-pvc",
        "tpu-inference.dev/build-id": "build-uuid",
        "tpu-inference.dev/expected-zone": "southamerica-west1-a",
    }


class _FakeClient(_MODULE.KubernetesClient):

    def __init__(self, responses):
        self.responses = iter(responses)

    def get_claim(self, name):
        return next(self.responses)


def test_wait_bound_returns_bound_claim(monkeypatch):
    monkeypatch.setattr(_MODULE.time, "sleep", lambda _: None)
    client = _FakeClient([
        (200, {"status": {"phase": "Pending"}}),
        (200, {"status": {"phase": "Bound"}, "spec": {"volumeName": "pv"}}),
    ])

    claim = client.wait_bound("bk-37-mlperf", 10)

    assert claim["spec"]["volumeName"] == "pv"
