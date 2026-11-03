"""Tests for Blueprint Task 4.1: Model Registry and Licenses.

Verifies:
- Every declared model has license, commercial_ok flag, measured_vram_mb under 5.2GB ceiling, and valid status.
- .agent/LICENSES.yaml is synchronized with declared models.
- Model registry dump contains required metadata.
"""

from pathlib import Path
import yaml
import pytest
from perception.model_registry import all_specs, get, ModelSpec


def test_model_registry_spec_fields():
    specs = all_specs()
    assert len(specs) >= 10, f"Expected at least 10 models, got {len(specs)}"

    for name, spec in specs.items():
        assert isinstance(spec, ModelSpec)
        assert spec.name == name
        assert spec.task, f"Model {name} missing task"
        assert spec.license, f"Model {name} missing license"
        assert isinstance(spec.commercial_ok, bool), f"Model {name} invalid commercial_ok"
        assert spec.commercial_ok is True, f"Model {name} not marked commercial_ok"
        assert 0.0 <= spec.measured_vram_mb <= 5200.0, (
            f"Model {name} VRAM {spec.measured_vram_mb} MB exceeds RTX 4050 5,200 MB ceiling"
        )
        assert spec.status in ("active", "candidate", "deprecated"), f"Model {name} invalid status {spec.status}"


def test_model_registry_serialization():
    specs = all_specs()
    for name, spec in specs.items():
        d = spec.to_dict()
        assert d["name"] == name
        assert "license" in d
        assert "commercial_ok" in d
        assert "measured_vram_mb" in d
        assert "status" in d


def test_licenses_yaml_sync():
    license_path = Path(__file__).resolve().parent.parent / ".agent" / "LICENSES.yaml"
    assert license_path.exists(), "LICENSES.yaml not found"

    with open(license_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "models" in data, "models section missing from LICENSES.yaml"
    yaml_models = {m["model"]: m for m in data["models"]}

    specs = all_specs()
    for name, spec in specs.items():
        assert name in yaml_models, f"Model {name} missing from .agent/LICENSES.yaml"
        ym = yaml_models[name]
        assert ym["license"] == spec.license, f"License mismatch for {name}: {ym['license']} vs {spec.license}"
        assert ym["commercial_ok"] == spec.commercial_ok
        assert ym["measured_vram_mb"] == spec.measured_vram_mb
        assert ym["status"] == spec.status
