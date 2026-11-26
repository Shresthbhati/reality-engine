"""Upload staging path security (Task 5: path traversal prevention).

The staging filename is derived from a caller-supplied upload id. The storage
layer must reject anything that could escape staging_root(), even if an
upstream route failed to validate. This is defense in depth: DB visibility
checks still apply, but they are not the only barrier.
"""
from __future__ import annotations

import pytest

from apps.api.storage import staging_path, validate_staging_id


@pytest.mark.parametrize(
    "bad",
    [
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        "a/b",
        "a\\b",
        ".",
        "..",
        "",
        "a\x00b",
        "%2e%2e%2fsecret",
        "upl_abc/../../x",
        "x" * 65,
        12345,
        None,
    ],
)
def test_validate_staging_id_rejects_traversal(bad):
    with pytest.raises(ValueError, match="invalid upload id|must be a string"):
        validate_staging_id(bad)


@pytest.mark.parametrize("good", ["upl_abcdef123456", "upl_0123456789ab", "ab_-CD09"])
def test_validate_staging_id_accepts_issued_shape(good):
    assert validate_staging_id(good) == good


def test_staging_path_is_confined_to_staging_root(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    import importlib

    import apps.api.storage as storage

    importlib.reload(storage)
    p = storage.staging_path("upl_abcdef123456")
    assert p.parent == storage.staging_root()
    assert p.name == "upl_abcdef123456.partial"
    # A traversal attempt must never produce a path outside staging_root.
    with pytest.raises(ValueError):
        storage.staging_path("../../outside")


def test_stage_chunk_rejects_traversal_without_writing(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    monkeypatch.setenv("STORAGE_ROOT", str(root))
    import importlib

    import apps.api.storage as storage

    importlib.reload(storage)

    with pytest.raises(ValueError):
        storage.stage_chunk("../escape", 0, b"data")
    # Nothing escaped into the parent of the artifacts dir.
    assert not (tmp_path / "escape.partial").exists()
