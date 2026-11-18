"""RE-SEC-000003: weights are hash-checked against the registry before they are loaded."""
import hashlib

import pytest

from perception import model_registry as mr


def test_require_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "_REGISTRY", {})
    ck = tmp_path / "w.pt"
    ck.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()

    mr.require_verified(ck)  # not registered: nothing to compare, passes
    mr.record_pretrained("m", "t", "manual", ck, sha256=good)
    mr.require_verified(ck)  # matches
    mr.record_pretrained("m", "t", "manual", ck, sha256="0" * 64)
    with pytest.raises(mr.CheckpointMismatch):
        mr.require_verified(ck)
