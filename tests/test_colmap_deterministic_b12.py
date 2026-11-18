import pytest
from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
from reconstruction.backend.colmap_config import ColmapConfig


def test_colmap_deterministic_policy_defaults():
    # By default, <= 60 images is deterministic
    b = ColmapReconstructionBackend()
    assert b.is_deterministic(10) is True
    assert b.is_deterministic(60) is True
    assert b.is_deterministic(61) is False
    assert b.is_deterministic(120) is False


def test_colmap_deterministic_override():
    # Explicit True forces deterministic even for 200 images
    b_true = ColmapReconstructionBackend(deterministic=True)
    assert b_true.is_deterministic(200) is True

    # Explicit False disables deterministic even for 5 images
    b_false = ColmapReconstructionBackend(deterministic=False)
    assert b_false.is_deterministic(5) is False


def test_colmap_config_deterministic_field():
    cfg = ColmapConfig(deterministic=True)
    b = ColmapReconstructionBackend(config=cfg)
    assert b.is_deterministic(100) is True


def test_colmap_version_query():
    b = ColmapReconstructionBackend()
    ver = b._get_colmap_version()
    assert "COLMAP" in ver
