import pytest

from provenance import Provenance, Provenanced, Uncertainty


def test_confidence_bounds_enforced():
    Uncertainty(confidence=0.0)
    Uncertainty(confidence=1.0)
    with pytest.raises(ValueError):
        Uncertainty(confidence=1.1)
    with pytest.raises(ValueError):
        Uncertainty(confidence=-0.1)


@pytest.mark.parametrize(
    "provenance,expected_canonical",
    [
        (Provenance.OBSERVED, True),
        (Provenance.RECONSTRUCTED, True),
        (Provenance.ESTIMATED, True),
        (Provenance.INFERRED, True),
        (Provenance.GENERATED, False),
        (Provenance.UNKNOWN, False),
        (Provenance.CONFLICT, False),
    ],
)
def test_is_canonical(provenance, expected_canonical):
    p = Provenanced(value=1.0, provenance=provenance)
    assert p.is_canonical() is expected_canonical


def test_roundtrip_dict():
    p = Provenanced(value=42, provenance=Provenance.OBSERVED, uncertainty=Uncertainty(0.9, "lidar"))
    restored = Provenanced.from_dict(p.to_dict())
    assert restored == p


def test_uncertainty_calibrated_default_and_roundtrip():
    """RE-GEO-000013: Default uncertainty is uncalibrated unless explicitly calibrated."""
    u_default = Uncertainty(0.7)
    assert not u_default.calibrated
    assert "calibrated" not in u_default.to_dict()

    u_cal = Uncertainty(0.95, "laser_measured", calibrated=True)
    assert u_cal.calibrated
    d = u_cal.to_dict()
    assert d.get("calibrated") is True
    restored = Uncertainty.from_dict(d)
    assert restored == u_cal

