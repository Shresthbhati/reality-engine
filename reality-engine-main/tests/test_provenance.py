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
