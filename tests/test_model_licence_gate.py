import pytest

from perception import model_registry as mr


def test_allowed_model_passes():
    assert mr.assert_production_licences(["sam2_hiera_tiny"]) is None


def test_non_commercial_model_rejected():
    with pytest.raises(mr.LicenceNotAllowedError) as exc:
        mr.assert_production_licences(["mast3r_vitlarge"])
    assert "CC-BY-NC" in str(exc.value)
    assert "mast3r_vitlarge" in str(exc.value)


def test_unknown_model_raises_keyerror():
    with pytest.raises(KeyError):
        mr.assert_production_licences(["no_such_model"])


def test_every_registry_entry_has_a_licence():
    empty = [n for n, s in mr.all_specs().items() if not s.license.strip()]
    assert empty == []
