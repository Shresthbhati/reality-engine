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


@pytest.mark.asyncio
async def test_job_start_licence_gate_rejects_non_commercial_model(tmp_path):
    """RE-SEC-000001: Job start rejects non-commercial models before execution."""
    from apps.api.jobs import _run_reconstruct_session
    from apps.api.models import Job, Session, World
    from unittest.mock import AsyncMock

    mock_db = AsyncMock()
    mock_db.get.side_effect = lambda model, obj_id: (
        Session(id="s1", world_id="w1") if model is Session else (
            World(id="w1", name="Test World") if model is World else None
        )
    )

    job = Job(
        id="job-noncom",
        type="RECONSTRUCT_SESSION",
        entity_id="s1",
        payload={"models": ["mast3r_vitlarge"]},
    )

    with pytest.raises(mr.LicenceNotAllowedError) as exc:
        await _run_reconstruct_session(mock_db, job)
    assert "mast3r_vitlarge" in str(exc.value)
    assert "CC-BY-NC" in str(exc.value)

