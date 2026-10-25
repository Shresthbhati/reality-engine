"""Real incremental registration: new photographs are REGISTERED into the previous sparse model
with ``colmap image_registrator`` (+ point_triangulator + bundle_adjuster) instead of re-solving
the scene, and the result is kept only if it is at least as good as a full rebuild.

Real COLMAP on the South Building photographs; nothing is mocked. The batches:

    A = [15:18] (3)   B = [18:21] (3)   SIX = [15:21] (6)   C = [21:24] (3)   D = [24:27] (3)

Measured facts these tests pin (COLMAP 4.2.0, CPU, robust SIFT):
  * a 3-photo prior model has ~67 points: too few 2D-3D matches to place new photographs, so
    image_registrator registers NONE and the full rebuild wins -- the fallback must say so
  * a 6-photo prior (~1130 points) accepts new photographs: 7 of 9 register incrementally, the same
    as a full rebuild, and the established cameras move < 1% of the scene extent
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.integration.test_progressive_product_journey import IMGS, pytestmark  # noqa: F401 -- marks + dataset

from evidence.session import EvidenceItem, EvidenceKind
from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
from reconstruction.colmap_session import ColmapSession

A, B, SIX, C, D = IMGS[15:18], IMGS[18:21], IMGS[15:21], IMGS[21:24], IMGS[24:27]


def items(paths):
    return [EvidenceItem(id="ev-" + p.stem, kind=EvidenceKind.PHOTO, source_uri=p.resolve().as_uri()) for p in paths]


def positions(result):
    return {p.evidence_id: np.asarray(p.position, float) for p in result.camera_poses}


def shift(before, after):
    """Movement of the cameras present in both, as a fraction of the scene extent (same raw COLMAP frame)."""
    pts = np.asarray(list(after.values()))
    extent = float(np.sqrt(np.mean(np.sum((pts - pts.mean(0)) ** 2, axis=1))))
    return {k: float(np.linalg.norm(after[k] - before[k])) / extent for k in before if k in after}


@pytest.fixture
def session(tmp_path):
    return ColmapSession(tmp_path / "colmap")


def test_rich_prior_is_extended_incrementally_and_established_cameras_stay_put(session):
    be = ColmapReconstructionBackend(session=session)
    first = be.reconstruct(items(SIX))
    assert be.last_run_info["mode"] == "full" and len(first.camera_poses) == 6
    assert session.commit()

    second = be.reconstruct(items(SIX + C))
    info = be.last_run_info
    assert info["mode"] == "incremental", info
    assert info["new_images"] == [f"ev-{p.stem}" for p in C] and info["prior_registered"] == 6
    assert {"image_registrator", "point_triangulator", "bundle_adjuster"} <= set(info["seconds"])
    assert info["incremental_registered"] == len(second.camera_poses) >= 7
    assert info["full_registered"] <= info["incremental_registered"]        # the full rebuild did not do better
    # new structure was actually triangulated (image_registrator alone adds none) while old points are kept
    assert len(second.points) > len(first.points)
    moved = shift(positions(first), positions(second))
    assert len(moved) == 6 and max(moved.values()) < 0.03, moved            # established geometry preserved


def test_a_prior_too_weak_to_place_new_photos_falls_back_to_a_full_rebuild_and_says_why(session):
    be = ColmapReconstructionBackend(session=session)
    be.reconstruct(items(A))
    assert session.commit()
    result = be.reconstruct(items(A + B))
    info = be.last_run_info
    assert info["mode"] == "full", info
    assert info["incremental_registered"] == 3 and info["full_registered"] == 6 == len(result.camera_poses)
    assert "registered more photographs" in info["reason"]


def test_a_discarded_run_does_not_advance_the_base(session):
    be = ColmapReconstructionBackend(session=session)
    be.reconstruct(items(SIX))
    session.commit()
    be.reconstruct(items(SIX + C))
    session.discard()                                                       # e.g. the candidate was rejected
    be.reconstruct(items(SIX + C))
    info = be.last_run_info
    assert info["prior_images"] == 6 and info["new_images"] == [f"ev-{p.stem}" for p in C], info
    assert info["mode"] == "incremental"


def test_committed_runs_chain_and_waiting_photos_are_retried(session):
    be = ColmapReconstructionBackend(session=session)
    be.reconstruct(items(SIX)); session.commit()
    second = be.reconstruct(items(SIX + C)); session.commit()
    waiting = be.last_run_info["unregistered"]
    assert len(waiting) >= 1                                                # some C photos could not be placed yet
    third = be.reconstruct(items(SIX + C + D))
    info = be.last_run_info
    assert info["prior_images"] == 9 and info["prior_registered"] == len(second.camera_poses)
    assert info["new_images"] == [f"ev-{p.stem}" for p in D]
    assert len(third.camera_poses) >= len(second.camera_poses)              # registration never collapses on new evidence
    # Established geometry must survive either way. A full re-solve (chosen only when it registers MORE
    # photographs, here 11 vs 10) starts a new coordinate gauge, so compare frame-independently.
    from engine.pipeline import world_delta

    cc = world_delta.camera_consistency({k: v.tolist() for k, v in positions(second).items()},
                                        {k: v.tolist() for k, v in positions(third).items()})
    assert cc is not None and cc["relative"] < 0.03 and max(cc["per_camera"].values()) < 0.05, cc
    assert info["frame"] == ("preserved" if info["mode"] == "incremental" else "re-solved")
    if info["mode"] == "incremental":
        assert max(shift(positions(second), positions(third)).values()) < 0.05


def test_changed_pipeline_settings_force_a_full_rebuild(session):
    ColmapReconstructionBackend(session=session).reconstruct(items(SIX))
    session.commit()
    be = ColmapReconstructionBackend(session=session, robust_sift=False)
    be.reconstruct(items(SIX))
    assert be.last_run_info["mode"] == "full" and "unusable" in be.last_run_info["reason"]


def test_no_new_photos_reuses_the_committed_model_without_recomputation(session):
    be = ColmapReconstructionBackend(session=session)
    first = be.reconstruct(items(SIX)); session.commit()
    again = be.reconstruct(items(SIX))
    info = be.last_run_info
    assert info["mode"] == "reused" and info["new_images"] == [] and "image_registrator" not in info["seconds"]
    assert len(again.camera_poses) == len(first.camera_poses)
    assert max(shift(positions(first), positions(again)).values()) < 1e-6
