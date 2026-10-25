"""Spatial continuity: a change of REPRESENTATION must not be reported as a change of the WORLD.

Synthetic lattices with known ground truth, so every relation is provable. Generic on purpose:
nothing here is specific to any dataset or building part.
"""

from __future__ import annotations

import math

import numpy as np

from engine.pipeline import spatial_continuity as sc

EXT = 10.0          # scene extent used for the relative tolerances
IDENT = lambda a: a  # noqa: E731 -- candidate already in the previous frame


def lattice(x0, x1, y0, y1, nx=None, ny=None, spacing=0.3, origin=(0, 0, 0), u=(1, 0, 0), v=(0, 1, 0)):
    """Regular grid of points on a rectangle spanned by unit vectors u, v starting at origin."""
    nx = nx or max(3, int(round((x1 - x0) / spacing)) + 1)
    ny = ny or max(3, int(round((y1 - y0) / spacing)) + 1)
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    o, u, v = np.asarray(origin, float), np.asarray(u, float), np.asarray(v, float)
    return np.array([o + a * u + b * v for a in xs for b in ys])


def rec(rid, typ, pts, evidence=("e1",)):
    pts = np.asarray(pts, float)
    return {"id": rid, "type": typ, "center": pts.mean(0).tolist(), "points": pts.tolist(),
            "bounds": [pts.min(0).tolist(), pts.max(0).tolist()],
            "evidence": None if evidence is None else list(evidence), "provenance": "RECONSTRUCTED",
            "confidence": 0.8}


def recon(prev, cand, new_evidence=(), to_prev=IDENT, scale=1.0):
    return sc.reconcile(prev, cand, ext_prev=EXT, to_prev=to_prev, scale=scale, new_evidence_ids=new_evidence)


def only(res, kind):
    return [r for r in res["relations"] if r["kind"] == kind]


WALL = lattice(0, 6, 0, 2)        # a 6 x 2 wall in the z = 0 plane


def test_same_surface_resampled_is_preserved_even_when_point_density_differs():
    dense = lattice(0, 6, 0, 2, nx=40, ny=14)
    res = recon([rec("w", "wall", WALL)], [rec("w2", "wall", dense)])
    assert res["counts"]["preserved"] == 1 and sum(res["counts"].values()) == 1, res["relations"]


def test_small_shift_and_tilt_within_tolerance_is_refined_not_removed_and_new():
    shifted = WALL + np.array([0, 0, 0.6])            # 0.06 of the extent along the normal
    res = recon([rec("w", "wall", WALL)], [rec("w2", "wall", shifted)])
    (r,) = only(res, "refined")
    assert r["gap_rel"] > sc.PRESERVED_SHIFT_REL and r["cover_prev"] >= 0.6 and not only(res, "removed") and not only(res, "new")


def test_orientation_change_is_measured():
    th = math.radians(10)
    tilted = lattice(0, 6, 0, 2, u=(math.cos(th), 0, math.sin(th)))
    res = recon([rec("w", "wall", WALL)], [rec("w2", "wall", tilted)])
    (r,) = only(res, "refined")
    assert 8.0 < r["angle_deg"] < 12.0


def test_split_one_wall_into_two_fragments_is_a_representation_change():
    a1, a2 = lattice(0, 3, 0, 2), lattice(3.05, 6, 0, 2)
    res = recon([rec("w", "wall", WALL)], [rec("f1", "wall", a1), rec("f2", "wall", a2)])
    (r,) = only(res, "split")
    assert len(r["cand"]) == 2 and r["cover_prev"] >= 0.6 and "not of the world" in r["explanation"]
    assert not only(res, "removed") and not only(res, "new")


def test_merge_two_walls_into_one_surface():
    a1, a2 = lattice(0, 3, 0, 2), lattice(3.05, 6, 0, 2)
    res = recon([rec("f1", "wall", a1), rec("f2", "wall", a2)], [rec("w", "wall", WALL)])
    (r,) = only(res, "merge")
    assert len(r["prev"]) == 2 and not only(res, "removed") and not only(res, "new")


def test_new_geometry_and_true_removal_are_reported_as_such():
    far = lattice(0, 6, 0, 2, origin=(0, 0, 5))             # a parallel wall 0.5 x extent away
    res = recon([rec("w", "wall", WALL)], [rec("n", "wall", far)])
    assert len(only(res, "removed")) == 1 and len(only(res, "new")) == 1
    assert only(res, "removed")[0]["prev"] == ["w"] and only(res, "new")[0]["cand"] == ["n"]


def test_a_candidate_that_reaches_further_is_extended_and_one_that_covers_less_is_reduced():
    bigger = lattice(0, 12, 0, 2)
    assert only(recon([rec("w", "wall", WALL)], [rec("b", "wall", bigger)]), "extended")
    smaller = lattice(0, 2, 0, 2)
    (r,) = only(recon([rec("w", "wall", WALL)], [rec("s", "wall", smaller)]), "reduced")
    assert r["cover_prev"] < 0.6


def test_different_entity_types_never_match():
    res = recon([rec("w", "wall", WALL)], [rec("f", "floor", WALL)])
    assert len(only(res, "removed")) == 1 and len(only(res, "new")) == 1


def test_candidate_in_another_frame_is_matched_after_alignment():
    th = 0.8
    R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    s, t = 2.5, np.array([10.0, -4.0, 3.0])
    cand_pts = (s * (WALL @ R.T)) + t                       # the candidate's own frame
    to_prev = lambda a: ((a - t) @ R) / s                    # noqa: E731 -- its inverse
    res = recon([rec("w", "wall", WALL)], [rec("w2", "wall", cand_pts)], to_prev=to_prev, scale=s)
    assert res["counts"]["preserved"] == 1 and sum(res["counts"].values()) == 1, res["relations"]


def test_entities_without_points_fall_back_to_centre_matching_and_say_so():
    door = {"id": "d", "type": "door", "center": [1.0, 1.0, 0.0], "bounds": [[0.8, 0, 0], [1.2, 2, 0]],
            "evidence": None, "provenance": "INFERRED", "confidence": 0.2}
    moved = dict(door, id="d2", center=[1.05, 1.0, 0.0])
    res = recon([door], [moved])
    (r,) = res["relations"]
    assert r["signal"] == "center_only" and r["kind"] in ("preserved", "refined") and res["signals"] == {"center_only": 1}


def test_an_unsupported_move_is_flagged_but_a_move_backed_by_new_evidence_is_not():
    shifted = WALL + np.array([0, 0, 0.8])
    prev = [rec("w", "wall", WALL, evidence=("e1", "e2"))]
    unsupported = recon(prev, [rec("w2", "wall", shifted, evidence=("e1", "e2"))], new_evidence=("e3",))
    supported = recon(prev, [rec("w2", "wall", shifted, evidence=("e1", "e2", "e3"))], new_evidence=("e3",))
    assert only(unsupported, "refined")[0]["unsupported_move"] is True
    (r,) = only(supported, "refined")
    assert r["unsupported_move"] is False and r["supporting_new_evidence"] == ["e3"]


def test_unknown_evidence_never_produces_a_conflict_claim():
    shifted = WALL + np.array([0, 0, 0.8])
    res = recon([rec("w", "wall", WALL, evidence=None)], [rec("w2", "wall", shifted, evidence=None)])
    (r,) = only(res, "refined")
    assert r["supporting_new_evidence"] is None and r["unsupported_move"] is False


def test_regions_group_nearby_structure_and_track_which_new_evidence_touched_them():
    other = lattice(0, 3, 0, 2, origin=(0, 0, 8))            # a separate structure far from WALL
    prev = [rec("w", "wall", WALL, evidence=("e1",)), rec("o", "wall", other, evidence=("e1",))]
    cand = [rec("w2", "wall", WALL, evidence=("e1",)), rec("o2", "wall", other, evidence=("e1", "e9"))]
    res = recon(prev, cand, new_evidence=("e9",))
    assert len(res["regions"]) == 2
    touched = [r for r in res["regions"] if r["affected_by_new_evidence"]]
    assert len(touched) == 1 and touched[0]["affected_by_new_evidence"] == ["e9"]
    assert {r["status"] for r in res["regions"]} == {"unchanged"}       # both preserved: touched != changed


def test_fit_plane_and_sampling_are_deterministic_and_small():
    fit = sc.fit_plane(WALL)
    assert abs(abs(fit["normal"][2]) - 1.0) < 1e-9 and fit["rms"] < 1e-9
    big = lattice(0, 6, 0, 2, nx=100, ny=100)
    a, b = sc.sample_points(big), sc.sample_points(big)
    assert a == b and len(a) == sc.MAX_POINTS
    assert sc.fit_plane(WALL[:3]) is None                    # too few points -> no invented plane


def test_stacked_overlapping_planes_regrouped_differently_are_the_same_area_not_a_pile_of_changes():
    """Segmentation often produces several overlapping planes for one surface. Two stacked planes becoming
    three overlapping fragments carries the same area: it is a regrouping, not 2 removals and 3 additions."""
    prev = [rec("s1", "wall", WALL), rec("s2", "wall", WALL + np.array([0, 0, 0.02]))]
    cand = [rec("g1", "wall", lattice(0, 2.5, 0, 2)), rec("g2", "wall", lattice(2.0, 4.5, 0, 2)),
            rec("g3", "wall", lattice(4.0, 6, 0, 2))]
    res = recon(prev, cand)
    (r,) = only(res, "regrouped")
    assert len(r["prev"]) == 2 and len(r["cand"]) == 3 and r["cover_prev"] >= 0.6 and r["cover_cand"] >= 0.6
    assert res["counts"]["removed"] == 0 and res["counts"]["new"] == 0 and res["counts"]["ambiguous"] == 0


def test_overlapping_planes_that_only_partly_cover_each_other_stay_ambiguous_not_forced():
    prev = [rec("s1", "wall", WALL), rec("s2", "wall", WALL + np.array([0, 0, 0.02]))]
    cand = [rec("g1", "wall", lattice(0, 1.2, 0, 2)), rec("g2", "wall", lattice(0.8, 2.0, 0, 2))]
    res = recon(prev, cand)
    assert not only(res, "regrouped") and (only(res, "reduced") or only(res, "ambiguous"))


def test_two_parallel_walls_further_apart_than_the_tolerance_are_different_surfaces_even_if_they_overlap_in_projection():
    """A front and a back wall project onto the same footprint. They must not be merged into 'the same area'."""
    back = WALL + np.array([0, 0, 1.5])                    # 0.15 of the extent behind
    res = recon([rec("front", "wall", WALL)], [rec("back", "wall", back)])
    assert len(only(res, "removed")) == 1 and len(only(res, "new")) == 1 and not only(res, "regrouped")
