from types import SimpleNamespace as NS

from apps.api.jobs import _drop_exact_duplicates


def _ev(i, checksum):
    return NS(id=i, checksum=checksum)


def test_identical_checksums_keep_first_and_report_rest():
    rows = [(_ev("a", "h1"), "pa"), (_ev("b", "h2"), "pb"), (_ev("c", "h1"), "pc")]
    kept, dropped = _drop_exact_duplicates(rows)
    assert [e.id for e, _ in kept] == ["a", "b"]
    assert dropped == [{"evidence_id": "c", "reason": "byte-identical duplicate of a"}]


def test_missing_checksum_is_never_treated_as_duplicate():
    rows = [(_ev("a", None), "pa"), (_ev("b", None), "pb")]
    kept, dropped = _drop_exact_duplicates(rows)
    assert len(kept) == 2 and not dropped


def test_capture_quality_flags_the_measured_failure_and_passes_a_good_set():
    from reconstruction.scene_identity import capture_quality

    bad = capture_quality(10, ["a", "b", "c"], 92, n_duplicates_dropped=10)  # the measured living-room set
    assert bad["state"] == "insufficient" and bad["registered"] == 3 and bad["advice"]
    assert "3 of 10" in bad["message"] and bad["duplicates_dropped"] == 10
    good = capture_quality(10, [str(i) for i in range(10)], 5000)
    assert good["state"] == "sufficient" and not good["advice"]
    mid = capture_quality(10, [str(i) for i in range(6)], 800)
    assert mid["state"] == "partial"


def test_placed_cameras_without_a_closed_room_is_only_partial():
    from reconstruction.scene_identity import capture_quality

    ids = [str(i) for i in range(9)]
    q = capture_quality(10, ids, 770, structure={"wall": 3, "ceiling": 1})   # the measured living-room job
    assert q["state"] == "partial" and q["room_found"] is False
    assert "3 wall" in q["message"] and "floor" in " ".join(q["reasons"]) and q["advice"]
    room = capture_quality(10, ids, 5000, structure={"floor": 1, "ceiling": 1, "wall": 4})
    assert room["state"] == "sufficient" and room["room_found"] is True
    assert capture_quality(10, ids, 5000, structure={"room": 1})["room_found"] is True
    assert capture_quality(10, ids, 5000)["state"] == "sufficient"   # callers that do not know the structure are unchanged


# the real attempt text from a job whose COLMAP workspace path was 262 characters
_PATH_FAILED = [
    {"level": 2, "name": "sparse photogrammetric reconstruction", "outcome": "failed", "detail": (
        "VerticalSliceError: reconstruction stage failed: all candidate backends failed or declined: "
        "ColmapReconstructionBackend: WorkspacePathTooLong: COLMAP workspace path is 262 characters (limit 170); "
        "native COLMAP on Windows crashes silently beyond ~260 characters including file names. "
        "Point WORLDSTORE_ROOT at a shorter directory.")},
    {"level": 0, "name": "single-image visual hypothesis", "outcome": "succeeded", "detail": "10 independent single-view hypotheses fused (unregistered)"},
]


def test_a_too_long_workspace_path_is_reported_as_setup_not_as_bad_photos():
    from reconstruction.scene_identity import capture_quality

    q = capture_quality(10, [], 43740, attempts=_PATH_FAILED)
    assert q["state"] == "insufficient" and q["cause"] == "workspace_path_too_long"
    assert "262" in q["message"] and "170" in q["message"] and "not a problem with the photos" in q["message"]
    assert any("WORLDSTORE_ROOT" in a for a in q["advice"])
    assert not any("overlap" in a or "lighting" in a for a in q["advice"])   # no photo advice for a setup failure
    assert any("262 characters" in r for r in q["reasons"])


def test_other_failures_and_healthy_runs_keep_the_photo_verdict():
    from reconstruction.scene_identity import capture_quality

    other = [{"level": 2, "outcome": "failed", "detail": "ColmapReconstructionBackend: mapper produced no model"}]
    q = capture_quality(10, ["a", "b", "c"], 92, attempts=other)
    assert q["state"] == "insufficient" and "cause" not in q and any("overlap" in a for a in q["advice"])
    # a stale path failure must not override a run that did register the photos
    ok = capture_quality(10, [str(i) for i in range(10)], 5000, attempts=_PATH_FAILED)
    assert ok["state"] == "sufficient" and "cause" not in ok
    assert "cause" not in capture_quality(10, ["a", "b", "c"], 92)   # callers that pass no attempts are unchanged
