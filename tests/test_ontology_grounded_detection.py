"""Ontology-grounded production detection: bounded small-object queries, honest
label→ontology grounding, and tiled small-object inference.

These tests cover the layer added in ``perception/detection/ontology_grounding.py``
plus its wiring into ``perception/detection_adapter.py``. The unit tests are
deterministic and need no model weights. The adapter tests substitute a stub
detector so they test *our* mapping/tiling logic, not Grounding DINO's weights;
the final test runs the real cached model and skips honestly when it cannot load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BENCH_IMAGE = REPO / ".benchmarks" / "matcher_benchmark_out" / "pair_00_a.png"

#: A few prompts that only exist because an ontology small class is queried.
SMALL_PROMPTS = {"wall socket", "power outlet", "plug point", "door handle",
                 "air vent", "door hinge", "light switch", "radiator"}


def _small_hits(prompts: list[str]) -> set[str]:
    return {p for p in prompts if p in SMALL_PROMPTS}


class TestProductionQueries:
    def test_legacy_survey_list_has_no_small_classes(self) -> None:
        """Documents the defect this change removes: the old hard-coded list
        contained zero small classes, so they were never queried at all."""
        from perception.detection_adapter import DEFAULT_SURVEY_QUERIES

        assert _small_hits(list(DEFAULT_SURVEY_QUERIES)) == set()

    def test_ontology_queries_do_include_small_classes(self) -> None:
        from perception.detection.ontology_grounding import production_queries

        q = production_queries(budget=40)
        assert len(q) <= 40
        assert len(q) == len(set(q)), "prompts must be deduplicated"
        assert all(p == p.lower().strip() for p in q)
        assert _small_hits(q), f"no small-class prompt made the budget: {q}"

    def test_budget_is_respected_at_small_sizes(self) -> None:
        from perception.detection.ontology_grounding import production_queries

        for budget in (1, 5, 10):
            q = production_queries(budget=budget)
            assert 0 < len(q) <= budget

    def test_domain_filter_excludes_outdoor_from_indoor(self) -> None:
        from perception.detection.ontology_grounding import production_queries

        indoor = production_queries(domain="indoor", budget=60)
        assert not any("kerb" in p or "curb" in p or "footpath" in p for p in indoor)

    def test_extra_prompts_are_prepended_and_kept(self) -> None:
        from perception.detection.ontology_grounding import production_queries

        q = production_queries(budget=20, extra=("my custom widget",))
        assert q[0] == "my custom widget"


class TestGroundLabel:
    def test_small_class_resolves_with_provenance(self) -> None:
        from perception.detection.ontology_grounding import ground_label
        from perception.ontology import get_default_ontology

        g = ground_label("power outlet")
        assert g.resolved is True
        assert g.is_small is True
        assert g.category == "object"
        # The canonical (B03-referenced) socket node is ``fixtures/electrical/socket``;
        # a second, unmarked ``electrical.switchboard.module.socket`` also exists.
        assert g.ontology_path.endswith("socket")
        assert g.group in {"fixtures", "electrical"}
        # every emitted path must be resolvable by the canonical ontology
        assert get_default_ontology().get(g.ontology_path) is not None

    def test_architecture_class_maps_to_architecture_category(self) -> None:
        from perception.detection.ontology_grounding import ground_label

        g = ground_label("wall")
        assert g.resolved is True
        assert g.category == "architecture"
        assert g.ontology_path == "architecture/wall"

    def test_unknown_label_is_honest_fallback(self) -> None:
        from perception.detection.ontology_grounding import ground_label
        from perception.ontology import FALLBACK_PATH

        raw = "zzqxvblorf"
        g = ground_label(raw)
        assert g.resolved is False
        assert g.ontology_path == FALLBACK_PATH
        assert g.taxonomy_source == "fallback"
        # the raw detector label is preserved, never replaced by an invented class
        assert g.raw_label == raw
        assert g.canonical_label == raw

    def test_metadata_is_json_safe(self) -> None:
        import json

        from perception.detection.ontology_grounding import ground_label

        meta = ground_label("door handle").to_metadata()
        assert json.loads(json.dumps(meta)) == meta
        assert meta["label_resolved"] is True
        assert meta["is_small"] is True


class TestTiling:
    def test_windows_cover_the_whole_image(self) -> None:
        from perception.detection.ontology_grounding import tile_windows

        w, h = 2000, 1500
        windows = tile_windows(w, h, scale=2, overlap=0.2)
        assert len(windows) == 4
        assert min(x0 for x0, _, _, _ in windows) == 0
        assert max(x1 for _, _, x1, _ in windows) == w
        assert min(y0 for _, y0, _, _ in windows) == 0
        assert max(y1 for _, _, _, y1 in windows) == h
        for x0, y0, x1, y1 in windows:
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h

    def test_adjacent_windows_overlap(self) -> None:
        from perception.detection.ontology_grounding import tile_windows

        windows = tile_windows(1000, 1000, scale=3, overlap=0.25)
        xs = sorted({x0 for x0, _, _, _ in windows})
        tw = windows[0][2] - windows[0][0]
        for left, right in zip(xs, xs[1:]):
            assert right < left + tw, "tiles must overlap, not tile edge-to-edge"

    def test_scale_one_is_the_whole_image(self) -> None:
        from perception.detection.ontology_grounding import tile_windows

        assert tile_windows(640, 480, scale=1) == [(0, 0, 640, 480)]


class TestNms:
    def test_overlapping_duplicates_are_suppressed_keeping_best_score(self) -> None:
        from perception.detection.ontology_grounding import nms

        boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60)]
        kept = nms(boxes, [0.9, 0.8, 0.7])
        assert kept == [0, 2]

    def test_disjoint_boxes_all_survive(self) -> None:
        from perception.detection.ontology_grounding import nms

        boxes = [(0, 0, 5, 5), (100, 100, 105, 105)]
        assert nms(boxes, [0.5, 0.4]) == [0, 1]

    def test_degenerate_boxes_do_not_crash(self) -> None:
        from perception.detection.ontology_grounding import nms

        assert nms([(0, 0, 0, 0), (1, 1, 1, 1)], [0.1, 0.2]) == [1, 0]


class TestDetectionAdapterGrounding:
    """Adapter behaviour with a deterministic stub detector (no weights needed)."""

    @pytest.fixture()
    def _stub(self, monkeypatch):
        import perception.detection_adapter as da

        monkeypatch.setattr(da, "_load_detector", lambda *a, **k: (object(), object(), "cpu"))
        monkeypatch.setattr(da, "_get_vram_peak", lambda: None)

        detections = [
            ((10.0, 10.0, 40.0, 40.0), 0.91, "power outlet"),
            ((0.0, 0.0, 300.0, 200.0), 0.77, "wall"),
            ((50.0, 50.0, 60.0, 60.0), 0.55, "zzqxvblorf"),
        ]
        monkeypatch.setattr(da, "_detect_in_image", lambda *a, **k: list(detections))
        return da

    def test_labels_are_grounded_with_provenance(self, _stub) -> None:
        if not BENCH_IMAGE.is_file():
            pytest.skip(f"benchmark image missing: {BENCH_IMAGE}")
        out = _stub._grounding_dino_run([str(BENCH_IMAGE)], run_id="t", threshold=0.3)
        assert out.status in ("succeeded", "partial")
        assert out.n_observations == 3

        by_label = {o.label: o for o in out.observations}
        socket = by_label["power outlet"]
        assert socket.metadata["ontology_path"].endswith("socket")
        assert socket.metadata["is_small"] is True
        assert socket.metadata["label_resolved"] is True

        wall = by_label["wall"]
        assert wall.category == "architecture"
        assert wall.metadata["ontology_path"] == "architecture/wall"

        unknown = by_label["zzqxvblorf"]
        assert unknown.metadata["label_resolved"] is False
        assert unknown.metadata["ontology_path"].startswith("fallback")

        # confidence is the model's own score, never rescaled
        assert socket.confidence == 0.91
        assert socket.metadata["raw_score"] == 0.91
        assert socket.metadata["is_calibrated"] is False

        assert out.summary["n_small_detections"] == 1
        assert out.summary["n_unresolved_labels"] == 1
        assert out.summary["query_source"] == "ontology"

    def test_caller_queries_override_ontology_default(self, _stub) -> None:
        if not BENCH_IMAGE.is_file():
            pytest.skip(f"benchmark image missing: {BENCH_IMAGE}")
        out = _stub._grounding_dino_run(
            [str(BENCH_IMAGE)], run_id="t", queries=["chair"]
        )
        assert out.summary["query_source"] == "caller"
        assert out.summary["n_queries"] == 1

    def test_tiled_path_maps_boxes_and_calls_nms(self, monkeypatch) -> None:
        import perception.detection_adapter as da

        calls: list[int] = []
        monkeypatch.setattr(
            da, "_detect_in_image",
            lambda *a, **k: [((1.0, 1.0, 5.0, 5.0), 0.6, "wall")],
        )

        def _fake_nms(boxes, scores, iou_threshold=0.5):
            calls.append(len(boxes))
            return [0]

        monkeypatch.setattr(da, "nms", _fake_nms)

        from PIL import Image

        image = Image.new("RGB", (400, 300), (128, 128, 128))
        dets, n_tiles = da._detect_tiled(
            object(), object(), image, ["wall"], 0.3, "cpu", 2, 0.2
        )
        assert n_tiles == 4
        assert calls and calls[0] == 4, "nms must see every tile's detections"
        assert len(dets) == 1
        bx0, by0, bx1, by1 = dets[0][0]
        assert 0 <= bx0 <= 400 and 0 <= by0 <= 300


@pytest.mark.skipif(not BENCH_IMAGE.is_file(), reason="benchmark image missing")
def test_real_grounding_dino_emits_grounded_observations() -> None:
    """Real model, real image. Skips (never fails) when the weights cannot load."""
    from perception.detection_adapter import _grounding_dino_available, _grounding_dino_run

    reason = _grounding_dino_available()
    if reason is not None:
        pytest.skip(f"Grounding DINO unavailable: {reason}")

    out = _grounding_dino_run(
        [str(BENCH_IMAGE)], run_id="real_grounding", threshold=0.30,
        query_budget=20, domain="indoor",
    )
    if out.status == "failed" and "model load failed" in (out.error_message or ""):
        pytest.skip(f"model weights not loadable here: {out.error_message}")

    assert out.status in ("succeeded", "partial")
    assert out.summary["query_source"] == "ontology"
    for o in out.observations:
        assert "ontology_path" in o.metadata
        assert o.metadata["is_calibrated"] is False
        assert o.confidence == o.metadata["raw_score"]
