"""Tests for evidence-backed attribute recognition and observed-vs-inferred scene graph (PR-B)."""
from __future__ import annotations

import numpy as np

from perception.attributes import (
    extract_observed_attributes,
    infer_prior_attributes,
)
from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
)
from perception.reasoning.scene_graph_builder import (
    build_scene_graph_from_observations,
    to_world_ir_relationships,
)
from provenance import Provenance
from world_ir.schema_v1 import RelationshipKind


def _obs(
    eid: str,
    label: str,
    bbox: tuple[float, float, float, float] | None,
    *,
    img_id: str = "img_0.jpg",
    ontology_path: str | None = None,
    mask: np.ndarray | None = None,
) -> ObservationRecord:
    return ObservationRecord(
        evidence_id=eid,
        source_image_id=img_id,
        coordinate_frame=CoordinateFrame(frame_id="image", scale_state="image", units="pixels"),
        model_id=ModelIdentity(
            engine_id="detection",
            architecture_id="grounding_dino",
            checkpoint_id="test",
            license="Apache-2.0",
            hardware_class="cpu",
        ),
        label=label,
        category="object",
        bbox_px=bbox,
        mask=mask,
        confidence=0.85,
        calibration_quality="unverified",
        metadata={"ontology_path": ontology_path or label, "raw_label": label},
    )


def test_attributes_never_fabricate_color_or_depth_without_evidence() -> None:
    o = _obs("o1", "chair", (10.0, 10.0, 50.0, 90.0))
    bundle = extract_observed_attributes(o, image_rgb=None, depth_map=None)
    names = {a.attribute_name for a in bundle.observed}
    assert "2d_orientation" in names
    assert "apparent_pixel_scale" in names
    assert "dominant_color" not in names
    assert "relative_depth_band" not in names
    assert bundle.inferred == ()


def test_attributes_extract_mask_color_depth_and_lexical_cues() -> None:
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 20:50] = True
    rgb[mask] = [180, 40, 40]  # red

    depth = np.full((100, 100), 4.0, dtype=np.float32)
    depth[mask] = 1.5  # foreground relative to 4.0m scene median

    o = _obs("o2", "wooden door", (20.0, 20.0, 50.0, 60.0), mask=mask)
    bundle = extract_observed_attributes(o, image_rgb=rgb, depth_map=depth)
    by_name = {a.attribute_name: a for a in bundle.observed}

    assert by_name["dominant_color"].value == "red"
    assert by_name["dominant_color"].evidence_source == "mask_pixels"
    assert by_name["relative_depth_band"].value == "foreground"
    assert by_name["relative_depth_band"].evidence_source == "depth_mask_crop"
    assert by_name["material_cue"].value == "wood"
    assert all(a.provenance == "observed" for a in bundle.observed)

    priors = infer_prior_attributes(o, {"finish": ("matte", "gloss")})
    assert len(priors) == 1
    assert priors[0].provenance == "inferred"


def test_scene_graph_separates_observed_containment_from_inferred_priors() -> None:
    wall = _obs("w1", "wall", (0.0, 0.0, 400.0, 400.0), ontology_path="architecture/wall")
    switch = _obs("s1", "switch", (100.0, 150.0, 130.0, 190.0), ontology_path="fixtures/electrical/switch")
    isolated_fan = _obs("f1", "ceiling_fan", (50.0, 10.0, 150.0, 60.0), img_id="img_1.jpg")

    depth_map = np.full((400, 400), 2.5, dtype=np.float32)
    report = build_scene_graph_from_observations(
        [wall, switch, isolated_fan],
        depth_maps={"img_0.jpg": depth_map},
    )

    assert report.observed_count == 1
    assert report.inferred_count == 1

    obs_edge = next(e for e in report.edges if e.provenance == "observed")
    assert obs_edge.source_id == "s1"
    assert obs_edge.target_id == "w1"
    assert obs_edge.kind == "attached_to"

    inf_edge = next(e for e in report.edges if e.provenance == "inferred")
    assert inf_edge.source_id == "f1"
    assert inf_edge.target_id == "inferred:img_1.jpg:ceiling"

    rels = to_world_ir_relationships(report.edges)
    assert rels[0].provenance == Provenance.OBSERVED
    assert rels[0].kind == RelationshipKind.ATTACHED_TO
    assert rels[1].provenance == Provenance.INFERRED
