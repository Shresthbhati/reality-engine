"""Unit tests for Blueprint Task 8.3: Parametric object templates.

Tests parametric fitting for 7 furniture classes:
table, cabinet, sofa, chair, bed, tv, media_console.
Verifies component breakdown, bounding box fitting, and residual calculations.
"""

import pytest
import numpy as np

from perception.architecture.parametric import (
    ParametricComponent,
    ParametricObjectFit,
    fit_table_template,
    fit_cabinet_template,
    fit_sofa_template,
    fit_chair_template,
    fit_bed_template,
    fit_tv_template,
    fit_media_console_template,
    fit_parametric_object,
)


def _generate_box_points(center, dims, num_points=100):
    """Generate points roughly within a 3D box."""
    c = np.array(center, dtype=np.float64)
    d = np.array(dims, dtype=np.float64)
    mins = c - d / 2.0
    maxs = c + d / 2.0
    pts = np.random.uniform(mins, maxs, size=(num_points, 3))
    return pts


def test_fit_table_template():
    pts = _generate_box_points([1.0, 2.0, 0.4], [1.2, 0.8, 0.75])
    fit = fit_table_template(pts)
    assert fit.template_name == "table"
    assert "table" in fit.entity_class
    assert fit.confidence > 0.0
    assert fit.dimensions[0] > 0 and fit.dimensions[1] > 0 and fit.dimensions[2] > 0
    # Should contain top and 4 legs
    comp_names = [c.name for c in fit.components]
    assert "top" in comp_names
    assert "leg_fl" in comp_names
    assert "leg_fr" in comp_names
    assert "leg_bl" in comp_names
    assert "leg_br" in comp_names
    assert len(fit.components) >= 5


def test_fit_cabinet_template():
    pts = _generate_box_points([0.0, 1.0, 0.9], [0.9, 0.5, 1.8])
    fit = fit_cabinet_template(pts)
    assert fit.template_name == "cabinet"
    assert "cabinet" in fit.entity_class
    comp_names = [c.name for c in fit.components]
    assert "body" in comp_names
    assert "door_left" in comp_names
    assert "door_right" in comp_names
    assert "shelf" in comp_names
    assert len(fit.components) >= 4


def test_fit_sofa_template():
    pts = _generate_box_points([0.0, 0.0, 0.45], [2.1, 0.9, 0.85])
    fit = fit_sofa_template(pts)
    assert fit.template_name == "sofa"
    assert "sofa" in fit.entity_class
    comp_names = [c.name for c in fit.components]
    assert "frame" in comp_names
    assert "armrest_left" in comp_names
    assert "armrest_right" in comp_names
    assert "seat_cushion" in comp_names
    assert len(fit.components) >= 5


def test_fit_chair_template():
    pts = _generate_box_points([0.5, 0.5, 0.45], [0.55, 0.55, 0.9])
    fit = fit_chair_template(pts)
    assert fit.template_name == "chair"
    assert "chair" in fit.entity_class
    comp_names = [c.name for c in fit.components]
    assert "seat" in comp_names
    assert "backrest" in comp_names
    assert "leg_fl" in comp_names
    assert len(fit.components) >= 6


def test_fit_bed_template():
    pts = _generate_box_points([0.0, 0.0, 0.4], [1.6, 2.0, 0.9])
    fit = fit_bed_template(pts)
    assert fit.template_name == "bed"
    assert "bed" in fit.entity_class
    comp_names = [c.name for c in fit.components]
    assert "frame" in comp_names
    assert "mattress" in comp_names
    assert "headboard" in comp_names
    assert len(fit.components) >= 3


def test_fit_tv_template():
    pts = _generate_box_points([0.0, 0.0, 1.2], [1.2, 0.1, 0.7])
    fit = fit_tv_template(pts)
    assert fit.template_name == "tv"
    assert "tv" in fit.entity_class
    comp_names = [c.name for c in fit.components]
    assert "screen" in comp_names
    assert "frame" in comp_names
    assert "stand" in comp_names
    assert len(fit.components) >= 3


def test_fit_media_console_template():
    pts = _generate_box_points([0.0, 0.0, 0.3], [1.8, 0.45, 0.55])
    fit = fit_media_console_template(pts)
    assert fit.template_name == "media_console"
    assert "media_console" in fit.entity_class
    comp_names = [c.name for c in fit.components]
    assert "body" in comp_names
    assert "shelf" in comp_names
    assert "door_left" in comp_names
    assert "door_right" in comp_names
    assert len(fit.components) >= 4


def test_fit_parametric_object_dispatch():
    pts = _generate_box_points([1.0, 1.0, 0.5], [1.0, 1.0, 1.0])
    
    # Test dispatch by ontology class
    classes = [
        ("furniture/table", "table"),
        ("furniture/cabinet", "cabinet"),
        ("furniture/sofa", "sofa"),
        ("furniture/chair", "chair"),
        ("furniture/bed", "bed"),
        ("electronics/tv", "tv"),
        ("furniture/media_console", "media_console"),
    ]
    for cls_name, expected_tmpl in classes:
        fit = fit_parametric_object(cls_name, pts)
        assert fit.template_name == expected_tmpl
        assert len(fit.components) > 0

    # Test unknown / fallback
    fit_unknown = fit_parametric_object("decor/vase", pts)
    assert fit_unknown.template_name == "generic_box"
    assert len(fit_unknown.components) == 1
    assert fit_unknown.components[0].name == "bounding_box"


def test_parametric_fit_residuals_and_serialization():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    fit = fit_table_template(pts)
    assert fit.residual_rms >= 0.0
    
    d = fit.to_dict()
    assert d["template_name"] == "table"
    assert "center" in d
    assert "dimensions" in d
    assert len(d["components"]) >= 5
    assert "residual_rms" in d
