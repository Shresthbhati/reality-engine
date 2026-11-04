"""Tests for Task 7.1: ConfidenceVector populated on every Entity (Blueprint 11, Task 7.1).

Validates:
1. ConfidenceVector dataclass has all 11 required dimensions:
   classification, identity, position, orientation, dimensions, geometry,
   completeness, material, colour, topology, cross_view.
2. Unmeasured dimensions are strictly None (null in JSON) -- never fabricated or defaulted to 0.5.
3. Every Entity defaults to having confidence_vector populated.
4. Serialization (to_dict) and deserialization (from_dict) round-trip accurately.
5. Backward compatibility with older serialized entity dicts lacking confidence_vector.
6. Validation gate catches out-of-bounds confidence values (< 0.0 or > 1.0).
7. WorldDiff tracks confidence_vector modifications deterministically.
8. API endpoint GET /api/worlds/{world_id}/entities/{entity_id}/confidence returns the vector.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from apps.api.main import app
from apps.api.db import get_sessionmaker
from apps.api.models import World, WorldVersion
from world_ir import (
    ConfidenceVector,
    Entity,
    EntityType,
    WorldIR,
    diff_worlds,
)
from world_ir.validation import validate_world_ir


def test_confidence_vector_dimensions_and_defaults():
    cv = ConfidenceVector()
    expected_fields = [
        "classification", "identity", "position", "orientation",
        "dimensions", "geometry", "completeness", "material",
        "colour", "topology", "cross_view",
    ]

    # Every unmeasured dimension must be None by default
    for field in expected_fields:
        assert getattr(cv, field) is None

    d = cv.to_dict()
    assert len(d) == 11
    for k, v in d.items():
        assert v is None


def test_confidence_vector_roundtrip_with_measured_values():
    cv = ConfidenceVector(
        classification=0.95,
        identity=0.88,
        position=0.99,
        orientation=1.0,
        dimensions=0.85,
        geometry=0.91,
        completeness=0.75,
        material=None,  # unmeasured
        colour=None,    # unmeasured
        topology=1.0,
        cross_view=0.82,
    )

    d = cv.to_dict()
    assert d["classification"] == 0.95
    assert d["material"] is None
    assert d["colour"] is None
    assert d["topology"] == 1.0

    restored = ConfidenceVector.from_dict(d)
    assert restored.classification == 0.95
    assert restored.material is None
    assert restored.topology == 1.0
    assert restored.cross_view == 0.82


def test_entity_has_confidence_vector_by_default():
    e = Entity(id="ent-test-01", name="Test Entity", type=EntityType.STRUCTURE)
    assert hasattr(e, "confidence_vector")
    assert isinstance(e.confidence_vector, ConfidenceVector)
    assert e.confidence_vector.classification is None
    assert e.confidence_vector.geometry is None

    # Measure position and geometry
    e.confidence_vector.position = 0.92
    e.confidence_vector.geometry = 0.89

    d = e.to_dict()
    assert "confidence_vector" in d
    assert d["confidence_vector"]["position"] == 0.92
    assert d["confidence_vector"]["classification"] is None

    restored = Entity.from_dict(d)
    assert restored.confidence_vector.position == 0.92
    assert restored.confidence_vector.classification is None


def test_entity_backward_compatibility_with_legacy_dict():
    # Legacy dict from older version without confidence_vector
    legacy_data = {
        "id": "ent-legacy-01",
        "type": "wall",
        "name": "Old Wall",
        "provenance": "OBSERVED",
        "confidence": 0.8,
    }
    e = Entity.from_dict(legacy_data)
    assert isinstance(e.confidence_vector, ConfidenceVector)
    # Must not fabricate values: every dimension is None
    assert e.confidence_vector.classification is None
    assert e.confidence_vector.orientation is None


def test_validation_gate_rejects_out_of_bounds_confidence_vector():
    world = WorldIR(id="wld-val-cv", name="Validation CV Test")
    e = Entity(id="ent-invalid-cv", name="Invalid CV", type=EntityType.WALL)
    e.confidence_vector.classification = 1.25  # Invalid: > 1.0

    world.entities = {e.id: e}
    report = validate_world_ir(world)

    codes = [issue.code for issue in report.issues]
    assert "invalid_confidence_vector_value" in codes
    assert any("confidence_vector.classification (1.25)" in issue.message for issue in report.issues)


def test_diff_worlds_tracks_confidence_vector_changes():
    w1 = WorldIR(id="wld-diff-cv", name="Diff CV Test")
    e1 = Entity(id="ent-d1", name="Tracked Wall", type=EntityType.WALL)
    e1.confidence_vector.completeness = 0.50
    w1.entities = {e1.id: e1}

    w2 = WorldIR(id="wld-diff-cv", name="Diff CV Test")
    e2 = Entity(id="ent-d1", name="Tracked Wall", type=EntityType.WALL)
    e2.confidence_vector.completeness = 0.85  # Updated completeness
    w2.entities = {e2.id: e2}

    diff = diff_worlds(w1, w2)
    assert "ent-d1" in diff.modified_entity_ids
    ent_diff = next(d for d in diff.entity_diffs if d.entity_id == "ent-d1")
    changed_fields = [c.field for c in ent_diff.changes]
    assert "confidence_vector" in changed_fields


@pytest.mark.asyncio
async def test_api_entity_confidence_endpoint():
    from uuid import uuid4
    from world_ir.serialization import save_world

    test_id = f"cv-{uuid4().hex[:8]}"
    world_id = f"wld-{test_id}"
    version_id = f"ver-{test_id}"
    ent_id = f"ent-{test_id}"

    w = WorldIR(id=world_id, name="API CV Test")
    ent = Entity(
        id=ent_id,
        name="Armchair",
        type=EntityType.STRUCTURE,
        confidence=0.88,
    )
    ent.confidence_vector.classification = 0.94
    ent.confidence_vector.position = 0.91
    ent.confidence_vector.geometry = 0.86
    w.entities = {ent.id: ent}

    from apps.api.routes_worlds import _worldstore_root
    from worldstore.store import WorldStore

    store = WorldStore(_worldstore_root())
    store.save_version(w, parent=None, version_id=version_id)

    session_maker = get_sessionmaker()
    async with session_maker() as db:
        w_row = World(id=world_id, name="API CV Test", status="active", current_version_id=version_id)
        v_row = WorldVersion(id=version_id, world_id=world_id)
        db.add(w_row)
        db.add(v_row)
        await db.commit()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/api/worlds/{world_id}/entities/{ent_id}/confidence")
        assert resp.status_code == 200
        data = resp.json()
        assert data["world_id"] == world_id
        assert data["entity_id"] == ent_id
        assert data["scalar_confidence"] == 0.88
        cv = data["confidence_vector"]
        assert cv["classification"] == 0.94
        assert cv["position"] == 0.91
        assert cv["geometry"] == 0.86
        assert cv["material"] is None
        assert cv["colour"] is None
