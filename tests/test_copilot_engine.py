"""Unit, integration, and HTTP contract tests for Grounded Reality Copilot (Epic E11-1).

Verifies:
1. Deterministic natural language query translation (QueryTranslator).
2. Grounded Reality Copilot spatial query answering (RealityCopilotEngine).
3. 100% precision on spatial entity queries with zero fabricated measurements.
4. Clearance and fit reasoning between obstacles.
5. Strict honesty when entities or properties are unrecorded/unknown.
6. API endpoint /api/worlds/{id}/assistant/ask.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from assistant.copilot_engine import CopilotResponse, RealityCopilotEngine
from assistant.query_translator import ParsedQuery, QueryIntent, QueryTranslator
from provenance import Provenance
from world_ir import Entity, EntityType, Relationship, RelationshipKind
from world_ir.schema_v1 import Geometry, GeometryType, Uncertainty, Vector3
from world_ir.world_v1 import WorldIR


# ============================================================================
# 1. Query Translator Unit Tests
# ============================================================================

def test_translator_proximity_query():
    qt = QueryTranslator()
    pq = qt.translate("find all electrical outlets near the kitchen sink")
    assert pq.intent == QueryIntent.PROXIMITY
    assert pq.target_type == "electrical.socket"
    assert pq.reference_name == "kitchen sink"
    assert pq.distance_threshold_m == 2.0


def test_translator_proximity_with_explicit_distance():
    qt = QueryTranslator()
    pq = qt.translate("find sockets within 1.5 meters of bed")
    assert pq.intent == QueryIntent.PROXIMITY
    assert pq.target_type == "electrical.socket"
    assert pq.reference_name == "bed"
    assert pq.distance_threshold_m == 1.5

    pq_cm = qt.translate("find switches within 80cm of door")
    assert pq_cm.intent == QueryIntent.PROXIMITY
    assert pq_cm.distance_threshold_m == 0.8


def test_translator_distance_between():
    qt = QueryTranslator()
    pq = qt.translate("measure distance between chair-1 and table-2")
    assert pq.intent == QueryIntent.MEASUREMENT_DISTANCE
    assert pq.reference_name == "chair-1"
    assert pq.second_reference_name == "table-2"


def test_translator_clearance_fit():
    qt = QueryTranslator()
    pq = qt.translate("can a 2.0m sofa fit between door-1 and window-2?")
    assert pq.intent == QueryIntent.CLEARANCE_FIT
    assert pq.fit_dimension_m == 2.0
    assert pq.target_name == "sofa"
    assert pq.reference_name == "door-1"
    assert pq.second_reference_name == "window-2"


def test_translator_dimension_and_area():
    qt = QueryTranslator()
    pq_height = qt.translate("what is the height of wall-north?")
    assert pq_height.intent == QueryIntent.MEASUREMENT_DIMENSION
    assert pq_height.dimension_property == "height"
    assert pq_height.target_name == "wall-north"

    pq_area = qt.translate("what is the area of bedroom-1?")
    assert pq_area.intent == QueryIntent.MEASUREMENT_AREA
    assert pq_area.dimension_property == "area"
    assert pq_area.target_name == "bedroom-1"


def test_translator_containment_and_attachment():
    qt = QueryTranslator()
    pq_cont = qt.translate("what is inside living room?")
    assert pq_cont.intent == QueryIntent.CONTAINMENT
    assert pq_cont.reference_name == "living room"

    pq_att = qt.translate("what items are mounted on wall-north?")
    assert pq_att.intent == QueryIntent.ATTACHMENT
    assert pq_att.reference_name == "wall-north"


def test_translator_provenance_and_confidence():
    qt = QueryTranslator()
    pq_prov = qt.translate("where did chair-1 come from?")
    assert pq_prov.intent == QueryIntent.PROVENANCE
    assert pq_prov.target_name == "chair-1"

    pq_conf = qt.translate("how confident are we about wall-north?")
    assert pq_conf.intent == QueryIntent.CONFIDENCE
    assert pq_conf.target_name == "wall-north"


def test_translator_structured_dict():
    qt = QueryTranslator()
    pq = qt.translate({
        "intent": "proximity",
        "target": "socket",
        "reference": "sink",
        "radius_m": 3.0,
    })
    assert pq.intent == QueryIntent.PROXIMITY
    assert pq.target_type == "electrical.socket"
    assert pq.reference_name == "sink"
    assert pq.distance_threshold_m == 3.0


# ============================================================================
# 2. Copilot Engine Setup & Spatial Query Execution
# ============================================================================

def _build_test_world() -> WorldIR:
    world = WorldIR(id="w-test")

    # Geometry helper
    def add_geom(gid: str, min_pt: tuple, max_pt: tuple):
        world.geometries[gid] = Geometry(
            id=gid,
            type=GeometryType.BOX,
            bounds_min=Vector3(*min_pt),
            bounds_max=Vector3(*max_pt),
            provenance=Provenance.RECONSTRUCTED,
        )

    # 1. Host Wall North [0.0, 0.0, 0.0] to [6.0, 2.8, 0.2]
    add_geom("g-wall-n", (0.0, 0.0, 0.0), (6.0, 2.8, 0.2))
    world.entities["wall-north"] = Entity(
        id="wall-north",
        type=EntityType.WALL,
        name="Wall North",
        geometry_ids=["g-wall-n"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.92,
        uncertainty=Uncertainty(confidence=0.92, note="Reconstructed from 12 viewpoints"),
    )

    # 2. Door 1 [1.0, 0.0, 0.0] to [2.0, 2.1, 0.2]
    add_geom("g-door-1", (1.0, 0.0, 0.0), (2.0, 2.1, 0.2))
    world.entities["door-1"] = Entity(
        id="door-1",
        type=EntityType.DOOR,
        name="Entry Door",
        geometry_ids=["g-door-1"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.95,
        relationships=[Relationship(kind=RelationshipKind.ATTACHED_TO, target_id="wall-north")],
    )

    # 3. Window 1 [4.0, 1.0, 0.0] to [5.5, 2.0, 0.2] (Clearance between door and window is 4.0 - 2.0 = 2.0m)
    add_geom("g-win-1", (4.0, 1.0, 0.0), (5.5, 2.0, 0.2))
    world.entities["window-1"] = Entity(
        id="window-1",
        type=EntityType.WINDOW,
        name="North Window",
        geometry_ids=["g-win-1"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.88,
        relationships=[Relationship(kind=RelationshipKind.ATTACHED_TO, target_id="wall-north")],
    )

    # 4. Kitchen Sink [1.0, 0.0, 1.0] to [1.8, 0.9, 1.6]
    add_geom("g-sink", (1.0, 0.0, 1.0), (1.8, 0.9, 1.6))
    world.entities["sink-1"] = Entity(
        id="sink-1",
        type=EntityType.STRUCTURE,
        name="Kitchen Sink",
        geometry_ids=["g-sink"],
        provenance=Provenance.OBSERVED,
        confidence=0.96,
        custom_properties={"entity_class": "plumbing.sink"},
    )

    # 5. Socket 1 near sink: [1.2, 1.1, 1.0] to [1.3, 1.2, 1.05] (dy = 0.2m, dx = 0.0, dz = 0.0 => distance = 0.2m)
    add_geom("g-sock-1", (1.2, 1.1, 1.0), (1.3, 1.2, 1.05))
    world.entities["socket-1"] = Entity(
        id="socket-1",
        type=EntityType.STRUCTURE,
        name="Sink Socket",
        geometry_ids=["g-sock-1"],
        provenance=Provenance.OBSERVED,
        confidence=0.94,
        custom_properties={"entity_class": "electrical.socket"},
    )

    # 6. Socket 2 far from sink: [5.0, 1.1, 1.0] to [5.1, 1.2, 1.05] (distance to sink > 3.0m)
    add_geom("g-sock-2", (5.0, 1.1, 1.0), (5.1, 1.2, 1.05))
    world.entities["socket-2"] = Entity(
        id="socket-2",
        type=EntityType.STRUCTURE,
        name="Living Socket",
        geometry_ids=["g-sock-2"],
        provenance=Provenance.OBSERVED,
        confidence=0.90,
        custom_properties={"entity_class": "electrical.socket"},
    )

    # 7. Chair 1 [2.0, 0.0, 2.0] to [2.5, 0.9, 2.5]
    add_geom("g-chair-1", (2.0, 0.0, 2.0), (2.5, 0.9, 2.5))
    world.entities["chair-1"] = Entity(
        id="chair-1",
        type=EntityType.STRUCTURE,
        name="Office Chair",
        geometry_ids=["g-chair-1"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.85,
        custom_properties={"entity_class": "furniture.chair"},
    )

    # 8. Desk 1 [3.5, 0.0, 2.0] to [5.0, 0.75, 3.0]
    add_geom("g-desk-1", (3.5, 0.0, 2.0), (5.0, 0.75, 3.0))
    world.entities["desk-1"] = Entity(
        id="desk-1",
        type=EntityType.STRUCTURE,
        name="Work Desk",
        geometry_ids=["g-desk-1"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.87,
        custom_properties={"entity_class": "furniture.table"},
    )

    # 9. Entity with no geometry bounds (unrecorded dimensions)
    world.entities["mystery-lamp"] = Entity(
        id="mystery-lamp",
        type=EntityType.STRUCTURE,
        name="Mystery Lamp",
        geometry_ids=[],
        provenance=Provenance.UNKNOWN,
    )

    return world


def test_copilot_spatial_proximity():
    world = _build_test_world()
    engine = RealityCopilotEngine(world, scale_state="metric")

    res = engine.ask("find all electrical outlets near the kitchen sink")
    assert res.success is True
    assert len(res.entities) == 1
    assert res.entities[0]["id"] == "socket-1"
    assert res.entities[0]["distance_m"] < 1.0
    assert len(res.measurements) == 1
    assert res.measurements[0].value < 1.0
    assert res.measurements[0].unit == "m"


def test_copilot_proximity_excludes_far_entities():
    world = _build_test_world()
    engine = RealityCopilotEngine(world, scale_state="metric")

    # socket-2 is > 3m away from sink; asking with default 2m radius must exclude it
    res = engine.ask("find sockets within 1.0m of sink-1")
    assert res.success is True
    ids = [e["id"] for e in res.entities]
    assert "socket-1" in ids
    assert "socket-2" not in ids


def test_copilot_distance_measurement():
    world = _build_test_world()
    engine = RealityCopilotEngine(world, scale_state="metric")

    # Chair is [2.0, 0.0, 2.0] to [2.5, 0.9, 2.5]
    # Desk is [3.5, 0.0, 2.0] to [5.0, 0.75, 3.0]
    # Surface distance in X is 3.5 - 2.5 = 1.0m; Y and Z overlap (distance = 1.0m)
    res = engine.ask("measure distance between chair-1 and desk-1")
    assert res.success is True
    assert len(res.measurements) == 1
    assert pytest.approx(res.measurements[0].value, abs=1e-3) == 1.0
    assert res.measurements[0].unit == "m"
    assert "1.00 m" in res.answer


def test_copilot_clearance_fit_reasoning():
    world = _build_test_world()
    engine = RealityCopilotEngine(world, scale_state="metric")

    # Clearance between door-1 (max x=2.0) and window-1 (min x=4.0) is 2.0m
    # 1. A 1.6m desk should fit with +0.40m margin
    res_fits = engine.ask("can a 1.6m desk fit between door-1 and window-1?")
    assert res_fits.success is True
    assert "Yes" in res_fits.answer
    assert "margin: +0.40m" in res_fits.answer

    # 2. A 2.5m sofa should NOT fit with -0.50m shortfall
    res_no = engine.ask("can a 2.5m sofa fit between door-1 and window-1?")
    assert res_no.success is True
    assert "No" in res_no.answer
    assert "shortfall: -0.50m" in res_no.answer


def test_copilot_dimension_measurement():
    world = _build_test_world()
    engine = RealityCopilotEngine(world, scale_state="metric")

    # Wall height is 2.8m
    res = engine.ask("what is the height of wall-north?")
    assert res.success is True
    assert len(res.measurements) == 1
    assert pytest.approx(res.measurements[0].value, abs=1e-3) == 2.8
    assert "2.80 m" in res.answer


def test_copilot_attachment_query():
    world = _build_test_world()
    engine = RealityCopilotEngine(world, scale_state="metric")

    res = engine.ask("what items are mounted on wall-north?")
    assert res.success is True
    ids = {e["id"] for e in res.entities}
    assert "door-1" in ids
    assert "window-1" in ids


def test_copilot_zero_hallucination_on_missing_entity():
    world = _build_test_world()
    engine = RealityCopilotEngine(world, scale_state="metric")

    # Inquiring about non-existent entity
    res = engine.ask("find all sockets near the bathtub")
    assert res.success is False
    assert "bathtub" in res.answer
    assert "not found" in res.answer
    assert len(res.entities) == 0
    assert len(res.grounded_facts) == 0


def test_copilot_unrecorded_dimension_honesty():
    world = _build_test_world()
    engine = RealityCopilotEngine(world, scale_state="metric")

    # Mystery lamp has no geometry bounds
    res = engine.ask("what is the height of mystery-lamp?")
    assert res.success is False
    assert "Not recorded" in res.answer
    assert len(res.measurements) == 0


def test_copilot_provenance_and_confidence():
    world = _build_test_world()
    engine = RealityCopilotEngine(world, scale_state="metric")

    res = engine.ask("where did wall-north come from?")
    assert res.success is True
    assert "RECONSTRUCTED" in res.answer
    assert "Confidence: 0.92" in res.answer


# ============================================================================
# 3. HTTP API Contract Test: /api/worlds/{id}/assistant/ask
# ============================================================================

@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def test_api_assistant_ask_endpoint(client):
    # 1. Create world
    resp = client.post("/api/worlds", json={"name": "Copilot World"})
    assert resp.status_code == 201
    wid = resp.json()["id"]

    # 2. Commit test world
    from apps.api import worldstore_service as svc
    import apps.api.db as db_mod

    async def commit():
        async with db_mod.get_sessionmaker()() as db:
            await svc.commit_version(db, world_id=wid, world=_build_test_world(), parent=None)

    asyncio.run(commit())

    # 3. Test natural language query via HTTP
    ask_resp = client.post(
        f"/api/worlds/{wid}/assistant/ask",
        json={"query": "find all electrical outlets near the kitchen sink"},
    )
    assert ask_resp.status_code == 200
    data = ask_resp.json()
    assert data["success"] is True
    assert data["intent"] == "proximity"
    assert len(data["entities"]) == 1
    assert data["entities"][0]["id"] == "socket-1"
    assert len(data["measurements"]) == 1
    assert "socket-1" in data["answer"]

    # 4. Test clearance fit query via HTTP
    fit_resp = client.post(
        f"/api/worlds/{wid}/assistant/ask",
        json={"query": "can a 1.5m desk fit between door-1 and window-1?"},
    )
    assert fit_resp.status_code == 200
    fit_data = fit_resp.json()
    assert fit_data["success"] is True
    assert "Yes" in fit_data["answer"]

    # 5. Test missing entity rejection via HTTP
    missing_resp = client.post(
        f"/api/worlds/{wid}/assistant/ask",
        json={"query": "find items near nonexistent_thing"},
    )
    assert missing_resp.status_code == 200
    missing_data = missing_resp.json()
    assert missing_data["success"] is False
    assert "not found" in missing_data["answer"]
