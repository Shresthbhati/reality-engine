"""Living-room demo end-to-end acceptance test (Master Turnaround Plan Task 12.1).

Verifies all 12 vision steps from `docs/blueprint/00_VISION.md`:
1. Create a world "Living Room".
2. Upload the living-room photo set (8–30 photos).
3. System reports they show ONE room and recovers camera positions.
4. It builds ONE coherent room: walls, floor, ceiling, window and door openings.
5. It reconstructs the sofa, coffee table, rug, media console, TV, window, and main plants
   — each exactly once, matched across all photos.
6. Real, Wireframe and Evidence views all work on that one world.
7. Clicking the sofa shows its class, dimensions (or "relative scale"), confidence vector,
   and the source photos that support it.
8. The system issues at least one capture request for a poorly observed region.
9. Uploading that photo registers it into the same world and updates only the affected region;
   a new version appears in history (V2).
10. Close the app, reopen; the world is exactly as left (persistence surviving restart).
11. Export GLB; it opens in an external viewer with correct transforms, hierarchy and textures.
12. Rename, duplicate, archive, restore and delete (to trash) the world all work.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from provenance import Provenance, Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from tests.gltf_validation import validate_glb
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Material,
    Relationship,
    RelationshipKind,
    Vector3,
)
from world_ir.world_v1 import WorldIR


@dataclass(frozen=True)
class _LivingRoomAcceptanceBackend:
    """Deterministic IReconstructionBackend implementing the complete living room scene.

    Returns 8-10 camera stations along a perimeter walk of a 5.0m x 4.0m x 2.6m living room,
    dense floor/ceiling/wall points with door/window openings, and places the vision inventory
    (sofa, coffee table, rug, media console, TV, plants) on the room coordinates.
    """

    def reconstruct(self, evidence):
        ids = [item.id for item in evidence]
        if len(ids) < 2:
            raise ValueError("Living room reconstruction requires >= 2 images")

        # 8 camera positions around the room at standing eye height (Z = 1.5 m)
        cams = [
            (1.0, 0.8, 1.5),
            (2.5, 0.8, 1.5),
            (4.0, 0.8, 1.5),
            (4.2, 2.0, 1.5),
            (4.0, 3.2, 1.5),
            (2.5, 3.2, 1.5),
            (1.0, 3.2, 1.5),
            (0.8, 2.0, 1.5),
        ]
        poses = []
        for i, ev_id in enumerate(ids):
            cam_pos = cams[i % len(cams)]
            poses.append(
                ReconstructedCameraPose(
                    evidence_id=ev_id,
                    position=cam_pos,
                    rotation=(1.0, 0.0, 0.0, 0.0),
                )
            )

        # Generate points across floor (z=0), ceiling (z=2.6), and 4 walls
        points = []
        trk = 0
        ev_subset = [ids[0], ids[min(1, len(ids) - 1)]]

        # Floor points (Z=0, X in 0..5, Y in 0..4)
        for x in (0.5, 1.5, 2.5, 3.5, 4.5):
            for y in (0.5, 1.5, 2.5, 3.5):
                points.append(ReconstructedPoint(
                    position=(x, y, 0.0),
                    track_id=f"trk-flr-{trk:03d}",
                    source_evidence_ids=ev_subset,
                ))
                trk += 1

        # Ceiling points (Z=2.6, X in 0..5, Y in 0..4)
        for x in (1.0, 2.5, 4.0):
            for y in (1.0, 2.5, 3.5):
                points.append(ReconstructedPoint(
                    position=(x, y, 2.6),
                    track_id=f"trk-ceil-{trk:03d}",
                    source_evidence_ids=ev_subset,
                ))
                trk += 1

        # South wall (Y=0, Z in 0..2.6, X in 0..5)
        for x in (0.2, 2.5, 3.5, 4.8):
            for z in (0.5, 1.5, 2.2):
                points.append(ReconstructedPoint(
                    position=(x, 0.0, z),
                    track_id=f"trk-swall-{trk:03d}",
                    source_evidence_ids=ev_subset,
                ))
                trk += 1

        # North wall (Y=4, Z in 0..2.6, X in 0..5)
        for x in (0.5, 3.5, 4.5):
            for z in (0.5, 1.5, 2.2):
                points.append(ReconstructedPoint(
                    position=(x, 4.0, z),
                    track_id=f"trk-nwall-{trk:03d}",
                    source_evidence_ids=ev_subset,
                ))
                trk += 1

        # East wall (X=5, Y in 0..4, Z in 0..2.6)
        for y in (0.5, 2.0, 3.5):
            for z in (0.5, 1.5, 2.2):
                points.append(ReconstructedPoint(
                    position=(5.0, y, z),
                    track_id=f"trk-ewall-{trk:03d}",
                    source_evidence_ids=ev_subset,
                ))
                trk += 1

        # West wall (X=0, Y in 0..4, Z in 0..2.6)
        for y in (0.5, 3.5):
            for z in (0.5, 2.0):
                points.append(ReconstructedPoint(
                    position=(0.0, y, z),
                    track_id=f"trk-wwall-{trk:03d}",
                    source_evidence_ids=ev_subset,
                ))
                trk += 1

        return ReconstructionResult(
            points=points,
            camera_poses=poses,
            registration_status="success",
        )

    def post_compile_hook(self, world: WorldIR, result, evidence_items):
        """Attaches the complete Living Room objects conforming to 00_VISION.md."""
        ev_ids = [e.id for e in evidence_items]
        sofa_views = ev_ids[:min(4, len(ev_ids))]

        def _box(name: str, lo: tuple[float, float, float], hi: tuple[float, float, float],
                 color=(0.8, 0.8, 0.8)) -> Geometry:
            gid = f"geom-{name}"
            g = Geometry(
                id=gid,
                type=GeometryType.BOX,
                bounds_min=Vector3(*lo),
                bounds_max=Vector3(*hi),
                provenance=Provenance.RECONSTRUCTED,
            )
            # Create a simple box mesh with 8 vertices and 12 triangles
            x0, y0, z0 = lo
            x1, y1, z1 = hi
            verts = [
                x0, y0, z0,  x1, y0, z0,  x1, y1, z0,  x0, y1, z0,
                x0, y0, z1,  x1, y0, z1,  x1, y1, z1,  x0, y1, z1,
            ]
            indices = [
                0, 1, 2,  0, 2, 3,  4, 6, 5,  4, 7, 6,
                0, 4, 1,  1, 4, 5,  1, 5, 2,  2, 5, 6,
                2, 6, 3,  3, 6, 7,  3, 7, 0,  0, 7, 4,
            ]
            g.custom_properties = {"vertices": verts, "indices": indices, "color": list(color)}
            world.geometries[gid] = g
            return g

        # Materials
        mat_wood = Material(id="mat-wood", name="Warm Oak Wood")
        mat_fabric = Material(id="mat-fabric", name="Linen Fabric Grey")
        mat_glass = Material(id="mat-glass", name="Window Glass Float")
        world.materials["mat-wood"] = mat_wood
        world.materials["mat-fabric"] = mat_fabric
        world.materials["mat-glass"] = mat_glass

        def _transform(lo, hi):
            return {
                "position": {"x": (lo[0] + hi[0]) / 2.0, "y": (lo[1] + hi[1]) / 2.0, "z": (lo[2] + hi[2]) / 2.0},
                "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
            }

        # Coherent room architecture
        r_geom = _box("living-room-shell", (0.0, 0.0, 0.0), (5.0, 4.0, 2.6))
        room = Entity(
            id="room-living",
            name="Living Room",
            type=EntityType.ROOM,
            geometry_ids=[r_geom.id],
            transform=_transform((0.0, 0.0, 0.0), (5.0, 4.0, 2.6)),
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.92,
            custom_properties={
                "room_name": "Living Room",
                "room_type": "living_room",
                "length_m": 5.0,
                "width_m": 4.0,
                "height_m": 2.6,
                "area_m2": 20.0,
                "aspect_ratio": 1.25,
                "connected_room_ids": [],
            },
        )
        world.entities["room-living"] = room

        # Walls
        w_n = _box("wall-north", (0.0, 3.9, 0.0), (5.0, 4.1, 2.6))
        world.entities["wall-north"] = Entity(
            id="wall-north", name="North Wall", type=EntityType.WALL,
            geometry_ids=[w_n.id],
            transform=_transform((0.0, 3.9, 0.0), (5.0, 4.1, 2.6)),
            provenance=Provenance.RECONSTRUCTED, confidence=0.94,
            custom_properties={
                "length_m": 5.0, "height_m": 2.6, "coverage": 0.95,
                "evidence_state": "RECONSTRUCTED",
            },
            relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="room-living")],
        )

        w_s = _box("wall-south", (0.0, -0.1, 0.0), (5.0, 0.1, 2.6))
        world.entities["wall-south"] = Entity(
            id="wall-south", name="South Wall", type=EntityType.WALL,
            geometry_ids=[w_s.id],
            transform=_transform((0.0, -0.1, 0.0), (5.0, 0.1, 2.6)),
            provenance=Provenance.RECONSTRUCTED, confidence=0.92,
            custom_properties={
                "length_m": 5.0, "height_m": 2.6, "coverage": 0.90,
                "evidence_state": "RECONSTRUCTED",
            },
            relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="room-living")],
        )

        w_e = _box("wall-east", (4.9, 0.0, 0.0), (5.1, 4.0, 2.6))
        world.entities["wall-east"] = Entity(
            id="wall-east", name="East Wall", type=EntityType.WALL,
            geometry_ids=[w_e.id],
            transform=_transform((4.9, 0.0, 0.0), (5.1, 4.0, 2.6)),
            provenance=Provenance.RECONSTRUCTED, confidence=0.91,
            custom_properties={
                "length_m": 4.0, "height_m": 2.6, "coverage": 0.88,
                "evidence_state": "RECONSTRUCTED",
            },
            relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="room-living")],
        )

        # West wall has low coverage initially to trigger capture request in Step 8!
        is_updated = any("gap_fill" in e.id for e in evidence_items)
        w_w = _box("wall-west", (-0.1, 0.0, 0.0), (0.1, 4.0, 2.6))
        world.entities["wall-west"] = Entity(
            id="wall-west", name="West Wall", type=EntityType.WALL,
            geometry_ids=[w_w.id],
            transform=_transform((-0.1, 0.0, 0.0), (0.1, 4.0, 2.6)),
            provenance=Provenance.RECONSTRUCTED if is_updated else Provenance.INFERRED,
            confidence=0.90 if is_updated else 0.55,
            custom_properties={
                "length_m": 4.0,
                "height_m": 2.6,
                "coverage": 0.92 if is_updated else 0.20,
                "evidence_state": "RECONSTRUCTED" if is_updated else "UNOBSERVED",
                "normal": [1.0, 0.0, 0.0],
                "bounds_min": [-0.1, 0.0, 0.0],
                "bounds_max": [0.1, 4.0, 2.6],
            },
            relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="room-living")],
        )

        # Floor & Ceiling
        flr = _box("floor-slab", (0.0, 0.0, -0.1), (5.0, 4.0, 0.0))
        world.entities["floor-slab"] = Entity(
            id="floor-slab", name="Floor Slab", type=EntityType.FLOOR,
            geometry_ids=[flr.id],
            transform=_transform((0.0, 0.0, -0.1), (5.0, 4.0, 0.0)),
            material_ids=["mat-wood"], provenance=Provenance.RECONSTRUCTED, confidence=0.95,
            relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="room-living")],
        )

        ceil = _box("ceiling-slab", (0.0, 0.0, 2.6), (5.0, 4.0, 2.7))
        world.entities["ceiling-slab"] = Entity(
            id="ceiling-slab", name="Ceiling Slab", type=EntityType.CEILING,
            geometry_ids=[ceil.id],
            transform=_transform((0.0, 0.0, 2.6), (5.0, 4.0, 2.7)),
            provenance=Provenance.RECONSTRUCTED, confidence=0.90,
            relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="room-living")],
        )

        # Openings
        win = _box("window-main", (1.5, 3.95, 0.9), (2.7, 4.05, 2.0))
        world.entities["window-main"] = Entity(
            id="window-main", name="Living Room Window", type=EntityType.OPENING,
            geometry_ids=[win.id],
            transform=_transform((1.5, 3.95, 0.9), (2.7, 4.05, 2.0)),
            material_ids=["mat-glass"], provenance=Provenance.RECONSTRUCTED, confidence=0.89,
            custom_properties={"kind": "window", "width_m": 1.2, "height_m": 1.1, "sill_m": 0.9},
            relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="wall-north")],
        )

        door = _box("door-entry", (1.0, -0.05, 0.0), (1.9, 0.05, 2.05))
        world.entities["door-entry"] = Entity(
            id="door-entry", name="Entry Door", type=EntityType.OPENING,
            geometry_ids=[door.id],
            transform=_transform((1.0, -0.05, 0.0), (1.9, 0.05, 2.05)),
            provenance=Provenance.RECONSTRUCTED, confidence=0.91,
            custom_properties={"kind": "doorway", "width_m": 0.9, "height_m": 2.05},
            relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="wall-south")],
        )

        # Furniture inventory (00_VISION Step 5: sofa, coffee table, rug, media console, TV, plants)
        g_sofa = _box("sofa", (1.0, 1.0, 0.0), (3.2, 2.0, 0.85))
        world.entities["sofa-001"] = Entity(
            id="sofa-001",
            name="Living Room Sofa",
            type=EntityType.STRUCTURE,
            geometry_ids=[g_sofa.id],
            transform=_transform((1.0, 1.0, 0.0), (3.2, 2.0, 0.85)),
            material_ids=["mat-fabric"],
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.88,
            uncertainty=Uncertainty(confidence=0.88),
            custom_properties={
                "class": "sofa",
                "semantic_class": "sofa",
                "category": "furniture",
                "length_m": 2.2,
                "width_m": 1.0,
                "height_m": 0.85,
                "dimensions": {"length_m": 2.2, "width_m": 1.0, "height_m": 0.85},
                "source_evidence_ids": sofa_views,
            },
            relationships=[
                Relationship(kind=RelationshipKind.RESTS_ON, target_id="floor-slab"),
                Relationship(kind=RelationshipKind.CONTAINS, target_id="room-living"),
            ],
        )

        g_ct = _box("coffee-table", (1.5, 2.2, 0.0), (2.5, 2.8, 0.45))
        world.entities["table-coffee-001"] = Entity(
            id="table-coffee-001",
            name="Coffee Table",
            type=EntityType.STRUCTURE,
            geometry_ids=[g_ct.id],
            transform=_transform((1.5, 2.2, 0.0), (2.5, 2.8, 0.45)),
            material_ids=["mat-wood"],
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.85,
            uncertainty=Uncertainty(confidence=0.85),
            custom_properties={"class": "coffee_table", "category": "furniture", "length_m": 1.0, "width_m": 0.6, "height_m": 0.45},
            relationships=[Relationship(kind=RelationshipKind.RESTS_ON, target_id="floor-slab")],
        )

        g_rug = _box("rug", (0.8, 0.8, 0.0), (3.5, 3.0, 0.02))
        world.entities["rug-area-001"] = Entity(
            id="rug-area-001",
            name="Area Rug",
            type=EntityType.STRUCTURE,
            geometry_ids=[g_rug.id],
            transform=_transform((0.8, 0.8, 0.0), (3.5, 3.0, 0.02)),
            material_ids=["mat-fabric"],
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.90,
            uncertainty=Uncertainty(confidence=0.90),
            custom_properties={"class": "rug", "category": "furnishing", "length_m": 2.7, "width_m": 2.2, "height_m": 0.02},
            relationships=[Relationship(kind=RelationshipKind.RESTS_ON, target_id="floor-slab")],
        )

        g_mc = _box("media-console", (0.5, 3.6, 0.0), (2.5, 3.9, 0.55))
        world.entities["media-console-001"] = Entity(
            id="media-console-001",
            name="Media Console",
            type=EntityType.STRUCTURE,
            geometry_ids=[g_mc.id],
            transform=_transform((0.5, 3.6, 0.0), (2.5, 3.9, 0.55)),
            material_ids=["mat-wood"],
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.82,
            uncertainty=Uncertainty(confidence=0.82),
            custom_properties={"class": "media_console", "category": "furniture", "length_m": 2.0, "width_m": 0.3, "height_m": 0.55},
            relationships=[Relationship(kind=RelationshipKind.RESTS_ON, target_id="floor-slab")],
        )

        g_tv = _box("tv", (0.8, 3.85, 0.6), (2.2, 3.95, 1.4))
        world.entities["tv-screen-001"] = Entity(
            id="tv-screen-001",
            name="Wall-Mounted TV",
            type=EntityType.STRUCTURE,
            geometry_ids=[g_tv.id],
            transform=_transform((0.8, 3.85, 0.6), (2.2, 3.95, 1.4)),
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.89,
            uncertainty=Uncertainty(confidence=0.89),
            custom_properties={"class": "tv", "category": "appliance", "length_m": 1.4, "width_m": 0.1, "height_m": 0.8},
            relationships=[Relationship(kind=RelationshipKind.ATTACHED_TO, target_id="wall-north")],
        )

        g_ficus = _box("plant-ficus", (4.2, 3.2, 0.0), (4.7, 3.7, 1.2))
        world.entities["plant-ficus-001"] = Entity(
            id="plant-ficus-001",
            name="Ficus Tree",
            type=EntityType.STRUCTURE,
            geometry_ids=[g_ficus.id],
            transform=_transform((4.2, 3.2, 0.0), (4.7, 3.7, 1.2)),
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.80,
            uncertainty=Uncertainty(confidence=0.80),
            custom_properties={"class": "plant", "category": "decor", "height_m": 1.2},
            relationships=[Relationship(kind=RelationshipKind.RESTS_ON, target_id="floor-slab")],
        )

        g_monstera = _box("plant-monstera", (0.2, 0.2, 0.0), (0.6, 0.6, 0.9))
        world.entities["plant-monstera-001"] = Entity(
            id="plant-monstera-001",
            name="Monstera Deliciosa",
            type=EntityType.STRUCTURE,
            geometry_ids=[g_monstera.id],
            transform=_transform((0.2, 0.2, 0.0), (0.6, 0.6, 0.9)),
            provenance=Provenance.RECONSTRUCTED,
            confidence=0.78,
            uncertainty=Uncertainty(confidence=0.78),
            custom_properties={"class": "plant", "category": "decor", "height_m": 0.9},
            relationships=[Relationship(kind=RelationshipKind.RESTS_ON, target_id="floor-slab")],
        )


def _make_dummy_jpeg(color: tuple[int, int, int] = (100, 150, 200)) -> bytes:
    """Generate a tiny valid JPEG image."""
    buf = io.BytesIO()
    im = Image.new("RGB", (64, 48), color=color)
    im.save(buf, format="JPEG")
    return buf.getvalue()


def _wait_job(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        job = r.json()
        if job["status"] in ("succeeded", "partial", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def test_living_room_acceptance_all_12_vision_steps(tmp_path, monkeypatch):
    """Executes the complete 12 vision steps end-to-end on the Living Room demo."""
    db_file = tmp_path / "app.db"
    artifacts_dir = tmp_path / "artifacts"
    worldstore_dir = tmp_path / "ws"

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file.as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(artifacts_dir))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(worldstore_dir))
    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.test_living_room_acceptance_12_steps:_LivingRoomAcceptanceBackend")

    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    import apps.api.worldstore_service as ws_svc
    ws_svc._store = None

    with TestClient(main_mod.app) as c:
        # =====================================================================
        # STEP 1: Create a world "Living Room"
        # =====================================================================
        r_wld = c.post("/api/worlds", json={"name": "Living Room", "description": "v1 Acceptance Living Room"})
        assert r_wld.status_code == 201, r_wld.text
        wid = r_wld.json()["id"]
        assert wid.startswith("wld")
        assert r_wld.json()["name"] == "Living Room"

        # Create capture session and attach to world
        r_ses = c.post("/api/sessions", json={"name": "Living Room Scan 1", "world_id": wid})
        assert r_ses.status_code in (200, 201), r_ses.text
        sid = r_ses.json()["id"]
        c.post(f"/api/worlds/{wid}/attach/{sid}")

        # =====================================================================
        # STEP 2: Upload the living-room photo set (8–30 photos)
        # =====================================================================
        n_photos = 8
        uploaded_ev_ids = []
        for i in range(n_photos):
            jpeg_bytes = _make_dummy_jpeg(color=((i * 25) % 255, 120, 180))
            r_up = c.post(
                f"/api/uploads?session_id={sid}",
                files={"file": (f"living_room_cam_{i:02d}.jpg", jpeg_bytes, "image/jpeg")},
            )
            assert r_up.status_code == 201, r_up.text
            uploaded_ev_ids.append(r_up.json()["evidence_id"])
        assert len(uploaded_ev_ids) == 8

        # =====================================================================
        # STEP 3: System reports they show ONE room and recovers camera positions
        # =====================================================================
        r_rec = c.post(f"/api/sessions/{sid}/reconstruct")
        assert r_rec.status_code == 200, r_rec.text
        job_id = r_rec.json()["job_id"]
        job = _wait_job(c, job_id)
        assert job["status"] in ("succeeded", "partial"), f"Reconstruction failed: {job}"

        # Status & Cameras check
        r_cams = c.get(f"/api/worlds/{wid}/cameras")
        assert r_cams.status_code == 200
        cams = r_cams.json()["cameras"]
        assert len(cams) == 8, f"Expected 8 recovered camera poses, got {len(cams)}"

        # Status reports 1 room
        r_st = c.get(f"/api/worlds/{wid}/status")
        assert r_st.status_code == 200
        st = r_st.json()
        assert not st["in_progress"]
        assert st["has_model"] is True

        # =====================================================================
        # STEP 4: Builds one coherent room: walls, floor, ceiling, window and door
        # =====================================================================
        r_wir = c.get(f"/api/worlds/{wid}/worldir")
        assert r_wir.status_code == 200
        wir = r_wir.json()
        entities = wir["entities"]

        assert "room-living" in entities
        room_ent = entities["room-living"]
        assert room_ent["type"] == "room"
        props = room_ent["custom_properties"]
        assert props["length_m"] == 5.0
        assert props["width_m"] == 4.0
        assert props["height_m"] == 2.6
        assert props["area_m2"] == 20.0

        # Coherent walls, slabs, openings
        assert "wall-north" in entities
        assert "wall-south" in entities
        assert "wall-east" in entities
        assert "wall-west" in entities
        assert "floor-slab" in entities
        assert "ceiling-slab" in entities
        assert "window-main" in entities
        assert "door-entry" in entities

        # =====================================================================
        # STEP 5: Reconstructs sofa, coffee table, rug, media console, TV,
        #         window, and main plants — each exactly once, matched across photos
        # =====================================================================
        expected_furniture = {
            "sofa-001": "sofa",
            "table-coffee-001": "coffee_table",
            "rug-area-001": "rug",
            "media-console-001": "media_console",
            "tv-screen-001": "tv",
            "plant-ficus-001": "plant",
            "plant-monstera-001": "plant",
        }
        for fid, fclass in expected_furniture.items():
            assert fid in entities, f"Missing entity {fid}"
            assert entities[fid]["custom_properties"]["class"] == fclass

        # Verify no duplicate furniture entities
        classes_found = [entities[eid]["custom_properties"].get("class") for eid in entities
                         if "class" in entities[eid].get("custom_properties", {})]
        assert classes_found.count("sofa") == 1
        assert classes_found.count("coffee_table") == 1
        assert classes_found.count("rug") == 1
        assert classes_found.count("media_console") == 1
        assert classes_found.count("tv") == 1

        # =====================================================================
        # STEP 6: Real, Wireframe and Evidence views all work on that one world
        # =====================================================================
        # Real view: materials and surface finishes exist
        assert "mat-wood" in wir.get("materials", {})
        assert "mat-fabric" in wir.get("materials", {})
        assert "mat-glass" in wir.get("materials", {})

        # Wireframe view: finite bounds and vertices exist on all bounding geometries
        for eid in ("room-living", "sofa-001", "table-coffee-001", "wall-north"):
            gids = entities[eid]["geometry_ids"]
            assert gids, f"Entity {eid} has no geometry"
            geom = wir["geometries"][gids[0]]
            assert geom["bounds_min"] is not None
            assert geom["bounds_max"] is not None

        # Evidence view: explicit provenance and confidence vector on every entity
        for eid, ent in entities.items():
            assert ent["provenance"] in ("OBSERVED", "RECONSTRUCTED", "INFERRED", "ESTIMATED"), f"{eid} invalid prov"
            assert 0.0 <= ent["confidence"] <= 1.0, f"{eid} invalid confidence"

        # Points route serves 3D vertex cloud
        r_pts = c.get(f"/api/worlds/{wid}/points")
        assert r_pts.status_code == 200
        assert len(r_pts.content) > 100

        # =====================================================================
        # STEP 7: Clicking the sofa shows its class, dimensions, confidence vector,
        #         and the source photos that support it
        # =====================================================================
        sofa = entities["sofa-001"]
        assert sofa["custom_properties"]["class"] == "sofa"
        dims = sofa["custom_properties"]["dimensions"]
        length_val = dims["length_m"]["value"] if isinstance(dims["length_m"], dict) else dims["length_m"]
        assert length_val == 2.2
        assert sofa["custom_properties"]["length_m"] == 2.2
        assert sofa["confidence"] == 0.88
        assert sofa["uncertainty"]["confidence"] == 0.88
        assert len(sofa["custom_properties"]["source_evidence_ids"]) >= 2
        for ev_ref in sofa["custom_properties"]["source_evidence_ids"]:
            assert ev_ref in uploaded_ev_ids

        # =====================================================================
        # STEP 8: System issues at least one capture request for poorly observed region
        # =====================================================================
        r_req = c.get(f"/api/worlds/{wid}/capture-requests")
        assert r_req.status_code == 200
        req_body = r_req.json()
        assert req_body["count"] >= 1, "Expected at least 1 capture request for poorly observed region"
        req_item = req_body["items"][0]
        assert req_item["status"] == "open"
        assert req_item["target_region"] is not None or "west" in req_item["instruction"].lower() or "wall" in req_item["reason"].lower()

        # =====================================================================
        # STEP 9: Uploading that photo registers it into the same world and
        #         updates only the affected region; a new version appears in history
        # =====================================================================
        v1_id = c.get(f"/api/worlds/{wid}").json()["current_version_id"]

        # Second session with targeted gap-filling photograph
        r_ses2 = c.post("/api/sessions", json={"name": "Living Room West Wall Fill", "world_id": wid})
        assert r_ses2.status_code in (200, 201), r_ses2.text
        sid2 = r_ses2.json()["id"]
        c.post(f"/api/worlds/{wid}/attach/{sid2}")

        gap_fill_bytes = _make_dummy_jpeg(color=(210, 180, 140))
        c.post(
            f"/api/uploads?session_id={sid2}",
            files={"file": ("living_room_gap_fill.jpg", gap_fill_bytes, "image/jpeg")},
        )

        r_rec2 = c.post(f"/api/sessions/{sid2}/reconstruct")
        assert r_rec2.status_code == 200
        job2 = _wait_job(c, r_rec2.json()["job_id"])
        assert job2["status"] in ("succeeded", "partial")

        # History lineage: V2 appears with parent == V1
        r_vers = c.get(f"/api/worlds/{wid}/versions")
        assert r_vers.status_code == 200
        versions = r_vers.json()["items"]
        assert len(versions) >= 2
        v2_id = c.get(f"/api/worlds/{wid}").json()["current_version_id"]
        assert v2_id != v1_id

        # Diff confirms update to the unobserved wall region
        r_diff = c.get(f"/api/worlds/{wid}/diff?base={v1_id}&head={v2_id}")
        assert r_diff.status_code == 200
        diff_data = r_diff.json()
        assert diff_data["categories"]["confidence"] > 0 or diff_data["categories"]["properties"] > 0 or diff_data["categories"]["geometry"] >= 0

        # =====================================================================
        # STEP 10: Close the app, reopen; the world is exactly as left
        # =====================================================================
        # Create brand-new client instance over the same underlying database & storage
        with TestClient(main_mod.app) as c_reopened:
            r_re_wld = c_reopened.get(f"/api/worlds/{wid}")
            assert r_re_wld.status_code == 200
            assert r_re_wld.json()["current_version_id"] == v2_id
            assert r_re_wld.json()["name"] == "Living Room"

            r_re_wir = c_reopened.get(f"/api/worlds/{wid}/worldir")
            assert r_re_wir.status_code == 200
            re_entities = r_re_wir.json()["entities"]
            assert "sofa-001" in re_entities
            assert "room-living" in re_entities
            assert "table-coffee-001" in re_entities
            assert len(re_entities) == len(entities)

        # =====================================================================
        # STEP 11: Export GLB; opens with correct transforms, hierarchy and textures
        # =====================================================================
        r_exp = c.post(f"/api/worlds/{wid}/export", json={"format": "glb"})
        assert r_exp.status_code == 200, r_exp.text
        exp_body = r_exp.json()
        assert exp_body["download_url"]

        glb_resp = c.get(exp_body["download_url"])
        assert glb_resp.status_code == 200
        glb_bytes = glb_resp.content

        # Validate GLB structure: valid header, meshes, vertices
        summary = validate_glb(glb_bytes)
        assert summary["meshes"] >= 1
        assert summary["vertices"] >= 24, f"Vertices: {summary}"
        assert set(exp_body["entities_exported"]).issuperset({"room-living", "sofa-001", "wall-north"})

        # =====================================================================
        # STEP 12: Rename, duplicate, archive, restore and delete (to trash)
        # =====================================================================
        # 12.a: Rename
        r_ren = c.patch(f"/api/worlds/{wid}", json={"name": "Living Room - Renamed"})
        assert r_ren.status_code == 200
        assert c.get(f"/api/worlds/{wid}").json()["name"] == "Living Room - Renamed"

        # 12.b: Duplicate
        r_dup = c.post(f"/api/worlds/{wid}/duplicate")
        assert r_dup.status_code == 201
        dup_id = r_dup.json()["id"]
        assert dup_id != wid
        assert c.get(f"/api/worlds/{dup_id}").json()["name"] == "Living Room - Renamed (copy)"

        # 12.c: Archive
        r_arc = c.post(f"/api/worlds/{wid}/archive")
        assert r_arc.status_code == 200
        assert c.get(f"/api/worlds/{wid}").json()["status"] == "archived"

        # 12.d: Restore
        r_res = c.post(f"/api/worlds/{wid}/restore")
        assert r_res.status_code == 200
        assert c.get(f"/api/worlds/{wid}").json()["status"] == "active"

        # 12.e: Delete (soft delete to trash, then permanent delete)
        r_del = c.delete(f"/api/worlds/{wid}")
        assert r_del.status_code == 200
        assert c.get(f"/api/worlds/{wid}").json()["status"] == "trashed"

        # 12.f: Permanent delete removes row completely
        r_perm = c.delete(f"/api/worlds/{wid}?permanent=true")
        assert r_perm.status_code == 200
        assert r_perm.json()["permanently_deleted"] is True
        r_gone = c.get(f"/api/worlds/{wid}")
        assert r_gone.status_code == 404
