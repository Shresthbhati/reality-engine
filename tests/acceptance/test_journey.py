"""Phase G acceptance journey: ONE automated test proving the pieces work as ONE product.

world -> photo evidence -> real reconstruction -> procedural build (room/corridor/apartment/building/district)
-> scoped edit -> completion proposal -> assistant edit -> measure -> export (Godot bundle + glTF) -> restart
-> diff -> delete with dependents.

Everything runs through the real FastAPI app (TestClient) over a temp SQLite DB + WorldStore. Nothing is mocked:
the reconstruction step feeds real photographs (datasets/south_building) to the real COLMAP/MiDaS engines; if those
are not available the two reconstruction steps are reported SKIPPED with the reason (never faked) and the rest of
the journey runs on the procedural path.

The one seeded input is the completion shell: completion needs a RECONSTRUCTED floor/wall shell and a facade
photo set only yields INFERRED walls, so the real-photo world is asserted to be honestly REFUSED and the accept
path is exercised on a world seeded with a RECONSTRUCTED shell through the same service the API commits with
(identical to frontend/e2e/seed_shell.py).

A step-by-step table (PASS / SKIPPED(reason) / FAIL) is printed at the end; any FAIL fails the test.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import os
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
from fastapi.testclient import TestClient

DATASET = Path(__file__).resolve().parents[2] / "datasets" / "south_building" / "images"
IMGS = sorted(DATASET.glob("*.JPG")) if DATASET.exists() else []
SIX = IMGS[15:21]  # six strongly overlapping views (see tests/integration/test_progressive_product_journey.py)

pytestmark = [pytest.mark.slow, pytest.mark.e2e]


# ----------------------------------------------------------------------------------------------- harness


class StepSkipped(Exception):
    pass


class Journey:
    def __init__(self) -> None:
        self.rows: list[list[str]] = []

    @contextmanager
    def step(self, n: int, title: str):
        row = [str(n), title, "PASS", ""]
        self.rows.append(row)
        t = time.time()
        try:
            yield row
        except StepSkipped as exc:
            row[2], row[3] = "SKIPPED", str(exc)
        except Exception as exc:  # noqa: BLE001 -- a failing step is reported, later steps still run
            row[2], row[3] = "FAIL", f"{type(exc).__name__}: {exc}"[:400]
        row[3] = (row[3] + f" [{time.time() - t:.1f}s]").strip()

    def table(self) -> str:
        w = max(len(r[1]) for r in self.rows)
        lines = ["", "ACCEPTANCE JOURNEY", "-" * (w + 60)]
        for n, title, status, note in self.rows:
            if status == "SKIPPED":
                lines.append(f"{int(n):>2}. {title:<{w}}  SKIPPED({note.split(' [')[0]})")
            else:
                lines.append(f"{int(n):>2}. {title:<{w}}  {status:<4}  {note}")
        return "\n".join(lines)

    @property
    def failed(self) -> list[list[str]]:
        return [r for r in self.rows if r[2] == "FAIL"]


@contextmanager
def app(root: Path):
    """One app process's worth of state over `root`: the same root -> the same DB + WorldStore (restart proof)."""
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(root / 'app.db').as_posix()}"
    os.environ["STORAGE_ROOT"] = str(root / "artifacts")
    os.environ["WORLDSTORE_ROOT"] = str(root / "ws")
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    import apps.api.worldstore_service as ws

    ws._store = None
    with TestClient(main_mod.app) as c:
        yield c


def _engines() -> tuple[bool, str]:
    if len(IMGS) < 25:
        return False, "South Building photo set not present"
    try:
        from perception.availability import report_real_models

        status = {m.name: m.status for m in report_real_models()}
    except Exception as exc:  # pragma: no cover
        return False, f"availability probe failed: {exc}"
    missing = [n for n in ("colmap", "midas") if status.get(n) != "available"]
    return (not missing), f"real engines unavailable: {missing}"


def _wir(c, wid, version=None) -> dict:
    r = c.get(f"/api/worlds/{wid}/worldir", params={"version": version} if version else None)
    assert r.status_code == 200, r.text
    return r.json()


def _head(c, wid) -> str:
    return c.get(f"/api/worlds/{wid}").json()["current_version_id"]


def _entity_fp(wir: dict, eid: str) -> str:
    e = wir["entities"][eid]
    return json.dumps([e, [wir["geometries"].get(g) for g in e.get("geometry_ids", [])]], sort_keys=True)


def _lineage(c, wid) -> list[str]:
    """Version ids root -> HEAD, walked through parent links (not list order)."""
    items = {v["id"]: v for v in c.get(f"/api/worlds/{wid}/versions").json()["items"]}
    chain, cur = [], _head(c, wid)
    while cur:
        chain.append(cur)
        cur = items[cur]["parent_version_id"]
    return chain[::-1]


def _snapshot(wir: dict) -> dict[str, str]:
    return {eid: _entity_fp(wir, eid) for eid in wir["entities"]}


def _owner(wir: dict, eid: str) -> str | None:
    return (wir["entities"][eid].get("custom_properties") or {}).get("generated_by")


def _prov(wir: dict) -> dict:
    return dict(Counter(e["provenance"] for e in wir["entities"].values()))


def _settle(c, wid, timeout=900.0) -> dict:
    end = time.time() + timeout
    st = None
    while time.time() < end:
        st = c.get(f"/api/worlds/{wid}/status").json()
        if not st["in_progress"]:
            return st
        time.sleep(1.0)
    raise AssertionError(f"world {wid} never settled; last status: {st}")


def _seed_shell_world(c) -> str:
    """A world whose HEAD is a RECONSTRUCTED floor + wall shell, committed via the API's own write path."""
    from provenance import Provenance
    from world_ir import Entity, EntityType
    from world_ir.schema_v1 import Geometry, GeometryType, Vector3
    from world_ir.world_v1 import WorldIR
    from apps.api import worldstore_service as svc
    import apps.api.db as db_mod

    wid = c.post("/api/worlds", json={"name": "Seeded shell"}).json()["id"]
    wir = WorldIR(id=wid)
    for name, hi in {"floor": (6, 0.1, 5), "wall-n": (6, 2.6, 0.2)}.items():
        gid = f"g-{name}"
        wir.geometries[gid] = Geometry(id=gid, type=GeometryType.BOX, bounds_min=Vector3(0, 0, 0),
                                       bounds_max=Vector3(*hi), provenance=Provenance.RECONSTRUCTED)
        wir.entities[name] = Entity(id=name, type=EntityType.FLOOR if name == "floor" else EntityType.WALL,
                                    name=name, geometry_ids=[gid], provenance=Provenance.RECONSTRUCTED, confidence=0.8)

    async def go():
        async with db_mod.get_sessionmaker()() as db:
            await svc.commit_version(db, world_id=wid, world=wir, parent=None)

    asyncio.run(go())
    return wid


APARTMENT = {
    "name": "flat", "footprint": [10.0, 8.0], "entrance": "hall", "seed": 1, "plan_index": 0,
    "origin": [30.0, 0.0, 0.0],
    "rooms": [
        {"id": "living", "kind": "living", "min_area": 18, "max_area": 30},
        {"id": "kitchen", "kind": "kitchen", "min_area": 8, "max_area": 14},
        {"id": "bedroom_1", "kind": "bedroom", "min_area": 12, "max_area": 18},
        {"id": "bath_1", "kind": "bathroom", "min_area": 4, "max_area": 7},
        {"id": "hall", "kind": "hallway", "min_area": 6, "max_area": 12, "aspect_max": 3.5},
    ],
    "adjacency": [["living", "kitchen", "required"], ["living", "hall", "required"],
                  ["hall", "bedroom_1", "required"], ["bedroom_1", "bath_1", "preferred"]],
}
BUILDING = {
    "name": "tower", "footprint": [12.0, 10.0], "storeys": 3, "storey_heights": [3.0, 3.0, 3.0],
    "origin": [60.0, 0.0, 0.0],
}
DISTRICT = {"name": "dist", "blocks_x": 3, "blocks_y": 2, "seed": 5, "detail": {"max_full": 1},
            "origin": [120.0, 0.0, 0.0]}


# ----------------------------------------------------------------------------------------------- the journey


def test_acceptance_journey(tmp_path: Path, capsys):
    J = Journey()
    S: dict = {}  # shared journey state
    engines_ok, engines_why = _engines()

    with app(tmp_path) as c:
        # (1) create world --------------------------------------------------------------------------
        with J.step(1, "create world") as row:
            r = c.post("/api/worlds", json={"name": "Acceptance journey"})
            assert r.status_code == 201, r.text
            S["wid"] = wid = r.json()["id"]
            assert _head(c, wid) is None
            assert c.get(f"/api/worlds/{wid}/versions").json()["items"] == []
            assert any(w["id"] == wid for w in c.get("/api/worlds").json()["items"])
            row[3] = "fresh world has no HEAD and no versions"

        wid = S.get("wid")

        # (2) ingest photo evidence -----------------------------------------------------------------
        with J.step(2, "ingest evidence (6 real photos)") as row:
            if not engines_ok:
                raise StepSkipped(engines_why)
            files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in SIX]
            r = c.post("/api/reconstructions", files=files, data={"world_id": wid})
            assert r.status_code == 202, r.text
            body = r.json()
            assert body["world_id"] == wid and body["created_world"] is False
            assert len(body["evidence"]) == 6 and body["rejected"] == []
            S["evidence"] = body["evidence"]
            for ev, p in zip(sorted(body["evidence"], key=lambda e: e["name"]), SIX):
                art = c.get(f"/api/evidence/{ev['id']}/artifact")
                assert art.status_code == 200 and art.content == p.read_bytes(), ev["name"]
            row[3] = "6 photos stored byte-exact; reconstruction job accepted (202)"

        # (3) run the pipeline as far as it really goes ----------------------------------------------
        with J.step(3, "reconstruct: pipeline stages -> WorldIR") as row:
            if not engines_ok:
                raise StepSkipped(engines_why)
            st = _settle(c, wid)
            assert st["state"] in ("READY_TO_INSPECT", "PARTIALLY_COMPLETE"), st["state"]
            m = st["model"]
            assert m["images_used"] == 6 and m["images_registered"] == 6, m
            jobs = [j for j in c.get("/api/jobs").json()["items"] if j["type"] == "RECONSTRUCT_SESSION"]
            assert jobs and jobs[0]["status"] == "succeeded", jobs
            v1 = st["versions"][0]
            stages = {
                "job": f"{jobs[0]['status']}@{jobs[0]['stage']}",
                "sfm_level": m["level"], "model_state": m["model_state"], "scale": m["scale"]["state"],
                "registered": f"{m['images_registered']}/{m['images_used']}",
                "dense": v1.get("dense", {}).get("state"), "verdict": v1.get("verdict"),
            }
            S["recon_stages"] = stages
            wir = _wir(c, wid)
            walls = [e for e in wir["entities"].values() if e["type"] == "wall"]
            assert len(walls) >= 3, "six facade photos must yield several wall planes"
            assert all(e["provenance"] == "INFERRED" for e in walls)
            ply = c.get(f"/api/worlds/{wid}/points")
            assert ply.status_code == 200 and b"element vertex" in ply.content[:2000]
            assert len(c.get(f"/api/worlds/{wid}/cameras").json()["cameras"]) == 6
            # provenance traces an entity back to the exact uploaded file
            ev_ids = {e["id"] for e in S["evidence"]}
            prov = c.get(f"/api/worlds/{wid}/entities/{walls[0]['id']}/provenance").json()
            assert prov["trace_level"] in ("observation", "session"), prov
            assert prov["evidence"] and {e["evidence_id"] for e in prov["evidence"]} <= ev_ids
            S["trace_level"] = prov["trace_level"]
            S["traced_wall"] = walls[0]["id"]
            S["recon_head"] = _head(c, wid)
            S["recon_ids"] = sorted(wir["entities"])
            S["recon_snapshot"] = _snapshot(wir)
            row[3] = " ".join(f"{k}={v}" for k, v in stages.items())

        # (4) build rooms / corridor / apartment / building / district ------------------------------
        with J.step(4, "build room+corridor+apartment+building+district") as row:
            built = {}

            def build(path, body, key):
                before = _head(c, wid)
                r = c.post(f"/api/worlds/{wid}/{path}", json=body)
                assert r.status_code == 201, (path, r.status_code, r.text)
                out = r.json()
                assert out["version_id"] == _head(c, wid) != before, path
                assert out["entity_ids_added"], path
                built[key] = len(out["entity_ids_added"])
                return out

            build("rooms", {"name": "Study", "width": 4, "depth": 5, "height": 2.8, "seed": 7}, "room")
            build("corridors", {"name": "Hall", "length": 4.0, "attach": {"to": "Study-room", "side": "east"}}, "corridor")
            build("apartments", APARTMENT, "apartment")
            build("buildings", BUILDING, "building")
            build("districts", DISTRICT, "district")
            wir = _wir(c, wid)
            gen = {g["entity_id"]: g for g in c.get(f"/api/worlds/{wid}/generated").json()}
            assert set(gen) == {"Study-room", "Hall-corridor", "flat-apartment", "tower-building", "dist-district"}
            produced = [e for e in wir["entities"].values() if _owner(wir, e["id"])]
            assert len(produced) == sum(built.values())
            assert {e["provenance"] for e in produced} <= {"PROCEDURAL", "GENERATED"}
            assert {e["type"] for e in produced} >= {"room", "wall", "floor", "corridor", "building"}
            if S.get("recon_snapshot"):  # building on a reconstruction must not disturb a single reconstructed entity
                assert {k: _entity_fp(wir, k) for k in S.get("recon_snapshot", {})} == S.get("recon_snapshot", {})
            # evidence tracing survives later procedural versions; generated entities never claim photo evidence
            if engines_ok:
                p = c.get(f"/api/worlds/{wid}/entities/{S['traced_wall']}/provenance").json()
                assert p["trace_level"] != "none" and p["evidence"], f"reconstructed wall lost its evidence trace at HEAD: {p}"
            p = c.get(f"/api/worlds/{wid}/entities/Study-floor/provenance").json()
            assert p["trace_level"] == "none" and p["evidence"] == [], f"procedural entity traced to photos: {p}"
            S["built_head"] = _head(c, wid)
            row[3] = "entities added: " + ", ".join(f"{k}={v}" for k, v in built.items())

        # (5) scoped edit leaves unrelated entities byte-identical ------------------------------------
        with J.step(5, "scoped PUT edit: unrelated entities byte-identical") as row:
            before = _wir(c, wid)
            snap = _snapshot(before)
            r = c.put(f"/api/worlds/{wid}/rooms/Study", json={"width": 6.0})
            assert r.status_code == 200, r.text
            out = r.json()
            after = _wir(c, wid)
            assert set(after["entities"]) == set(before["entities"])
            changed = {e for e in after["entities"] if _entity_fp(after, e) != snap[e]}
            assert changed == set(out["entity_ids_changed"]) and out["changed"] == len(changed) > 0
            assert all(_owner(after, e) in ("Study-room", "Hall-corridor") for e in changed), sorted(changed)
            unrelated = [e for e in after["entities"] if e not in changed]
            assert len(unrelated) == out["unchanged"] > len(changed)
            assert all(_entity_fp(after, e) == snap[e] for e in unrelated)
            owners_untouched = {_owner(after, e) for e in unrelated}
            assert {"flat-apartment", "tower-building", "dist-district"} <= owners_untouched
            if engines_ok:
                assert all(e in unrelated for e in S.get("recon_ids", [])), "a reconstructed entity was touched by a room edit"
            fl = after["geometries"]["geom-Study-floor"]
            assert round(fl["bounds_max"]["x"] - fl["bounds_min"]["x"], 6) == 6.0
            S["study_floor"] = fl
            row[3] = f"changed={len(changed)} (Study+Hall), byte-identical={len(unrelated)}"

        # (6) completion proposal accept ------------------------------------------------------------
        with J.step(6, "completion: propose -> diff -> accept (GENERATED)") as row:
            note = []
            if engines_ok:  # the real-photo shell has only INFERRED walls: completion must refuse, naming why
                refused = c.post(f"/api/worlds/{wid}/completion/proposals", json={"template": "studio"})
                assert refused.status_code == 422 and "no usable shell" in refused.text, refused.text
                note.append("real-photo world honestly refused (INFERRED walls are not a shell)")
            sw = _seed_shell_world(c)
            head0 = _head(c, sw)
            r = c.post(f"/api/worlds/{sw}/completion/proposals", json={"template": "studio", "alternatives": 2})
            assert r.status_code == 201, r.text
            props = r.json()["proposals"]
            assert props and _head(c, sw) == head0, "proposing must not move HEAD"
            assert props[0]["status"] == "proposed" and "GENERATED" in props[0]["label"]
            pid = props[0]["id"]
            d = c.get(f"/api/worlds/{sw}/completion/proposals/{pid}/diff").json()
            assert d["counts"]["added"] > 0 and d["counts"]["changed"] == 0 and d["counts"]["removed"] == 0
            assert d["provenance"]["existing"] == {"RECONSTRUCTED": 2}
            acc = c.post(f"/api/worlds/{sw}/completion/proposals/{pid}/accept")
            assert acc.status_code == 200, acc.text
            assert _head(c, sw) == acc.json()["version_id"] != head0
            w2 = _wir(c, sw)
            made = [e for e in w2["entities"].values() if (e.get("custom_properties") or {}).get("completion_proposal") == pid]
            assert len(made) == d["counts"]["added"] and {e["provenance"] for e in made} == {"GENERATED"}
            assert w2["entities"]["floor"]["provenance"] == "RECONSTRUCTED"
            again = c.post(f"/api/worlds/{sw}/completion/proposals/{pid}/accept")
            assert again.status_code == 409
            note.append(f"accepted {len(made)} GENERATED entities on a seeded RECONSTRUCTED shell")
            row[3] = "; ".join(note)

        # (7) assistant propose_edit + confirm -------------------------------------------------------
        with J.step(7, "assistant: propose_edit -> confirm") as row:
            head = _head(c, wid)
            listed = c.post(f"/api/worlds/{wid}/assistant/call", json={"tool": "list_entities", "args": {"kind": "room"}})
            assert listed.status_code == 200 and listed.json()["result"]["total"] > 0, listed.text
            op = {"operation": {"op": "update_generated", "kind": "room", "name": "Study", "patch": {"height": 3.2}}}
            r = c.post(f"/api/worlds/{wid}/assistant/call", json={"tool": "propose_edit", "args": op})
            assert r.status_code == 200, r.text
            plan = r.json()["result"]
            assert plan["requires_confirmation"] is True and plan["preview"]["entities_changed"] > 0
            assert _head(c, wid) == head, "a proposal must not move HEAD"
            cf = c.post(f"/api/worlds/{wid}/assistant/confirm", json={"plan_id": plan["plan_id"]})
            assert cf.status_code == 200, cf.text
            assert _head(c, wid) == cf.json()["version_id"] != head
            spec = c.post(f"/api/worlds/{wid}/assistant/call",
                          json={"tool": "get_entity", "args": {"entity_id": "Study-room"}}).json()["result"]
            assert spec["spec"]["params"]["height"] == 3.2 and spec["spec"]["params"]["width"] == 6.0
            assert c.post(f"/api/worlds/{wid}/assistant/confirm", json={"plan_id": plan["plan_id"]}).status_code == 409
            row[3] = f"plan changed {plan['preview']['entities_changed']} entities; replay -> 409"

        # (8) measure --------------------------------------------------------------------------------
        with J.step(8, "measure: authored exact + unknown-scale honesty") as row:
            fl = _wir(c, wid)["geometries"]["geom-Study-floor"]
            lo, hi = fl["bounds_min"], fl["bounds_max"]
            pts = [[lo["x"], lo["y"], lo["z"]], [hi["x"], lo["y"], lo["z"]]]
            r = c.post(f"/api/worlds/{wid}/measurements", json={
                "kind": "distance", "points": pts, "scale_state": "authored", "method": "Study floor x-extent"})
            assert r.status_code == 201, r.text
            m1 = r.json()
            assert abs(m1["value"] - 6.0) < 1e-9 and m1["unit"] == "m" and m1["uncertainty"] == 0.0
            r = c.post(f"/api/worlds/{wid}/measurements", json={
                "kind": "distance", "points": [[0, 0, 0], [3, 4, 0]], "scale_state": "unknown", "method": "two picked points"})
            assert r.status_code == 201, r.text
            m2 = r.json()
            assert m2["value"] == 5.0 and m2["unit"] != "m" and m2["uncertainty"] is None
            listed = c.get(f"/api/worlds/{wid}/measurements").json()
            assert {m["id"] for m in listed} == {m1["id"], m2["id"]}
            bad = c.post(f"/api/worlds/{wid}/measurements", json={
                "kind": "distance", "points": pts, "scale_state": "authored", "method": "x", "value": 9.0})
            assert bad.status_code == 422
            S["n_measurements"] = len(listed)
            row[3] = f"authored {m1['value']:.1f} {m1['unit']} (+/-{m1['uncertainty']}); unknown-scale unit={m2['unit']!r}"

        # (9) export godot bundle + glb ---------------------------------------------------------------
        with J.step(9, "export: godot bundle (glb) + gltf") as row:
            r = c.post(f"/api/worlds/{wid}/export", json={"format": "godot"})
            assert r.status_code == 200, r.text
            b = r.json()
            assert b["bundle"] is True and b["version_id"] == _head(c, wid)
            assert {"world.glb", "manifest.json", "scene.tscn"} <= set(b["files"]), b["files"]
            dl = c.get(b["download_url"])
            assert dl.status_code == 200
            bundle = json.loads(dl.content)
            glb = base64.b64decode(bundle["world.glb"]["content"])
            assert glb[:4] == b"glTF" and int.from_bytes(glb[4:8], "little") == 2
            assert int.from_bytes(glb[8:12], "little") == len(glb), "GLB header length must match the file"
            jlen = int.from_bytes(glb[12:16], "little")
            assert glb[16:20] == b"JSON"
            gltf = json.loads(glb[20:20 + jlen])
            assert gltf["asset"]["version"] == "2.0" and len(gltf["meshes"]) > 0 and len(gltf["nodes"]) > 0
            assert len(b["entities_exported"]) > 0
            assert len(b["skip_reasons"]) == len(b["entities_skipped"]), "every skipped entity must say why"
            S["skip_kinds"] = dict(Counter(r[:60] for r in b["skip_reasons"]))
            g = c.post(f"/api/worlds/{wid}/export", json={"format": "gltf"})
            assert g.status_code == 200, g.text
            assert c.post(f"/api/worlds/{wid}/export", json={"format": "not-a-format"}).status_code == 422
            S["export_hash"] = b["content_hash"]
            S["export_version"] = b["version_id"]
            row[3] = (f"glb {len(glb)}B, {len(gltf['meshes'])} meshes, {len(b['entities_exported'])} entities exported, "
                      f"skipped={len(b['entities_skipped'])} {S['skip_kinds']}")

        # state captured for the restart proof
        pre = _wir(c, wid)
        S["pre"] = {"head": _head(c, wid), "entities": len(pre["entities"]), "prov": _prov(pre),
                    "versions": _lineage(c, wid),
                    "snap": _snapshot(pre)}

    # (10) restart: a NEW app instance over the same DB + WorldStore ---------------------------------
    with app(tmp_path) as c2:
        with J.step(10, "restart: HEAD/entities/provenance identical") as row:
            pre = S["pre"]
            assert _head(c2, wid) == pre["head"]
            post = _wir(c2, wid)
            assert len(post["entities"]) == pre["entities"]
            assert _prov(post) == pre["prov"]
            assert _snapshot(post) == pre["snap"], "entity content changed across restart"
            assert _lineage(c2, wid) == pre["versions"]
            assert len(c2.get(f"/api/worlds/{wid}/measurements").json()) == S["n_measurements"]
            e = c2.post(f"/api/worlds/{wid}/export", json={"format": "godot"}).json()
            assert e["content_hash"] == S["export_hash"], "export is not deterministic across restart"
            if engines_ok:
                prov = c2.get(f"/api/worlds/{wid}/entities/{S['traced_wall']}/provenance").json()
                assert prov["trace_level"] == S["trace_level"] and prov["evidence"], prov
                assert c2.get(f"/api/evidence/{S['evidence'][0]['id']}/artifact").status_code == 200
            row[3] = f"HEAD={pre['head']} entities={pre['entities']} provenance={pre['prov']}"

        # (11) diff between first and last version --------------------------------------------------
        with J.step(11, "diff first -> last version") as row:
            versions = _lineage(c2, wid)
            assert set(versions) == set(S["pre"]["versions"])
            first, last = versions[0], versions[-1]
            assert first != last and last == S["pre"]["head"]
            first_wir = _wir(c2, wid, first)
            d = c2.get(f"/api/worlds/{wid}/diff", params={"base": first, "head": last})
            assert d.status_code == 200, d.text
            body = d.json()
            cats = body["categories"]
            assert sum(cats.values()) > 0
            by_kind = {k: {x["entity_id"] for x in body["entity_diffs"] if x["kind"] == k}
                       for k in ("added", "removed", "modified")}
            expected_added = set(S["pre"]["snap"]) - set(first_wir["entities"])
            assert by_kind["added"] == expected_added and expected_added
            assert not by_kind["removed"]
            if engines_ok:  # the reconstructed entities are untouched between V1 and HEAD
                assert first == S["recon_head"]
                assert not set(S.get("recon_ids", [])) & by_kind["modified"]
            same = c2.get(f"/api/worlds/{wid}/diff", params={"base": last, "head": last}).json()
            assert sum(same["categories"].values()) == 0, "diff of a version against itself must be empty"
            row[3] = f"{len(versions)} versions; +{len(by_kind['added'])} added, ~{len(by_kind['modified'])} modified; " \
                     f"categories={ {k: v for k, v in cats.items() if v} }"

        # (12) delete with dependents -> 409, then cascade --------------------------------------------
        with J.step(12, "delete with dependents 409 -> cascade") as row:
            head = _head(c2, wid)
            n_before = len(_wir(c2, wid)["entities"])
            blocked = c2.delete(f"/api/worlds/{wid}/rooms/Study")
            assert blocked.status_code == 409 and "Hall" in blocked.text, blocked.text
            assert _head(c2, wid) == head, "a refused delete must not move HEAD"
            ok = c2.delete(f"/api/worlds/{wid}/rooms/Study", params={"cascade": "true"})
            assert ok.status_code == 200, ok.text
            out = ok.json()
            assert out["cascaded"] == ["Hall"] and out["removed"] > 0
            after = _wir(c2, wid)
            assert len(after["entities"]) == n_before - out["removed"]
            assert "Study-room" not in after["entities"] and "Hall-corridor" not in after["entities"]
            assert not any(_owner(after, e) in ("Study-room", "Hall-corridor") for e in after["entities"])
            assert {"flat-apartment", "tower-building", "dist-district"} <= {_owner(after, e) for e in after["entities"]}
            for eid in S.get("recon_ids", []) if engines_ok else []:
                assert _entity_fp(after, eid) == S.get("recon_snapshot", {})[eid]
            assert c2.delete(f"/api/worlds/{wid}/rooms/Study").status_code == 404
            # deletion is versioned: the previous version is still readable
            assert _wir(c2, wid, head)["entities"].keys() >= {"Study-room", "Hall-corridor"}
            row[3] = f"removed {out['removed']} entities incl. dependent {out['cascaded']}; previous version intact"

    with capsys.disabled():
        print(J.table())
        if engines_ok:
            print("reconstruction stages:", S.get("recon_stages"))
    assert not J.failed, "FAILED steps: " + "; ".join(f"{r[0]} {r[1]}: {r[3]}" for r in J.failed)
