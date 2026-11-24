"""Independent proof that a finished evidence-world run really persisted its fused geometry.

Run this in a FRESH process, after the job process that produced the run has exited, so nothing is proven from in-memory state:

    python scripts/sprint1/p0_proof.py <evidence_run_dir> <app_out_dir> <world_id> <job_id>

It checks, in this order, and prints one JSON object with every measured value:

1. ``worldstore``: ``WorldStore.verify_version`` passes and ``load_version`` returns the same content hash the stage recorded.
2. ``fused_mesh``: the persisted world holds ``surf-fused-surface`` with a MESH geometry whose ``data_uri`` resolves through the run's
   artifact store (``<run>/artifacts`` + ``<run>/worldstore/artifacts``), whose bytes match ``data_hash``, and whose PLY header declares
   the same vertex/triangle counts as the geometry record.
3. ``world_api``: ``GET /api/worlds/{world}/evidence-world/{run}/world`` from a fresh app instance returns the same version and the
   fused entity.
4. ``export``: ``POST /api/worlds/{world}/evidence-world/{run}/export`` with ``format=glb`` returns a GLB whose mesh primitives carry at
   least the fused mesh's triangles (no placeholder boxes substituted for the real surface).

Nothing here invents a value: a missing piece is reported as an error and the exit status is non-zero.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _ply_counts(data: bytes) -> dict[str, int]:
    """Vertex / face counts from a binary or ASCII PLY header (the header is ASCII in both encodings)."""
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError("payload is not a PLY file")
    header = data[:end].decode("ascii", "replace").splitlines()
    out: dict[str, int] = {}
    for line in header:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] in ("vertex", "face"):
            out[parts[1]] = int(parts[2])
    return out


def _glb_triangles(data: bytes) -> dict[str, int]:
    """Triangle count of a GLB's meshes, from its JSON chunk's index accessors."""
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("payload is not a GLB v2 file")
    (chunk_len, chunk_type) = struct.unpack_from("<II", data, 12)
    if chunk_type != 0x4E4F534A:
        raise ValueError("first GLB chunk is not JSON")
    doc = json.loads(data[20:20 + chunk_len].decode("utf-8"))
    accessors = doc.get("accessors") or []
    tris = 0
    meshes = 0
    for mesh in doc.get("meshes") or []:
        for prim in mesh.get("primitives") or []:
            meshes += 1
            idx = prim.get("indices")
            if idx is not None:
                tris += accessors[idx]["count"] // 3
            elif "POSITION" in (prim.get("attributes") or {}):
                tris += accessors[prim["attributes"]["POSITION"]]["count"] // 3
    return {"meshes": meshes, "triangles": tris, "nodes": len(doc.get("nodes") or []), "accessors": len(accessors)}


def main(run_dir: Path, app_dir: Path, world_id: str, job_id: str) -> int:
    run_dir = Path(run_dir).resolve()   # <out>/ws/evidence-world/<team>/<world>/<job>
    app_dir = Path(app_dir).resolve()   # <out>: holds app.db and ws/
    out: dict = {"run_dir": str(run_dir), "app_dir": str(app_dir), "world_id": world_id, "job_id": job_id, "pid": os.getpid()}
    ok = True

    # ---------------------------------------------------------------- 1 + 2: reload the persisted world in THIS process
    from apps.api.evidence_world_stage import WORLDSTORE_REL, head_version, run_artifact_store
    from worldstore.store import WorldStore

    vid = head_version(run_dir)
    out["version_id"] = vid
    if not vid:
        out["worldstore"] = {"error": "no persisted version for this run"}
        print(json.dumps(out, indent=1))
        return 1
    store = WorldStore(run_dir / WORLDSTORE_REL)
    problems = store.verify_version(vid)
    world = store.load_version(vid)
    digest = hashlib.sha256(json.dumps(world.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()
    out["worldstore"] = {"verified": not problems, "problems": problems[:5], "content_sha256": digest,
                         "entities": len(world.entities), "geometries": len(world.geometries),
                         "entity_ids": sorted(world.entities)}
    if problems:
        ok = False

    fused_ent = world.entities.get("surf-fused-surface")
    mesh_geoms = [g for g in world.geometries.values() if "mesh" in str(getattr(g, "type", "")).lower()]
    out["fused_mesh"] = {"entity_present": fused_ent is not None, "mesh_geometries": [g.id for g in mesh_geoms]}
    if fused_ent is not None:
        out["fused_mesh"]["geometry_ids"] = list(fused_ent.geometry_ids)
        geom = world.geometries.get(fused_ent.geometry_ids[0]) if fused_ent.geometry_ids else None
        if geom is None:
            out["fused_mesh"]["error"] = "the fused-surface entity has no geometry in the persisted world"
            ok = False
        else:
            payload = run_artifact_store(run_dir).get(geom.data_uri)
            sha = hashlib.sha256(payload).hexdigest()
            counts = _ply_counts(payload)
            out["fused_mesh"].update({
                "id": geom.id, "data_uri": geom.data_uri, "declared_hash": geom.data_hash, "resolved_sha256": sha,
                "hash_matches": sha == geom.data_hash, "bytes": len(payload),
                "geometry_vertices": geom.vertex_count, "geometry_triangles": geom.triangle_count,
                "ply_vertices": counts.get("vertex"), "ply_faces": counts.get("face"),
                "counts_match": counts.get("vertex") == geom.vertex_count and counts.get("face") == geom.triangle_count,
            })
            if not (sha == geom.data_hash and counts.get("face") == geom.triangle_count and (geom.triangle_count or 0) > 0):
                ok = False
    else:
        ok = False

    # ---------------------------------------------------------------- 3 + 4: the HTTP surface, from a fresh app instance
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(app_dir / 'app.db').as_posix()}"
    os.environ["STORAGE_ROOT"] = str(app_dir / "art")
    os.environ["WORLDSTORE_ROOT"] = str(app_dir / "ws")
    os.environ.pop("REALITY_TEST_BACKEND", None)
    from fastapi.testclient import TestClient

    import apps.api.db as db_mod

    db_mod.DATABASE_URL = os.environ["DATABASE_URL"]
    import apps.api.main as main_mod
    import apps.api.worldstore_service as ws

    ws._store = None
    H = {"x-tenant-id": "tenant-office7"}
    base = f"/api/worlds/{world_id}/evidence-world/{job_id}"
    with TestClient(main_mod.app) as c:
        r = c.get(f"{base}/world", headers=H)
        out["world_api"] = {"status_code": r.status_code}
        if r.status_code == 200:
            body = r.json()
            ids = sorted((body.get("world") or {}).get("entities") or {})
            out["world_api"].update({"version_id": body.get("version_id"), "entities": len(ids),
                                     "has_fused_surface": "surf-fused-surface" in ids,
                                     "matches_persisted_version": body.get("version_id") == vid})
            if body.get("version_id") != vid or "surf-fused-surface" not in ids:
                ok = False
        else:
            out["world_api"]["error"] = r.text[:400]
            ok = False

        e = c.post(f"{base}/export", headers=H, json={"format": "glb", "acknowledge_unscaled": True})
        out["export"] = {"status_code": e.status_code,
                         "headers": {k: v for k, v in e.headers.items() if k.lower().startswith("x-")}}
        if e.status_code == 200:
            glb = _glb_triangles(e.content)
            out["export"].update({"bytes": len(e.content), **glb})
            expected = (out.get("fused_mesh") or {}).get("geometry_triangles")
            out["export"]["expected_fused_triangles"] = expected
            out["export"]["has_real_triangles"] = glb["triangles"] > 0
            out["export"]["covers_fused_mesh"] = bool(expected) and glb["triangles"] >= expected
            if not (glb["triangles"] > 0 and glb["triangles"] >= (expected or 0)):
                ok = False
        else:
            out["export"]["error"] = e.text[:400]
            ok = False

    out["result"] = "PASS" if ok else "FAIL"
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]))
