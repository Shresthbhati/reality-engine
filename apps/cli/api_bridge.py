#!/usr/bin/env python3
"""Reality Engine Backend API Bridge.

A thin CLI script called by Next.js API routes via child_process.
Outputs NDJSON to stdout. Never crashes silently — always outputs a
valid JSON line with {"error": ...} on failure.

Usage:
    python apps/cli/api_bridge.py status
    python apps/cli/api_bridge.py list-worlds
    python apps/cli/api_bridge.py load-world <version_id>
    python apps/cli/api_bridge.py list-sessions

Environment:
    REALITY_STORE_PATH  — path to WorldStore root (default: ~/.reality_engine/store)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure repo root is always at the front of sys.path, avoiding site-packages shadowing
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _store_path() -> Path:
    env = os.environ.get("REALITY_STORE_PATH")
    if env:
        return Path(env)
    candidates = [
        Path.cwd() / "store",
        Path(__file__).resolve().parent.parent.parent / "store",
        Path.home() / ".reality_engine" / "store",
    ]
    for c in candidates:
        if (c / "versions").exists():
            return c
    for c in candidates:
        if c.exists():
            return c
    return Path.home() / ".reality_engine" / "store"


def _emit(obj: dict) -> None:
    print(json.dumps(obj, default=str), flush=True)


def cmd_status() -> int:
    store_root = _store_path()
    try:
        from worldstore import WorldStore
        store = WorldStore(store_root)
        versions = store.list_versions()
        _emit({
            "backend": True,
            "store_path": str(store_root),
            "version_count": len(versions),
            "latest_version": versions[-1].version_id if versions else None,
            "python_version": sys.version,
        })
        return 0
    except Exception as exc:
        _emit({
            "backend": False,
            "store_path": str(store_root),
            "error": str(exc),
            "python_version": sys.version,
        })
        return 1


def cmd_list_worlds() -> int:
    store_root = _store_path()
    try:
        from worldstore import WorldStore
        store = WorldStore(store_root)
        versions = store.list_versions()
        result = []
        for sv in versions:
            entry = {
                "version_id": sv.version_id,
                "world_id": sv.world_id,
                "parent": sv.parent,
                "artifact_uri": sv.artifact_uri,
            }
            try:
                try:
                    world = store.load_version(sv.version_id)
                except Exception:
                    from worldstore.tiles import load_version_partitioned
                    world = load_version_partitioned(store, sv.version_id)
                entry["entity_count"] = len(world.entities)
                entry["global_provenance"] = world.global_provenance.value
                entry["global_confidence"] = world.global_confidence
                entry["coordinate_frame"] = world.coordinate_frame
                entry["name"] = getattr(world, "name", str(sv.world_id))
                entry["schema_version"] = getattr(world, "schema_version", "v1")
                entry["created_at"] = getattr(world, "created_at", None)
                entry["modified_at"] = getattr(world, "modified_at", None)
            except Exception as load_err:
                entry["load_error"] = str(load_err)
                entry["entity_count"] = 0
            result.append(entry)
        _emit({"worlds": result, "count": len(result)})
        return 0
    except Exception as exc:
        _emit({"error": str(exc), "worlds": [], "count": 0})
        return 1


def cmd_load_world(version_id: str) -> int:
    store_root = _store_path()
    try:
        from worldstore import WorldStore
        store = WorldStore(store_root)
        if version_id == "latest":
            versions = store.list_versions()
            if not versions:
                _emit({"error": "No versions stored in WorldStore", "version_id": "latest", "entities": []})
                return 1
            version_id = versions[-1].version_id

        try:
            world = store.load_version(version_id)
        except Exception:
            from worldstore.tiles import load_version_partitioned
            world = load_version_partitioned(store, version_id)

        entities = []
        for e in world.entities.values():
            entity_dict = {
                "id": e.id,
                "type": e.type.value if hasattr(e.type, "value") else str(e.type),
                "provenance": e.provenance.value if hasattr(e.provenance, "value") else str(e.provenance),
                "confidence": e.confidence,
                "geometry_ids": list(getattr(e, "geometry_ids", [])),
                "relationships": [
                    {
                        "kind": r.kind.value if hasattr(r.kind, "value") else str(r.kind),
                        "target_id": r.target_id,
                        "confidence": getattr(r, "confidence", 1.0),
                    }
                    for r in getattr(e, "relationships", [])
                ],
                "observations": [
                    {
                        "id": getattr(o, "id", ""),
                        "sensor_type": getattr(o, "sensor_type", ""),
                        "timestamp": getattr(o, "timestamp", 0.0),
                        "frame_id": getattr(o, "frame_id", ""),
                        "data_uri": getattr(o, "data_uri", ""),
                        "confidence": getattr(o, "confidence", 1.0),
                    }
                    for o in getattr(e, "observations", [])
                ],
                "custom_properties": getattr(e, "custom_properties", {}),
                "tags": list(getattr(e, "tags", [])),
            }
            entities.append(entity_dict)

        geometries = {}
        for gid, g in getattr(world, "geometries", {}).items():
            geometries[gid] = {
                "id": g.id,
                "type": g.type.value if hasattr(g.type, "value") else str(g.type),
                "lod_level": getattr(g, "lod_level", 0),
                "vertex_count": getattr(g, "vertex_count", None),
                "triangle_count": getattr(g, "triangle_count", None),
                "data_uri": getattr(g, "data_uri", ""),
                "data_hash": getattr(g, "data_hash", ""),
                "bounds_min": g.bounds_min.to_dict() if getattr(g, "bounds_min", None) else None,
                "bounds_max": g.bounds_max.to_dict() if getattr(g, "bounds_max", None) else None,
                "provenance": g.provenance.value if hasattr(g.provenance, "value") else str(g.provenance),
                "confidence": getattr(g, "confidence", 1.0),
            }

        _emit({
            "version_id": version_id,
            "world_id": world.id,
            "name": getattr(world, "name", str(world.id)),
            "schema_version": getattr(world, "schema_version", "v1"),
            "global_provenance": world.global_provenance.value,
            "global_confidence": world.global_confidence,
            "coordinate_frame": world.coordinate_frame,
            "entity_count": len(entities),
            "entities": entities,
            "geometry_count": len(world.geometries),
            "geometries": geometries,
        })
        return 0
    except Exception as exc:
        _emit({"error": str(exc), "version_id": version_id, "entities": []})
        return 1


def cmd_points_path(version_id: str) -> int:
    store_root = _store_path()
    try:
        from worldstore import WorldStore
        store = WorldStore(store_root)
        if version_id == "latest":
            versions = store.list_versions()
            if not versions:
                _emit({"found": False, "error": "No versions stored"})
                return 1
            version_id = versions[-1].version_id

        world = store.load_version(version_id)
        candidates = []
        for gid, g in getattr(world, "geometries", {}).items():
            gtype = g.type.value if hasattr(g.type, "value") else str(g.type)
            if "point_cloud" in gtype.lower() or getattr(g, "data_uri", "").endswith(".ply"):
                if g.data_uri.startswith("artifact://"):
                    digest = g.data_uri[len("artifact://"):]
                    candidates.append(store_root / "artifacts" / digest)
                    candidates.append(store_root / "artifacts" / f"{digest}.ply")
                elif g.data_uri:
                    candidates.append(Path(g.data_uri))

        candidates.extend([
            store_root / "artifacts" / "points.ply",
            store_root.parent / "points.ply",
            store_root.parent / "pipeline_out" / "points.ply",
            store_root.parent / "datasets" / "room_capture" / "pipeline_out" / "points.ply",
        ])

        for c in candidates:
            if c.exists() and c.is_file() and c.stat().st_size > 0:
                _emit({
                    "found": True,
                    "version_id": version_id,
                    "path": str(c.resolve()),
                    "size_bytes": c.stat().st_size,
                })
                return 0

        _emit({
            "found": False,
            "version_id": version_id,
            "error": f"No point cloud artifact found for version {version_id}",
        })
        return 0
    except Exception as exc:
        _emit({"found": False, "version_id": version_id, "error": str(exc)})
        return 1


def cmd_list_sessions() -> int:
    """List evidence sessions from the canonical SessionWorkspace
    (evidence/session_store.py: <root>/sessions/<id>/session.json +
    sources.json). The previous implementation globbed *.json files in
    a directory that the canonical store never writes, so real
    sessions were always reported as absent."""
    store_root = _store_path()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from evidence.session_store import SessionWorkspace

        workspace = SessionWorkspace(str(store_root.parent))
        sessions = []
        for entry in workspace.list_sessions():
            sid = str(entry.get("id", ""))
            detail: dict = {
                "id": sid,
                "name": entry.get("name") or sid,
                "status": entry.get("status", ""),
                "evidence_count": entry.get("evidence_count", 0),
                "source_count": entry.get("source_count", 0),
            }
            if entry.get("error"):
                detail["error"] = entry["error"]
            # Per-source composition from sources.json (real file lists,
            # never invented): the UI's session browser shows what the
            # capture actually contains.
            sources_path = (
                store_root.parent / "sessions" / sid / "sources.json"
            )
            if sources_path.is_file():
                try:
                    sdata = json.loads(sources_path.read_text())
                    types: dict = {}
                    for s in sdata.get("sources", {}).values():
                        t = str(s.get("source_type", s.get("type", "other")))
                        types[t] = types.get(t, 0) + 1
                    detail["source_types"] = types
                except (json.JSONDecodeError, OSError):
                    pass
            sessions.append(detail)
        _emit({"sessions": sessions, "count": len(sessions)})
        return 0
    except Exception as exc:
        _emit({"error": str(exc), "sessions": [], "count": 0})
        return 1


def cmd_run_log() -> int:
    """Serve the newest measured reconstruction run record
    (datasets/*/runs/*.json) as a log timeline for the UI's Log
    Stream. Every entry is derived from the recorded run (stages,
    durations, outcomes, refusal reasons); when no record exists the
    answer says so -- nothing is synthesized."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dirs = [repo_root / "datasets"]
    best_m, best_path = None, None
    for base in run_dirs:
        if not base.is_dir():
            continue
        for p in base.glob("*/runs/*.json"):
            name = p.name
            if not name.startswith("run_"):
                continue
            if best_m is None or name > best_m:
                best_m, best_path = name, p
    if best_path is None:
        _emit({
            "available": False,
            "reason": "no reconstruction run record found under datasets/*/runs/ -- "
                      "run scripts/run_south_building_e2e.py to produce one",
        })
        return 0
    try:
        data = json.loads(best_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _emit({"available": False, "reason": f"run record unreadable: {exc}"})
        return 1

    entries = []
    recon = data.get("reconstruction", {})
    if recon:
        entries.append({
            "level": "INFO",
            "timestamp": best_m.replace("run_", "").replace("_gpu", "").replace(".json", "") if best_m else "",
            "module": recon.get("backend", "ReconstructionBackend"),
            "message": (
                f"status={recon.get('status')} poses={recon.get('poses')} "
                f"points={recon.get('points')} duration={recon.get('duration_s')}s"
            ),
        })
        mr = recon.get("merge_report")
        if mr:
            for name, status in (mr.get("status_by_session") or {}).items():
                reason = (mr.get("reasons") or {}).get(name, "")
                lvl = "INFO" if status in ("aligned", "reference") else "WARNING"
                entries.append({
                    "level": lvl, "timestamp": "", "module": "SubModelMerge",
                    "message": f"sub-model {name}: {status}" + (f" -- {reason}" if reason else ""),
                })
    outcome = data.get("outcome", {})
    if outcome:
        entries.append({
            "level": "INFO" if outcome.get("outcome") == "accepted" else "WARNING",
            "timestamp": "", "module": "RunClassification",
            "message": (
                f"outcome={outcome.get('outcome')} "
                f"reason={outcome.get('reason', '')}"
            ),
        })
    gt = data.get("gt_comparison", {})
    if gt and not gt.get("refused"):
        for o in gt.get("outliers", []):
            entries.append({
                "level": "WARNING", "timestamp": "", "module": "PoseEvaluation",
                "message": (
                    f"outlier camera {o.get('camera')}: rotation "
                    f"{o.get('rotation_deg')} deg, center {o.get('center_error')}"
                ),
            })
        entries.append({
            "level": "INFO", "timestamp": "", "module": "PoseEvaluation",
            "message": (
                f"pose fidelity median={gt.get('rotation_deg_median')} deg "
                f"over {gt.get('common_cameras')} cameras"
            ),
        })
    pf = data.get("point_fidelity", {})
    if pf and not pf.get("refused"):
        entries.append({
            "level": "INFO", "timestamp": "", "module": "PointFidelity",
            "message": (
                f"point->reference median={pf.get('dist_median')} "
                f"p90={pf.get('dist_p90')} over {pf.get('measured_points')} pts"
            ),
        })
    _emit({
        "available": bool(entries),
        "source": str(best_path.relative_to(repo_root)) if entries else None,
        "reason": None if entries else "run record exists but has no recognized stages",
        "entries": entries,
    })
    return 0


def cmd_diff(base_vid: str, head_vid: str) -> int:
    store_root = _store_path()
    try:
        from worldstore import WorldStore
        from world_ir.diff import diff_worlds

        store = WorldStore(store_root)
        versions = store.list_versions()
        if not versions:
            _emit({
                "error": "No versions stored in WorldStore to diff",
                "from_version_id": base_vid,
                "to_version_id": head_vid,
            })
            return 1

        vid_map = {v.version_id: v for v in versions}
        actual_base = versions[-1].version_id if base_vid == "latest" else base_vid
        actual_head = versions[-1].version_id if head_vid == "latest" else head_vid

        if actual_base not in vid_map:
            _emit({
                "error": f"Base version not found: {base_vid}",
                "from_version_id": base_vid,
                "to_version_id": head_vid,
            })
            return 1

        if actual_head not in vid_map:
            _emit({
                "error": f"Head version not found: {head_vid}",
                "from_version_id": base_vid,
                "to_version_id": head_vid,
            })
            return 1

        try:
            base_world = store.load_version(actual_base)
        except Exception:
            from worldstore.tiles import load_version_partitioned
            base_world = load_version_partitioned(store, actual_base)

        try:
            head_world = store.load_version(actual_head)
        except Exception:
            from worldstore.tiles import load_version_partitioned
            head_world = load_version_partitioned(store, actual_head)

        diff = diff_worlds(base_world, head_world)
        res = diff.to_dict()
        res["from_version_id"] = actual_base
        res["to_version_id"] = actual_head
        res["entities"] = res.get("entity_diffs", [])
        res["geometries"] = res.get("geometry_diffs", [])
        _emit(res)
        return 0
    except Exception as exc:
        _emit({
            "error": str(exc),
            "from_version_id": base_vid,
            "to_version_id": head_vid,
        })
        return 1


def main() -> int:
    if len(sys.argv) < 2:
        _emit({"error": "Usage: api_bridge.py <command> [args...]"})
        return 1

    cmd = sys.argv[1]
    if cmd == "status":
        return cmd_status()
    elif cmd == "list-worlds":
        return cmd_list_worlds()
    elif cmd == "load-world":
        if len(sys.argv) < 3:
            _emit({"error": "load-world requires <version_id>"})
            return 1
        return cmd_load_world(sys.argv[2])
    elif cmd == "diff":
        if len(sys.argv) < 4:
            _emit({"error": "diff requires <base_version_id> <head_version_id>"})
            return 1
        return cmd_diff(sys.argv[2], sys.argv[3])
    elif cmd == "points-path":
        if len(sys.argv) < 3:
            _emit({"error": "points-path requires <version_id>"})
            return 1
        return cmd_points_path(sys.argv[2])
    elif cmd == "list-sessions":
        return cmd_list_sessions()
    elif cmd == "run-log":
        return cmd_run_log()
    else:
        _emit({"error": f"Unknown command: {cmd}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
