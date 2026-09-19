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


def _store_path() -> Path:
    env = os.environ.get("REALITY_STORE_PATH")
    if env:
        return Path(env)
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
            # Try to load entity count from world without full load
            try:
                world = store.load_version(sv.version_id)
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
        world = store.load_version(version_id)

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
        })
        return 0
    except Exception as exc:
        _emit({"error": str(exc), "version_id": version_id, "entities": []})
        return 1


def cmd_list_sessions() -> int:
    store_root = _store_path()
    try:
        # Try to find session data in store vicinity
        # Sessions may be stored as evidence packages
        sessions_dir = store_root.parent / "sessions"
        sessions = []
        if sessions_dir.exists():
            for p in sorted(sessions_dir.glob("*.json")):
                try:
                    data = json.loads(p.read_text())
                    sessions.append(data)
                except Exception:
                    sessions.append({"id": p.stem, "error": "parse_error"})

        # Also try to find session data from worldstore parent
        _emit({"sessions": sessions, "count": len(sessions)})
        return 0
    except Exception as exc:
        _emit({"error": str(exc), "sessions": [], "count": 0})
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
    elif cmd == "list-sessions":
        return cmd_list_sessions()
    else:
        _emit({"error": f"Unknown command: {cmd}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
