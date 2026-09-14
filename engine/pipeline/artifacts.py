"""Pipeline artifact writers (one owner of the on-disk output contract).

`reality compile` and `scripts/run_vertical_slice.py` write the same
artifact set; this module is the single place that defines what those
files are.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional, Sequence


def write_points_ply(path: Path, points: Sequence) -> None:
    """Minimal binary PLY (float32 xyz): every real point preserved."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    ).encode("ascii")
    body = bytearray()
    for p in points:
        body += struct.pack(
            "<3f", float(p[0]), float(p[1]), float(p[2])
        )
    path.write_bytes(header + bytes(body))


def write_mesh_ply_from_artifact(
    out_dir: Path,
    artifact_store,
    mesh_facts: Optional[dict],
) -> Optional[Path]:
    """Write `mesh.ply` from the pipeline's MESH artifact when the stage
    ran. Returns the path, or None when there is no mesh (honest
    absence -- no empty file)."""
    if not mesh_facts or mesh_facts.get("status") != "ran":
        return None
    uri = mesh_facts.get("artifact_uri")
    if not uri:
        return None
    from reconstruction.meshing.mesh import MeshData

    mesh = MeshData.from_bytes(artifact_store.get(uri))
    path = out_dir / "mesh.ply"
    path.write_bytes(mesh.to_ply_bytes())
    return path


def write_cameras_json(
    path: Path,
    camera_poses,
    scale_state: str,
    image_size: Sequence[int],
) -> None:
    path.write_text(json.dumps({
        "frame": "world (meters, +Y up after frame canonicalization)",
        "scale_state": scale_state,
        "rotation_convention": "camera-to-world quaternion (w, x, y, z)",
        "image_size": list(image_size),
        "cameras": [
            {
                "evidence_id": eid,
                "position_m": list(pos),
                "rotation_wxyz": list(rot),
            }
            for eid, pos, rot in camera_poses
        ],
    }, indent=2))


def write_exports(world_dict: dict, artifact_store, out_dir: Path) -> Path:
    """glTF export of the compiled world (real TRIANGLES/POINTS
    primitives where artifacts resolve, placeholder cubes otherwise)."""
    from exporters.gltf.exporter import export_to_gltf
    from world_ir.world_v1 import WorldIR

    world = WorldIR.from_dict(world_dict)
    gltf = export_to_gltf(world, artifact_store=artifact_store)
    exports = out_dir / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    path = exports / "scene.gltf"
    path.write_text(json.dumps(gltf))
    return path
