"""P110 stage `quality_report`: geometry validity of the reconstructed mesh artifact.

``mesh_validity`` measures a triangle mesh with ``trimesh`` (optional, the ``meshing`` extra). The stage reads the mesh
the ladder's mesh stage put into the world (``ctx.state["ladder"].prog.world``: the MESH geometry's ``data_uri``) and
resolves its bytes through ``ctx.state["artifact_store"]`` (host-supplied, like the segments). It records the numbers
under ``metrics`` with a ``mesh_`` prefix. Anything it cannot measure -- no ladder/mesh, no store, an unreadable
artifact, no ``trimesh`` -- is ``mesh_verdict = "NOT_EVALUABLE"`` and ``skipped``; never a success-shaped PASS.
A room shell open at doorways is expected to be ``WARN``; only degenerate faces or several components are ``FAIL``
(graded ``degraded`` -- the stage never raises, so a required stage cannot fail the job on a mesh it merely judged).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec

_NOT_EVALUABLE = "NOT_EVALUABLE"


def mesh_validity(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float | str]:
    try:
        import trimesh
    except ImportError:
        return {"verdict": _NOT_EVALUABLE, "reason": "trimesh not installed"}
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(faces) == 0:
        return {"verdict": _NOT_EVALUABLE, "reason": "mesh has no faces"}
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)  # process=True would silently repair it
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    degenerate = int((~mesh.nondegenerate_faces()).sum())
    components = len(trimesh.graph.connected_components(mesh.edges_sorted, nodes=np.unique(faces)))
    watertight = bool(mesh.is_watertight)
    verdict = "FAIL" if degenerate > 0 or components > 1 else ("PASS" if watertight else "WARN")
    return {"watertight": 1.0 if watertight else 0.0, "non_manifold_edges": int((counts > 2).sum()),
            "degenerate_faces": degenerate, "components": components, "verdict": verdict}


def _load_mesh(ctx: StageContext) -> tuple[np.ndarray, np.ndarray] | str:
    """The mesh arrays, or the reason there is nothing to measure."""
    ladder = ctx.state.get("ladder")
    world = getattr(getattr(ladder, "prog", None), "world", None)
    if world is None:
        return "no reconstructed world (the multi-view ladder rung did not run)"
    geom = next((g for g in world.geometries.values()
                 if getattr(getattr(g, "type", None), "value", None) == "mesh" and getattr(g, "data_uri", None)), None)
    if geom is None:
        return "no mesh artifact in the world"
    store = ctx.state.get("artifact_store")
    if store is None:
        return "no artifact store supplied; the mesh artifact cannot be read"
    try:
        from world_ir.geometry_data import MeshData

        mesh = MeshData.from_bytes(store.get(geom.data_uri))
    except Exception as exc:  # noqa: BLE001 -- unreadable artifact: report it, do not fail the job
        return f"mesh artifact unreadable: {type(exc).__name__}: {exc}"
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)


class QualityReport:
    spec = StageSpec(name="quality_report", version="1", resource_class="cpu", required=True)

    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(timezone.utc)
        loaded = _load_mesh(ctx)
        if isinstance(loaded, str):
            report: dict[str, float | str] = {"verdict": _NOT_EVALUABLE, "reason": loaded}
        else:
            report = mesh_validity(*loaded)
        verdict = report["verdict"]
        # metrics is typed dict[str, float] but is a JSON column; the card puts the verdict string in it
        metrics: dict = {f"mesh_{k}": v for k, v in report.items() if k != "reason"}
        if verdict == _NOT_EVALUABLE:
            status, warnings = "skipped", [f"mesh not evaluable: {report.get('reason')}"]
        elif verdict == "FAIL":
            status, warnings = "degraded", [f"mesh FAIL: {report['degenerate_faces']} degenerate faces, "
                                            f"{report['components']} components"]
        else:
            status, warnings = "succeeded", ([] if verdict == "PASS" else ["mesh is not watertight (expected for an open room shell)"])
        return StageResult(
            name=self.spec.name, status=status, inputs_hash="", outputs={}, metrics=metrics, warnings=warnings,
            error=None, started_at=started, ended_at=datetime.now(timezone.utc), peak_rss_mb=None,
        )
