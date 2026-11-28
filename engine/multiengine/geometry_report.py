"""Geometry measurement layer for a production reality-engine run.

Records — without fabricating — the geometry quantities the mandate names, sourced from
what a run actually persisted (registered cameras, sparse SfM points, aligned depth
records, fusion/fused mesh, and, when available, an independent reference model):

    registered cameras  | aligned depth (scale/held-out gates; SFM-gauge, not metres)
    point density       | surface: mesh validity + topology
    accuracy vs reference | chamfer / fscore / completeness (BLOCKED when no independent reference)

Honesty discipline (from the existing mesh-integrity/quality-metrics gate discipline):
* every field is either measured with its inputs recorded, or {"status":"unavailable", ...}
  / {"status":"blocked", ...} — never a default number and never a metres claim from SfM gauge
* gauge depths are reported as SFM gauge units throughout; a metres claim is allowed only when
  a validated MetricAnchor is present and scale_validity says so
* ground-truth-dependent metrics (ATE, chamfer, fscore, completeness) are recorded against the
  dataset's reference model when one is available and are explicitly labeled as that reference's
  frame, not as a survey
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.metrics.geometry import (
    camera_registration,
    chamfer,
    fscore,
    mesh_accuracy_completeness,
)
from reconstruction.quality_metrics import mesh_integrity


# -------------------------------------------------------------------------- VRAM tracking
@dataclass
class VramTracker:
    """Record peak CUDA memory for the measured window, or mark it unavailable."""

    _start_allocated: int | None = None
    _peak_allocated: int | None = None
    _start_reserved: int | None = None
    _peak_reserved: int | None = None
    cuda_available: bool = False

    def __post_init__(self) -> None:
        try:
            import torch

            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                self._start_allocated = torch.cuda.memory_allocated()
                self._start_reserved = torch.cuda.memory_reserved()
                self._peak_allocated = self._start_allocated
                self._peak_reserved = self._start_reserved
        except Exception:  # noqa: BLE001 -- absence of torch is normal on some runners
            pass

    def record(self) -> None:
        if not self.cuda_available:
            return
        try:
            import torch

            alloc = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            if alloc > (self._peak_allocated or 0):
                self._peak_allocated = alloc
            if reserved > (self._peak_reserved or 0):
                self._peak_reserved = reserved
        except Exception:
            pass

    def stop(self) -> dict[str, Any]:
        if not self.cuda_available:
            return {"cuda_available": False, "vram_peak_bytes": None, "reason": "torch.cuda not available in this environment"}
        try:
            import torch

            torch.cuda.synchronize()
            self.record()
            return {"cuda_available": True, "vram_peak_allocated_bytes": self._peak_allocated,
                    "vram_peak_reserved_bytes": self._peak_reserved,
                    "vram_peak_allocated_mb": round(self._peak_allocated / 1e6, 2),
                    "vram_peak_reserved_mb": round(self._peak_reserved / 1e6, 2)}
        except Exception as exc:
            return {"cuda_available": True, "vram_peak_bytes": None,
                    "reason": f"could not finalize vram record: {type(exc).__name__}: {exc}"}


# -------------------------------------------------------------------------- reference poses
def reference_poses_from_colmap_text(sparse_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Per-image reference camera-to-world 4x4 poses from a COLMAP text model, or ({}, reason).

    Mirrors the real-capture script's own helper; the reference is whatever the dataset ships
    (`sparse/cameras.txt` + `sparse/images.txt`), never a survey.
    """

    def _parse_quat(line_parts: list[str], off: int) -> np.ndarray:
        qw, qx, qy, qz = (float(line_parts[off + i]) for i in range(4))
        return np.array([qw, qx, qy, qz], float)

    out = {}
    path = sparse_dir / "images.txt"
    if not path.is_file():
        return {}, "no COLMAP text model at sparse/images.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10 or not parts[0].isdigit():
            continue
        name = parts[9]
        q = _parse_quat(parts, 1)
        t = np.array([float(x) for x in parts[5:8]], float)
        out[name] = {"R": q, "t": t}
        i += 1
    if not out:
        return {}, "COLMAP text model had no parseable poses"
    return out, None


def camera_to_world_pose(q: Any, t: Any) -> np.ndarray:
    """COLMAP world->camera (R from wxyz, t from model) -> camera-to-world 4x4."""
    q = np.asarray(q, float)
    t = np.asarray(t, float)
    w, x, y, z = q / (np.linalg.norm(q) or 1.0)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


# -------------------------------------------------------------------------- depth gate info
@dataclass
class DepthAlignmentRecord:
    source_id: str
    depth_observation_id: str | None
    depth_engine: str | None
    alignment: dict[str, Any]
    status: str  # "aligned" | "not_aligned" | "unreadable"
    reason: str | None = None


@dataclass
class DepthAlignmentGates:
    held_out_relative_rmse: float | None = None
    max_held_out_relative_rmse: float | None = None
    residual_rms_error: float | None = None
    scale: float | None = None
    shift: float | None = None
    scale_units: str | None = None
    scale_state: str | None = None
    is_physically_metric: bool = False
    calibration_method: str | None = None
    scale_source: str | None = None
    r_squared: float | None = None
    inlier_count: int = 0
    point_count: int = 0
    metric_anchor: str | None = None


def _depth_gate(alignment: dict[str, Any]) -> DepthAlignmentGates:
    g = DepthAlignmentGates()
    g.held_out_relative_rmse = alignment.get("held_out_relative_rmse")
    g.max_held_out_relative_rmse = alignment.get("max_held_out_relative_rmse")
    g.residual_rms_error = alignment.get("residual_rms_error")
    g.scale = alignment.get("scale")
    g.shift = alignment.get("shift")
    g.scale_units = alignment.get("units")
    g.scale_state = alignment.get("scale_state")
    g.is_physically_metric = bool(alignment.get("is_physically_metric"))
    g.calibration_method = alignment.get("calibration_method")
    g.scale_source = alignment.get("scale_source")
    g.r_squared = alignment.get("r_squared")
    g.inlier_count = int(alignment.get("inlier_count") or 0)
    g.point_count = int(alignment.get("point_count") or 0)
    g.metric_anchor = alignment.get("metric_anchor")
    return g


# -------------------------------------------------------------------------- building block
@dataclass
class MeasurementInput:
    """Everything a geometry run needs from the live run, keyed by what is observed."""

    run_id: str
    registered_cameras: Mapping[str, Any] | None = None  # source -> PinholeCamera or None if not placed
    points3d: np.ndarray | None = None  # sparse SfM points (N,3), gauge
    sources: Sequence[str] = ()
    frame_count: int = 0
    depth_alignment_records: list[DepthAlignmentRecord] = field(default_factory=list)
    sparse_scale_state: str = "SFM_GAUGE"
    sparse_scale_units: str = "sfm_units"
    sparse_is_physically_metric: bool = False
    sparse_metric_anchor: str | None = None
    fusion_report: dict[str, Any] | None = None  # depth_fusion FusionResult.report or surfaces build_surface_entities report
    mesh_vertices: np.ndarray | None = None
    mesh_faces: np.ndarray | None = None
    mesh_frame: str | None = None
    mesh_scale_state: str | None = None
    mesh_scale_units: str | None = None
    mesh_is_physically_metric: bool = False
    mesh_metric_anchor: str | None = None
    pair_rotations: list[dict[str, Any]] | None = None  # rows with a,b,R (relative_pose.R) + engine
    runtime_by_stage: dict[str, float] | None = None
    vram_tracker: VramTracker | None = None


@dataclass
class GeometryRecord:
    run_id: str
    status: str  # "measured" | "partial" (some fields unavailable) | "blocked" (no geometry source)
    units: str
    units_basis: str
    is_physically_metric: bool
    registered_cameras: dict[str, Any]
    aligned_depth: dict[str, Any]
    sparse_points: dict[str, Any]
    surface: dict[str, Any]
    accuracy_vs_reference: dict[str, Any]
    runtime: dict[str, Any]
    vram: dict[str, Any]
    measured_at: str
    notes: list[str] = field(default_factory=list)


def _block(reason: str) -> dict[str, Any]:
    return {"status": "blocked", "reason": reason}


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


@dataclass
class AccuracyReference:
    """An independent reference model for ground-truth-dependent metrics, when one exists."""
    kind: str  # "colmap_text_model"
    mesh: tuple[np.ndarray, np.ndarray] | None = None  # (V, F) in reference frame
    points: np.ndarray | None = None
    camera_poses: dict[str, np.ndarray] | None = None  # image_name -> camera-to-world 4x4
    frame_label: str = "colmap_sparse_reference"
    provenance: str = "the dataset's COLMAP text model (not an independently surveyed reference)"


# ----------------------------------------------------------------- registered cameras
def _measure_registered_cameras(inp: MeasurementInput) -> dict[str, Any]:
    cam = inp.registered_cameras or {}
    n_input = inp.frame_count if inp.frame_count else len(inp.sources)
    n_registered = len([k for k in cam if cam[k] is not None])
    out: dict[str, Any] = {"n_input": n_input, "n_registered": n_registered,
                           "registered_fraction": (n_registered / n_input if n_input else 0.0),
                           "source_ids_present": sorted(cam), "placed_cameras": [k for k in cam if cam[k] is not None]}
    ref = reference_poses_from_colmap_text(Path("datasets/south_building/sparse"))[0]
    if ref:
        def _ctow_cam(cam: Any) -> np.ndarray:
            e = cam.extrinsics
            q = e.rotation
            t = e.position
            w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
            R = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ],
                float,
            )
            P = np.eye(4)
            P[:3, :3] = R
            P[:3, 3] = np.array([t.x, t.y, t.z], float)
            return P

        est = {k: _ctow_cam(cam[k]) for k in cam if cam[k] is not None and k in ref}

        def _ctow_ref(v: dict) -> np.ndarray:
            q, t = v["R"], v["t"]
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            R = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ],
                float,
            )
            P = np.eye(4)
            P[:3, :3] = R
            P[:3, 3] = np.array([t[0], t[1], t[2]], float)
            return P

        ref_poses = {k: _ctow_ref(v) for k, v in ref.items()}
        est_poses = {}
        for k, p in est.items():
            est_poses[k] = p
        res = camera_registration(est_poses, ref_poses)

        out["reference_model"] = {"kind": "colmap_text_model",
                                  "frame": "colmap_sparse_reference",
                                  "provenance": "the dataset's COLMAP text model (sparse/), used as the only available reference for camera placement — not a survey"}
        if res["status"] == "OK":
            out["reference"] = {"status": "measured", "registered_cameras": res}
        else:
            out["reference"] = {"status": "blocked", "reason": res.get("reason", "reference comparison could not be computed")}
    else:
        out["reference"] = {"status": "unavailable", "reason": "no reference model available for camera-placement comparison"}
    # pose-graph rotation-cycle consistency (deduced from pair rotations if present), rotation-only label
    if inp.pair_rotations:
        from reconstruction.quality_metrics import pose_graph_cycle_error

        nodes: list[str] = []
        edges: list[tuple[str, str, np.ndarray]] = []
        seen = set()
        for row in inp.pair_rotations:
            a, b = row.get("a"), row.get("b")
            R = row.get("relative_pose") and np.asarray(row["relative_pose"]["R"], float)
            if a and b and R is not None and a not in seen and b not in seen:
                nodes.extend([a, b])
                seen.add(a)
                seen.add(b)
                edges.append((a, b, R))
        if nodes and edges:
            try:
                pe = pose_graph_cycle_error(nodes, edges)
                out["pose_graph_cycle_residual_residual_deg"] = {"status": "measured" if pe.get("metric") else "unavailable",
                                                                "value": pe.get("value"), "mean_deg": pe.get("mean_deg"),
                                                                "p95_deg": pe.get("p95_deg"), "max_deg": pe.get("max_deg"),
                                                                "n_triples": pe.get("n_triples"),
                                                                "note": "rotation-only: no translation cycle (scale not determined here)"}
            except Exception as exc:
                out["pose_graph_cycle_residual_residual_deg"] = {"status": "unavailable", "reason": f"cycle residual could not be computed: {type(exc).__name__}: {exc}"}
        else:
            out["pose_graph_cycle_residual_residual_deg"] = {"status": "unavailable", "reason": "no measured pair rotations among the inputs"}
    else:
        out["pose_graph_cycle_residual_residual_deg"] = {"status": "unavailable", "reason": "no pair-rotation data supplied"}
    return out


# ----------------------------------------------------------------- aligned depth
def _measure_aligned_depth(inp: MeasurementInput) -> dict[str, Any]:
    recs = inp.depth_alignment_records
    n_offered = len(recs)
    n_aligned = sum(1 for r in recs if r.status == "aligned")
    out: dict[str, Any] = {"n_offered": n_offered, "n_aligned": n_aligned,
                           "aligned_fraction": (n_aligned / n_offered if n_offered else 0.0),
                           "records": [r.alignment for r in recs if r.status == "aligned"]}
    per_rec = []
    for r in recs:
        al = r.alignment if r.status == "aligned" else {}
        per_rec.append({"source": r.source_id, "depth_engine": r.depth_engine,
                        "status": r.status, "reason": r.reason, "alignment": al})
    out["per_record"] = per_rec
    if n_aligned:
        g = _depth_gate(al := next((r.alignment for r in recs if r.status == "aligned"), {}))
        out["gauge_units"] = g.scale_units or inp.sparse_scale_units
        out["gauge_state"] = g.scale_state or inp.sparse_scale_state
        out["is_physically_metric"] = g.is_physically_metric or inp.sparse_is_physically_metric
        out["held_out_relative_rmse"] = g.held_out_relative_rmse
        out["max_held_out_relative_rmse_applied"] = g.max_held_out_relative_rmse
        out["residual_rms_error_gauge"] = g.residual_rms_error
        out["scale"] = g.scale
        out["shift"] = g.shift
        out["r_squared"] = g.r_squared
        out["inlier_count"] = g.inlier_count
        out["point_count"] = g.point_count
        out["calibration_method"] = g.calibration_method
        out["scale_source"] = g.scale_source
        out["metric_anchor"] = g.metric_anchor
        out["metric_anchor_validated_for_metres"] = (
            bool(g.metric_anchor) and "validated" in str(g.metric_anchor).lower()
        ) or (bool(inp.sparse_metric_anchor) and "validated" in str(inp.sparse_metric_anchor).lower())
        # The depth record's scale_units is the canonical units field too.
        out["units"] = g.scale_units or inp.sparse_scale_units
        out["note"] = ("relative depth is aligned to the job's sparse SfM points; lengths here are in SfM gauge "
                       "units, NOT metres, unless an independently surveyed MetricAnchor is recorded and scale_validity permits a metres claim")
    else:
        out["gauge_units"] = inp.sparse_scale_units
        out["gauge_state"] = inp.sparse_scale_state
        out["is_physically_metric"] = inp.sparse_is_physically_metric
        out["units"] = inp.sparse_scale_units
        out["held_out_relative_rmse"] = None
        out["reason"] = "no depth record aligned successfully"
    # cross-view depth consistency, when a pair of aligned depth maps + their cameras + relative pose are present
    if inp.fusion_report and inp.registered_cameras and n_aligned:
        try:
            out["depth_consistency"] = {"status": "unavailable", "reason": "depth consistency requires paired aligned depth maps + relative pose in the same frame; not propagated to the stage yet"}
        except Exception:
            out["depth_consistency"] = {"status": "unavailable", "reason": "depth consistency not computed"}
    else:
        out["depth_consistency"] = {"status": "unavailable", "reason": "no fusion report or no aligned depth pairs"}
    return out


# ----------------------------------------------------------------- sparse points
def _measure_sparse_points(inp: MeasurementInput) -> dict[str, Any]:
    pts = inp.points3d
    if pts is None or len(pts) == 0:
        return {"status": "unavailable", "reason": "no sparse SfM points in the run inputs"}
    n = len(pts)
    bounds = [np.min(pts, axis=0).tolist(), np.max(pts, axis=0).tolist()]
    extent = (np.max(pts, axis=0) - np.min(pts, axis=0)).tolist()
    return {"status": "measured", "n_points": n, "bounds": bounds, "extent_gauge": extent,
            "scale_state": inp.sparse_scale_state, "scale_units": inp.sparse_scale_units,
            "is_physically_metric": inp.sparse_is_physically_metric, "metric_anchor": inp.sparse_metric_anchor,
            "note": "sparse points are in the SfM gauge, NOT metres, unless anchored"}


# ----------------------------------------------------------------- surface
def _measure_surface(inp: MeasurementInput) -> dict[str, Any]:
    from world_ir.geometry_data import MeshData

    f = inp.fusion_report or {}
    verts = inp.mesh_vertices
    faces = inp.mesh_faces
    out: dict[str, Any] = {"mesh": {"status": "unavailable", "reason": "no fused mesh available in the run"} if (verts is None or faces is None or len(faces) == 0) else {"status": "measured"}}
    if verts is not None and faces is not None and len(faces) > 0:
        mesh = MeshData.from_arrays(verts, faces)
        mi = mesh_integrity(mesh)
        out["mesh"] = {"status": "measured", "vertices": len(verts), "triangles": len(faces),
                       "frame": inp.mesh_frame, "scale_state": inp.mesh_scale_state or f.get("scale_state"),
                       "scale_units": inp.mesh_scale_units or f.get("scale_units"),
                       "is_physically_metric": inp.mesh_is_physically_metric or bool(f.get("physically_metric")),
                       "metric_anchor": inp.mesh_metric_anchor,
                       **mi, "note": "mesh integrity is structural only: a clean mesh can still be geometrically wrong"}
        out["surface_area"] = {"status": "unavailable", "reason": "surface area is mesh-frame/units dependent; reported in mesh_integrity when units are presented"}
        out["surface_area_gauge2"] = f.get("surface_area_gauge2") if f else None
        out["bounds"] = {"status": "unavailable", "reason": "bounds are in the mesh frame/units; present in mesh_integrity"}
        if f:
            out["bounds_gauge"] = f.get("bounds_gauge")
    else:
        out["surface_area"] = {"status": "unavailable", "reason": "no mesh"}
        out["bounds_gauge"] = None
    # point density: from fused depth valid pixels / volume if available
    if f and "volume_side_gauge" in f and "views_with_valid_depth" in f:
        side = float(f["volume_side_gauge"])
        vol = side ** 3 if side > 0 else 0.0
        n_valid = sum(r.get("depth_pixels", 0) for r in f.get("per_view", []) or [])
        out["point_density"] = {"status": "measured", "valid_depth_pixels": n_valid, "volume_side_gauge": side,
                                "volume_gauge3": vol, "points_per_gauge_unit3": (n_valid / vol if vol > 0 else None),
                                "note": "density is reported in SfM gauge units per voxel volume; it is NOT 'points per m^3' unless an independently surveyed anchor makes the frame metric"}
    else:
        out["point_density"] = {"status": "unavailable", "reason": "no volume bounds for density estimate"}
    return out


# ----------------------------------------------------------------- accuracy vs reference
def _measure_accuracy_vs_reference(inp: MeasurementInput, ref: AccuracyReference | None) -> dict[str, Any]:
    if ref is None:
        return {"status": "blocked", "reason": "no independent reference model supplied; accuracy/completeness metrics cannot be computed without fabricating one"}
    out: dict[str, Any] = {"reference": {"kind": ref.kind, "frame": ref.frame_label, "provenance": ref.provenance}}
    # mesh accuracy/completeness requires a reference MESH
    if ref.mesh is not None and inp.mesh_vertices is not None and inp.mesh_faces is not None:
        try:
            m = mesh_accuracy_completeness((inp.mesh_vertices, inp.mesh_faces), ref.mesh,
                                           n_samples=min(4000, int((inp.mesh_faces.size or 0) * 0.1) or 2000))
            out["mesh"] = {"status": "measured", **m, "note": "accuracy and completeness are measured against the reference's surface; they describe agreement with THAT reference, not absolute metric truth"}
        except Exception as exc:
            out["mesh"] = {"status": "blocked", "reason": f"mesh accuracy could not be computed: {type(exc).__name__}: {exc}"}
    else:
        out["mesh"] = {"status": "unavailable", "reason": "either the reference has no mesh, or the run has no mesh"}
    # chamfer / fscore require a reference POINT CLOUD (in the same frame/units as the run mesh)
    if ref.points is not None and inp.mesh_vertices is not None:
        pred_pts = np.asarray(inp.mesh_vertices, float)
        ref_pts = np.asarray(ref.points, float)
        try:
            ch = chamfer(pred_pts, ref_pts)
            fs = fscore(pred_pts, ref_pts)
            out["chamfer_and_fscore"] = {"status": "measured", "chamfer_m": ch["chamfer_m"], "accuracy_m": ch["accuracy_m"],
                                          "completeness_m": ch["completeness_m"], "fscore": fs, "note": "measured against the reference point cloud; that cloud is in the reference frame, NOT necessarily a surveyed frame"}
        except Exception as exc:
            out["chamfer_and_fscore"] = {"status": "blocked", "reason": f"chamfer/fscore could not be computed: {type(exc).__name__}: {exc}"}
    else:
        out["chamfer_and_fscore"] = {"status": "unavailable", "reason": "no reference point cloud or no run mesh for chamfer/fscore"}
    # camera placement accuracy: already in registered_cameras.reference
    return out


# ----------------------------------------------------------------- build
def build_geometry_record(inp: MeasurementInput, ref: AccuracyReference | None = None) -> GeometryRecord:
    """Assemble one GeometryRecord from the run's observed inputs. No input is invented."""
    cam = _measure_registered_cameras(inp)
    depth = _measure_aligned_depth(inp)
    sp = _measure_sparse_points(inp)
    surf = _measure_surface(inp)
    acc = _measure_accuracy_vs_reference(inp, ref)
    rt = dict(inp.runtime_by_stage or {})
    rt["total_s"] = round(sum(rt.values()), 2) if rt else 0.0
    vr = inp.vram_tracker.stop() if inp.vram_tracker else {"cuda_available": False, "vram_peak_bytes": None, "reason": "no vram tracker supplied"}
    parts: list[str] = []
    for name, val in (("registered_cameras", cam), ("aligned_depth", depth), ("sparse_points", sp),
                      ("surface", surf), ("accuracy_vs_reference", acc), ("runtime", rt), ("vram", vr)):
        st = val.get("status") if isinstance(val, dict) else None
        if st == "unavailable":
            parts.append(f"{name}: unavailable ({val.get('reason','')})")
        elif st == "blocked":
            parts.append(f"{name}: blocked ({val.get('reason','')})")
    is_metric = bool(sp.get("is_physically_metric") or depth.get("is_physically_metric") or
                     (surf.get("mesh") or {}).get("is_physically_metric"))
    return GeometryRecord(run_id=inp.run_id, status="measured" if not parts else "partial" if not any(v.get("status")=="blocked" for v in (cam,depth,sp,surf,acc,rt,vr) if isinstance(v,dict)) else "blocked",
                          units=depth.get("gauge_units") or "sfm_units", units_basis="SfM gauge (COLMAP_SFM_UNITS); not metres without an independently surveyed anchor",
                          is_physically_metric=is_metric, registered_cameras=cam, aligned_depth=depth,
                          sparse_points=sp, surface=surf, accuracy_vs_reference=acc, runtime=rt, vram=vr,
                          measured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), notes=parts)


def write_geometry_record(root: Path, record: GeometryRecord) -> dict[str, Any]:
    payload_dict = json.loads(json.dumps(record.__dict__, default=_json_default))
    payload_bytes = json.dumps(payload_dict, indent=2, sort_keys=True).encode("utf-8")
    uri, sha = put_geometry_artifact(root, payload_bytes)
    path = root / "geometry_report.json"
    path.write_bytes(payload_bytes)
    return {"file": "geometry_report.json", "uri": uri, "sha256": sha, "bytes": len(payload_bytes)}


def put_geometry_artifact(root: Path, data: bytes) -> tuple[str, str]:
    from engine.multiengine.evidence_matrix import EvidenceMatrix

    m = EvidenceMatrix(root)
    return m.artifacts.put(data)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if isinstance(o, np.floating) else int(o)
    raise TypeError(f"cannot serialize {type(o)} to JSON in geometry_report")
