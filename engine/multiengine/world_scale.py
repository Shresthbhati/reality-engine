"""Apply a validated scale calibration to a persisted world as a NEW immutable version.

``scale_anchors`` decides whether a physical scale is supported; this module is the only place that applies one. It transforms every
geometric quantity of the world consistently (entity positions, box / plane / mesh geometry, plane equations and hulls, relationship
distances, camera centres), propagates the scale's own uncertainty into position and dimension uncertainty, records the calibration and
its evidence in the world, and refuses -- loudly -- instead of half-scaling:

  * a calibration that is not applicable (insufficient / failed)          -> ``CalibrationRefused``
  * a world that already carries a calibration (double scaling)           -> ``AlreadyCalibrated`` (recalibrate from the gauge version)
  * a geometry it cannot transform consistently (unknown type, mesh without its artifact store) -> ``CalibrationRefused``

The parent (gauge) version is never modified; it stays in the store and the calibrated version names it as its parent.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from typing import Any, Mapping

import numpy as np

from engine.multiengine.alignment import Sim3
from engine.multiengine.scale_anchors import ScaleCalibration
from world_ir.schema_v1 import GeometryType, Vector3

CAL_KEY = "scale_calibration"
METRIC = "metric_anchored"


class CalibrationRefused(ValueError):
    """The calibration cannot be applied (not applicable, or the world cannot be transformed consistently)."""


class AlreadyCalibrated(CalibrationRefused):
    """The world already carries a physical calibration: scaling it again would multiply the scale twice."""


def world_calibration(world: Any) -> dict[str, Any] | None:
    """The shared calibration record of a world, wherever its producer stored it: the evidence-world calibration (``metadata['scale_calibration']``)
    or the legacy job's ``metadata['scale']['calibration']``. One reader, so readiness / export behave the same for both."""
    md = getattr(world, "metadata", None) or {}
    rec = md.get(CAL_KEY)
    if rec and isinstance(rec.get("calibration"), dict):
        return rec["calibration"]
    cal = (md.get("scale") or {}).get("calibration")
    return cal if isinstance(cal, dict) else None


def scale_block(world: Any) -> dict[str, Any]:
    """What a consumer (export response, status, UI banner) may say about a world's scale, from ONE function for the legacy and the evidence-world
    paths. ``units`` is "m" only when the scale is backed by an applicable measured calibration (or a device-depth scale); an accepted typical-size
    prior is ESTIMATED and never reads as metric-export-ready; otherwise the geometry is in model units."""
    md = getattr(world, "metadata", None) or {}
    legacy = md.get("scale") or {}
    cal = world_calibration(world)
    status = (cal or {}).get("status")
    state = str(legacy.get("state") or md.get("scale_state") or ("metric" if status in ("validated", "unvalidated", "estimated_prior") else "relative")).lower()
    basis = (cal or {}).get("basis") or legacy.get("scale_basis")
    device = basis == "device_depth"
    metric_ready = state in ("metric", "metric_anchored") and (status in ("validated", "unvalidated") or device)
    warnings: list[str] = []
    if not metric_ready:
        why = ("the scale is an accepted typical-size ESTIMATE, not a measurement" if status == "estimated_prior"
               else "no physical scale has been set" if state not in ("metric", "metric_anchored")
               else "the recorded scale has no applicable measured calibration")
        warnings.append(f"Geometry is in model units, not metres ({why})")
    elif status == "unvalidated":
        warnings.append("Metric scale rests on measurements that were not independently validated")
    return {"state": state, "units": "m" if metric_ready else "model_units", "metric_export_ready": metric_ready, "calibration_status": status,
            "basis": basis, "dimension_state": (cal or {}).get("dimension_state") or legacy.get("dimension_state"),
            "scale_m_per_gauge": (cal or {}).get("scale_m_per_gauge") or legacy.get("meters_per_unit"), "warnings": warnings}


def _tf(cal: ScaleCalibration) -> Sim3:
    if cal.transform is not None:
        return cal.transform
    return Sim3.from_arrays(float(cal.scale_m_per_gauge), np.eye(3), np.zeros(3))


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _scale_sigma_rel(cal: ScaleCalibration) -> float | None:
    if cal.scale_sigma is None or not cal.scale_m_per_gauge:
        return None
    return float(cal.scale_sigma / cal.scale_m_per_gauge)


def calibration_id(cal: ScaleCalibration, parent_version: str) -> str:
    blob = json.dumps({"p": parent_version, "c": cal.to_dict()}, sort_keys=True, default=str).encode()
    return "cal-" + hashlib.sha256(blob).hexdigest()[:12]


def _propagate(unc: Any, extent: Any, s: float, rel: float | None, rot: np.ndarray | None = None) -> Any:
    """1-sigma uncertainty after scaling by ``s``: sqrt((s*sigma)^2 + (extent * s * rel)^2), where ``rel`` is the scale's own relative sigma
    and ``extent`` is the dimension itself (dimension uncertainty) or the distance from the anchors' centroid (position uncertainty).
    A shape that cannot be propagated is kept scaled but flagged -- never silently dropped, never invented."""
    if unc is None:
        return None
    if _num(unc):
        if rel is None or not _num(extent):
            return {"sigma_m": s * float(unc), "scale_term": "not_propagated"}
        return float(math.hypot(s * float(unc), abs(float(extent)) * s * rel))
    if isinstance(unc, (list, tuple)) and len(unc) == 3 and all(_num(u) for u in unc):
        ext = list(extent) if isinstance(extent, (list, tuple)) and len(extent) == 3 else [extent] * 3
        if rel is None or not all(_num(e) for e in ext):
            return [{"sigma_m": s * float(u), "scale_term": "not_propagated"} for u in unc]
        return [float(math.hypot(s * float(u), abs(float(e)) * s * rel)) for u, e in zip(unc, ext, strict=True)]
    if isinstance(unc, dict) and isinstance(unc.get("covariance"), list):
        # A stated covariance: Sigma' = s^2 Sigma + (s rel)^2 e e^T -- one coherent scale error multiplies the whole vector (rank one)
        cov = np.asarray(unc["covariance"], float)
        r = np.eye(cov.shape[0]) if rot is None or rot.shape[0] != cov.shape[0] else rot  # a rotated frame rotates a POSITION covariance
        out = {**unc, "covariance": (s * s * r @ cov @ r.T).tolist(), "scaled_by": s}
        e = np.asarray(extent, float) if isinstance(extent, (list, tuple)) and len(extent) == cov.shape[0] else None
        if rel is not None and e is not None:
            re = r @ e
            out["covariance"] = (s * s * r @ cov @ r.T + (s * rel) ** 2 * np.outer(re, re)).tolist()
            out["scale_term"] = "propagated"
        else:
            out["scale_term"] = "not_propagated"
        return out
    return {"original": unc, "scale_term": "unrecognised_shape_not_propagated"}


def _transform_mesh(artifacts: Any, uri: str, tf: Sim3) -> tuple[str, str, np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(io.BytesIO(artifacts.get(uri)), file_type="ply", process=False)
    m = np.eye(4)
    m[:3, :3] = tf.scale * tf.R
    m[:3, 3] = tf.t
    mesh.apply_transform(m)
    new_uri, digest = artifacts.put(mesh.export(file_type="ply"))
    return new_uri, digest, mesh.bounds[0], mesh.bounds[1]


def _scale_determinations(dets: dict[str, Any], tf: Sim3, rel: float | None, cid: str, frame_id: str | None,
                          centroid: np.ndarray | None = None) -> dict[str, Any]:
    """The entity's cached position / dimensions determinations (value, candidates, covariance) follow the calibrated geometry, so a reader
    of ``determinations`` and a reader of ``transform`` never disagree. The gauge originals are kept under ``pre_calibration``."""
    out = copy.deepcopy(dets)
    s = float(tf.scale)
    for q in ("position", "dimensions"):
        d = out.get(q)
        if not isinstance(d, dict):
            continue
        d["pre_calibration"] = {"value": copy.deepcopy(dets[q].get("value")), "uncertainty": copy.deepcopy(dets[q].get("uncertainty")),
                                "frame": dets[q].get("frame")}

        def conv(v: Any) -> Any:
            if not (isinstance(v, (list, tuple)) and len(v) == 3 and all(_num(x) for x in v)):
                return v
            return tf.apply(np.asarray(v, float))[0].tolist() if q == "position" else (s * np.asarray(v, float)).tolist()

        d["value"] = conv(d.get("value"))
        d["candidates"] = [{**c, "value": conv(c.get("value"))} if isinstance(c, dict) else c for c in d.get("candidates") or []]
        val = d.get("value")
        ext = val if q == "dimensions" else (None if centroid is None or not _num(dets[q].get("value", [None])[0]) else
                                             (np.asarray(dets[q]["value"], float) - centroid).tolist())
        d["uncertainty"] = _propagate(d.get("uncertainty"), ext, s, rel, tf.R if q == "position" else None)
        d["scale_state"], d["units"], d["calibration_id"] = METRIC, "m", cid
        if q == "position" and frame_id:
            d["frame"] = frame_id
    return out


def apply_calibration(world: Any, cal: ScaleCalibration, *, parent_version: str, artifacts: Any = None,
                      cameras: Mapping[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    if not cal.applicable:
        raise CalibrationRefused(f"calibration status is {cal.status!r}: {cal.reason}")
    if (world.metadata or {}).get(CAL_KEY):
        raise AlreadyCalibrated("this world already carries a physical calibration; recalibrate from its gauge ancestor, never rescale it")
    tf = _tf(cal)
    s, R, t = float(tf.scale), tf.R, tf.t
    rel = _scale_sigma_rel(cal)
    centroid = None
    if cal.residuals:
        locs = np.array([r["location_gauge"] for r in cal.residuals if r.get("location_gauge") is not None], float)
        centroid = locs.mean(0) if len(locs) else None
    cid = calibration_id(cal, parent_version)
    frame_id = f"physical:{cid}" if cal.transform is not None else None
    w = copy.deepcopy(world)
    counts = {"entities": 0, "boxes": 0, "planes": 0, "meshes": 0, "relationships": 0, "uncertainties": 0}

    for g in w.geometries.values():
        qm = dict(g.quality_metrics or {})
        pre: dict[str, Any] = {}
        lo = np.array([g.bounds_min.x, g.bounds_min.y, g.bounds_min.z]) if g.bounds_min is not None else None
        hi = np.array([g.bounds_max.x, g.bounds_max.y, g.bounds_max.z]) if g.bounds_max is not None else None
        if lo is not None and hi is not None:
            pre["bounds"] = [lo.tolist(), hi.tolist()]
        if g.type is GeometryType.BOX:
            if lo is None or hi is None:
                raise CalibrationRefused(f"box geometry {g.id} has no bounds to transform")
            centre, dims = (lo + hi) / 2, hi - lo
            c2 = tf.apply(centre)[0]
            if cal.transform is not None:  # the box model has no orientation: keep its extents, hull the rotated box, flag it
                corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                pc = tf.apply(corners)
                nlo, nhi = pc.min(0), pc.max(0)
                qm["bounds_after_rotation"] = "axis-aligned hull of the rotated box; the true extents are dimensions_m"
            else:
                nlo, nhi = c2 - s * dims / 2, c2 + s * dims / 2
            qm["dimensions_m"] = (s * dims).tolist()
            unc = qm.get("uncertainty")
            if isinstance(unc, dict):
                ref = float(np.linalg.norm(centre - centroid)) if centroid is not None else None
                pos_unc = unc.get("position")
                ref_vec = (centre - centroid).tolist() if centroid is not None and isinstance(pos_unc, dict) else ref
                qm["uncertainty"] = {"position": _propagate(pos_unc, ref_vec, s, rel, R),
                                     "dimensions": _propagate(unc.get("dimensions"), dims.tolist(), s, rel),
                                     "propagated_from_gauge": unc, "scale_sigma_rel": rel}
                counts["uncertainties"] += 1
            g.bounds_min, g.bounds_max = Vector3(*nlo), Vector3(*nhi)
            counts["boxes"] += 1
        elif g.type is GeometryType.PLANE:
            plane, hull = qm.get("plane"), qm.get("observed_hull")
            if not plane or not hull:
                raise CalibrationRefused(f"plane geometry {g.id} has no plane/hull to transform")
            n2 = R @ np.asarray(plane["normal"], float)
            qm["plane"] = {"normal": n2.tolist(), "offset": float(s * plane["offset"] + n2 @ t)}
            ph = tf.apply(np.asarray(hull, float))
            qm["observed_hull"] = np.round(ph, 6).tolist()
            qm["observed_area"] = float(qm.get("observed_area", 0.0)) * s * s
            qm["rms_residual"] = float(qm.get("rms_residual", 0.0)) * s
            g.bounds_min, g.bounds_max = Vector3(*ph.min(0)), Vector3(*ph.max(0))
            counts["planes"] += 1
        elif g.type is GeometryType.MESH:
            if artifacts is None or not g.data_uri:
                raise CalibrationRefused(f"mesh geometry {g.id} cannot be transformed without its artifact store; refusing to half-scale the world")
            pre["data_uri"], pre["data_hash"] = g.data_uri, g.data_hash
            g.data_uri, g.data_hash, mlo, mhi = _transform_mesh(artifacts, g.data_uri, tf)
            g.bounds_min, g.bounds_max = Vector3(*mlo), Vector3(*mhi)
            if _num(qm.get("voxel_length_gauge")):
                qm["voxel_length_m"] = s * float(qm["voxel_length_gauge"])
            counts["meshes"] += 1
        else:
            raise CalibrationRefused(f"geometry {g.id} has type {g.type}; no consistent transform is defined for it")
        if isinstance(qm.get("scale"), dict):  # the nested label of the source observations' scale must not keep saying gauge
            pre["scale"] = qm["scale"]
            qm["scale"] = {"scale_state": METRIC, "units": "m", "calibration_id": cid}
        qm.update({"scale_state": METRIC, "units": "m", "calibration_id": cid, "pre_calibration": pre})
        if frame_id:
            qm["frame_id"] = frame_id
        g.quality_metrics = qm

    for e in w.entities.values():
        tr = e.transform
        if cal.transform is not None and tr and any(k in tr for k in ("rotation", "orientation", "quaternion", "rotation_wxyz")):
            raise CalibrationRefused(f"entity {e.id} carries an orientation; composing it with the calibration rotation is not implemented, "
                                     "and leaving it unrotated would misplace the object")
        ev = (e.custom_properties or {}).get("evidence")
        if isinstance(ev, dict) and isinstance(ev.get("determinations"), dict):
            ev["determinations"] = _scale_determinations(ev["determinations"], tf, rel, cid, frame_id, centroid)
        if tr and tr.get("position"):
            p = tr["position"]
            q = tf.apply(np.array([p["x"], p["y"], p["z"]], float))[0]
            e.transform = {**tr, "position": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2])}, "scale_state": METRIC, "units": "m",
                           "physically_metric": True, "calibration_id": cid,
                           "pre_calibration": {"position": p, "frame_id": tr.get("frame_id"), "scale_state": tr.get("scale_state")}}
            if frame_id:
                e.transform["frame_id"] = frame_id
            counts["entities"] += 1
        for r in e.relationships:
            keys = [k for k in (r.metadata or {}) if k.endswith("_gauge") and _num(r.metadata[k])]
            for k in keys:
                r.metadata[k[:-6] + "_m"] = s * float(r.metadata[k])
            if keys:
                r.metadata["scale_state"], r.metadata["calibration_id"] = METRIC, cid
                counts["relationships"] += 1
            elif r.metadata is not None and "scale_state" not in r.metadata:  # prose basis with embedded lengths: say which units they are in
                r.metadata["basis_numbers_in"] = "sfm_gauge_units (pre-calibration); multiply by scale_m_per_gauge for metres"
                r.metadata["calibration_id"] = cid

    cam_out = {k: tf.apply(np.asarray(v, float))[0].tolist() for k, v in cameras.items()} if cameras else None
    record = {"id": cid, "parent_version": parent_version, "scale_m_per_gauge": s, "scale_sigma": cal.scale_sigma, "scale_sigma_rel": rel,
              "frame_id": frame_id, "calibration": cal.to_dict(), "camera_centres_physical": cam_out, "applied": counts,
              "note": "gauge ancestor preserved as the parent version; every metric quantity carries this calibration_id"}
    w.metadata = {**(w.metadata or {}), CAL_KEY: record, "scale_state": METRIC}
    return w, record


def commit_calibrated_version(store: Any, parent_version: str, cal: ScaleCalibration, *, artifacts: Any = None,
                              cameras: Mapping[str, Any] | None = None, source_session_ids: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    """Apply ``cal`` to ``parent_version`` and save the result as a new immutable, verified version. Idempotent: the version id is derived from
    the parent and the calibration content, so repeating the same request returns the same version instead of stacking a second one."""
    from world_ir.validation import validate_world_ir
    from worldstore.store import WorldStoreError

    if not cal.applicable:
        raise CalibrationRefused(f"calibration status is {cal.status!r}: {cal.reason}")
    vid = f"{parent_version}-m{calibration_id(cal, parent_version)[4:]}"
    if (store._versions_dir / f"{vid}.json").exists():
        return vid, store.load_version(vid).metadata[CAL_KEY]
    world, record = apply_calibration(store.load_version(parent_version), cal, parent_version=parent_version, artifacts=artifacts, cameras=cameras)
    errors = [i for i in validate_world_ir(world).issues if str(getattr(i.severity, "value", i.severity)).lower() == "error"]
    if errors:
        raise CalibrationRefused("calibrated world failed validation: " + "; ".join(f"{i.code}: {i.message}" for i in errors[:3]))
    try:
        store.save_version(world, parent=parent_version, version_id=vid, source_session_ids=source_session_ids)
    except WorldStoreError:
        pass  # a concurrent identical request won the race; the content is identical by construction
    problems = store.verify_version(vid)
    if problems:
        try:  # an unverifiable version must never be reachable as HEAD
            store.quarantine_version(vid)
        except Exception:  # noqa: BLE001
            pass
        raise CalibrationRefused(f"calibrated version failed integrity verification and was quarantined: {problems[:2]}")
    return vid, record
