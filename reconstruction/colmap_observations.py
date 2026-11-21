"""Per-image keypoint observations of a COLMAP sparse model, expressed in the JOB's coordinate frame.

Why this exists: aligning a relative depth map to "all sparse points projected through the camera" is wrong on real scenes -- points
hidden behind furniture or walls project into the image too, and on a real office run that gave a depth/geometry correlation of ~0.2
where the image's OWN observed keypoints give 0.72-0.98. The correct reference for image ``i`` is the set of 3D points that were
actually observed in image ``i`` (pixel + point).

The model is in COLMAP's raw frame; the job re-expresses cameras and points in its canonical frame by a similarity transform. That
transform is recovered from the camera centres (the same registered cameras exist in both frames) and its residual is checked: if the
two frames are not related by a similarity to within ``max_resid`` (relative to the trajectory extent), nothing is returned -- a
frame that cannot be reconciled is an error, never a guess.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from reconstruction.similarity import umeyama


def read_model(model_dir: Path) -> dict[str, dict[str, Any]]:
    """``{image stem: {"centre": raw camera centre, "xy": Nx2 pixels, "xyz": Nx3 raw points}}`` for the registered images.
    ``{}`` when pycolmap is unavailable or the model cannot be read."""
    try:
        import pycolmap

        rec = pycolmap.Reconstruction(str(model_dir))
        out: dict[str, dict[str, Any]] = {}
        for im in rec.images.values():
            pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            R, t = np.asarray(pose.rotation.matrix(), float), np.asarray(pose.translation, float)
            xy, xyz = [], []
            for p in im.points2D:
                if p.has_point3D():
                    xy.append(np.asarray(p.xy, float))
                    xyz.append(np.asarray(rec.points3D[p.point3D_id].xyz, float))
            out[Path(im.name).stem] = {"centre": -R.T @ t, "xy": np.array(xy).reshape(-1, 2), "xyz": np.array(xyz).reshape(-1, 3)}
        return out
    except Exception:  # noqa: BLE001 -- missing library or unreadable model: no observations, never invented ones
        return {}


def observations_in_job_frame(
    model: Mapping[str, Mapping[str, Any]],
    job_centres: Mapping[str, Sequence[float]],
    *,
    max_resid: float = 1e-3,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Map the model's observations into the job frame. ``job_centres`` is ``{image stem: camera centre in the job frame}``.
    Returns ``(observations, facts)``; ``observations`` is empty (with the reason in ``facts``) when fewer than 4 cameras match or the
    similarity residual exceeds ``max_resid`` x the trajectory extent."""
    names = sorted(set(model) & set(job_centres))
    facts: dict[str, Any] = {"matched_cameras": len(names)}
    if len(names) < 4:
        return {}, {**facts, "status": "refused", "reason": "fewer than 4 cameras are common to the model and the job"}
    raw = np.array([model[n]["centre"] for n in names], float)
    job = np.array([job_centres[n] for n in names], float)
    try:
        s, R, t = umeyama(raw, job)
    except ValueError as exc:
        return {}, {**facts, "status": "refused", "reason": f"frames cannot be related: {exc}"}
    resid = np.linalg.norm(s * raw @ R.T + t - job, axis=1)
    extent = float(np.linalg.norm(job.max(0) - job.min(0))) or 1.0
    facts.update({"scale_raw_to_job": s, "max_centre_residual": float(resid.max()), "trajectory_extent": extent})
    if resid.max() > max_resid * extent:
        return {}, {**facts, "status": "refused", "reason": "the job frame is not a similarity of the model frame (camera-centre residual too large)"}
    obs = {n: {"xy": model[n]["xy"], "xyz": s * model[n]["xyz"] @ R.T + t} for n in names}
    return obs, {**facts, "status": "ok"}
