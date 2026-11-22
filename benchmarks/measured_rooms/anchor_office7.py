"""Scale-anchor a production run on 7-Scenes "office" photographs using the dataset's OWN camera poses as point anchors.

The ground-truth camera centres (KinectFusion, metres) were never given to the pipeline. They are an INDEPENDENT but imperfect physical
reference: centimetre-level estimates, not a laser survey. sigma_m below is therefore a stated assumption, recorded in the output.
Alternate registered cameras are held out, so the reported error is on cameras the transform never saw.

    python benchmarks/measured_rooms/anchor_office7.py <run_dir> <frames_dir_with_pose_files> [sigma_m]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from engine.multiengine.scale_anchors import PointAnchor, calibrate_points

ASSUMED_POSE_SIGMA_M = 0.02


def _job_frame_camera_centres(run_dir: Path, frames: Path) -> dict[str, list[float]]:
    """Job-frame camera centres by original frame name. The cameras JSON is a content-addressed artifact of the run; evidence ids are
    mapped back to the uploaded frame by nearest 32x24 grey thumbnail (the colmap session keeps the images under their evidence id)."""
    from PIL import Image

    cams = {}
    for f in (run_dir / "art").rglob("*"):
        if f.is_file() and f.stat().st_size < 2_000_000:
            try:
                d = json.loads(f.read_bytes())
            except Exception:  # noqa: BLE001
                continue
            if isinstance(d, dict) and d.get("cameras") and isinstance(d["cameras"], list) and "position_m" in d["cameras"][0]:
                cams = {c["evidence_id"]: c["position_m"] for c in d["cameras"]}
                break

    def small(p: Path) -> np.ndarray:
        return np.asarray(Image.open(p).convert("L").resize((32, 24)), float)

    orig = {p.name.replace(".color.png", ""): small(p) for p in frames.glob("frame-*.color.png")}
    sess = next(iter((run_dir / "ws" / "colmap-sessions").glob("*/current/images")), None)
    out: dict[str, list[float]] = {}
    if sess is None or not orig:
        return out
    for f in sess.iterdir():
        ev = f.stem
        if ev in cams:
            out[min(orig, key=lambda n: float(((orig[n] - small(f)) ** 2).mean()))] = cams[ev]
    return out


def main(run_dir: Path, frames: Path, sigma_m: float = ASSUMED_POSE_SIGMA_M) -> int:
    sfm = _job_frame_camera_centres(run_dir, frames)
    gt = {p.name.replace(".pose.txt", ""): np.loadtxt(p).reshape(4, 4)[:3, 3] for p in frames.glob("frame-*.pose.txt")}
    cams = sfm
    names = sorted(set(sfm) & set(gt))
    anchors = [PointAnchor(n, tuple(map(float, sfm[n])), tuple(map(float, gt[n])), sigma_m, "7scenes_kinectfusion_camera_pose",
                           "2013-01-01T00:00:00+00:00",  # dataset release year only; the capture time of each frame is not recorded
                           "dataset-supplied KinectFusion camera-to-world pose (cm-level estimate); NOT a laser survey",
                           held_out=(i % 2 == 1)) for i, n in enumerate(names)]
    cal = calibrate_points(anchors)
    out = {"registered": len(cams), "matched_to_ground_truth": len(names), "assumed_pose_sigma_m": sigma_m,
           "held_out_policy": "every second matched camera (by sorted name) is held out", "calibration": cal.to_dict(),
           "limits": ["anchors are camera centres only (positions), so they say nothing about wall/object dimensions directly",
                      "ground truth is itself an estimate; this is agreement with KinectFusion, not independent laser survey"]}
    (run_dir / "scale_calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    brief = {k: v for k, v in cal.to_dict().items() if k not in ("residuals", "transform")}
    print(json.dumps({**{k: v for k, v in out.items() if k != "calibration"}, "calibration": brief}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else ASSUMED_POSE_SIGMA_M))
