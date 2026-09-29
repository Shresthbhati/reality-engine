"""Experiment: does persistent COLMAP state (image_registrator) beat a fresh rebuild? Measured, not assumed.

    METHOD A  fresh:        all photos so far -> feature_extractor + matcher + mapper, from nothing
    METHOD B  incremental:  previous sparse model kept; features for the NEW photos only;
                            image_registrator + point_triangulator + bundle_adjuster

Both run through the same backend and the same photographs, so the only difference is the strategy.
Reported per step: registration success, point count, mean reprojection error (COLMAP's own track error),
camera stability (frame-independent, versus the previous step), spatial coverage (extent of the registered
cameras and of the points), wall-clock runtime, and what happened when the prior model was too weak.

    python scripts/experiment_incremental_registration.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.pipeline.world_delta import camera_consistency  # noqa: E402
from evidence.session import EvidenceItem, EvidenceKind  # noqa: E402
from reconstruction.backend.colmap_backend import ColmapReconstructionBackend  # noqa: E402
from reconstruction.colmap_session import ColmapSession  # noqa: E402

IMGS = sorted((Path(__file__).resolve().parents[1] / "datasets" / "south_building" / "images").glob("*.JPG"))


def items(paths):
    return [EvidenceItem(id="ev-" + p.stem, kind=EvidenceKind.PHOTO, source_uri=p.resolve().as_uri()) for p in paths]


def mean_reprojection_error(model_dir: Path) -> float:
    errs = []
    for line in (model_dir / "points3D.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            errs.append(float(line.split()[7]))
    return float(np.mean(errs)) if errs else float("nan")


def extent(points: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((points - points.mean(0)) ** 2, axis=1)))) if len(points) else 0.0


def measure(result, session, seconds, info):
    cams = {p.evidence_id: [float(x) for x in p.position] for p in result.camera_poses}
    pts = np.asarray([p.position for p in result.points], float)
    return {
        "registered": len(cams), "points": len(pts), "reproj_px": mean_reprojection_error(session.staging / "sparse" / "0"),
        "cam_extent": extent(np.asarray(list(cams.values()), float)), "pt_extent": extent(pts),
        "seconds": seconds, "mode": info.get("mode"), "cams": cams,
    }


def run(label, prior_batches, new_batch):
    """Build the prior with the same backend, then add ``new_batch`` by method A and by method B."""
    everything = [p for b in prior_batches for p in b] + list(new_batch)
    rows = {}
    # ---- prior state (committed), identical for both methods
    inc_session = ColmapSession(Path(tempfile.mkdtemp()))
    be = ColmapReconstructionBackend(session=inc_session)
    prior_paths = [p for b in prior_batches for p in b]
    prior = be.reconstruct(items(prior_paths))
    inc_session.commit()
    prior_cams = {p.evidence_id: [float(x) for x in p.position] for p in prior.camera_poses}

    # ---- METHOD B: incremental
    t = time.time()
    res_b = be.reconstruct(items(everything))
    rows["B incremental"] = measure(res_b, inc_session, time.time() - t, be.last_run_info)
    rows["B incremental"]["note"] = be.last_run_info.get("reason")
    rows["B incremental"]["inc_registered"] = be.last_run_info.get("incremental_registered")
    rows["B incremental"]["full_registered"] = be.last_run_info.get("full_registered")

    # ---- METHOD A: fresh rebuild of everything from nothing
    fresh_session = ColmapSession(Path(tempfile.mkdtemp()))
    fresh = ColmapReconstructionBackend(session=fresh_session)
    t = time.time()
    res_a = fresh.reconstruct(items(everything))
    rows["A fresh"] = measure(res_a, fresh_session, time.time() - t, fresh.last_run_info)

    print(f"\n=== {label}: prior {len(prior_paths)} photos ({len(prior.points)} points, {len(prior_cams)} cameras) "
          f"+ {len(new_batch)} new ===")
    print(f"{'method':<15}{'registered':>11}{'points':>8}{'reproj px':>10}{'cam extent':>11}{'pt extent':>10}{'seconds':>9}  stability vs prior")
    for name, r in rows.items():
        cc = camera_consistency(prior_cams, r["cams"])
        stab = f"{cc['relative']:.4f} (max {max(cc['per_camera'].values()):.4f})" if cc else "n/a (<4 shared cameras)"
        print(f"{name:<15}{r['registered']:>11}{r['points']:>8}{r['reproj_px']:>10.3f}{r['cam_extent']:>11.3f}"
              f"{r['pt_extent']:>10.3f}{r['seconds']:>9.1f}  {stab}   [{r['mode']}]")
    print("  incremental step:", {k: rows["B incremental"].get(k) for k in ("inc_registered", "full_registered", "note")})


def main():
    A, B, C, D = IMGS[15:18], IMGS[18:21], IMGS[21:24], IMGS[24:27]
    run("weak prior (3 photos)", [A], B)
    run("rich prior (6 photos)", [A, B], C)
    run("chain step (9 photos, 7 placed)", [A, B, C], D)


if __name__ == "__main__":
    main()
