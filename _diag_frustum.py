"""Diagnose where cameras actually see points - frustum boundary test."""
from reconstruction.backend.interface import (
    ReconstructedCameraPose, ReconstructedPoint, ReconstructionResult
)
from reconstruction.calibration.camera import CameraIntrinsics, camera_from_pose
from perception.quality.assessment import assess_evidence_quality

intr = CameraIntrinsics(fx=640, fy=640, cx=320, cy=240, width=640, height=480)
c0_pose = ReconstructedCameraPose('c0', (0, 0, 0), (1, 0, 0, 0))
c1_pose = ReconstructedCameraPose('c1', (0.3, 0, 0), (1, 0, 0, 0))
cams = [camera_from_pose(intr, c0_pose), camera_from_pose(intr, c1_pose)]
poses = [c0_pose, c1_pose]

print("Camera c0 at (0,0,0) looking +Z; c1 at (0.3,0,0) looking +Z")
print(f"fx={intr.fx}, cx={intr.cx}, width={intr.width}")
print(f"GSD formula: dist_m * 1000 / fx = dist_m * {1000/intr.fx:.4f} mm/px")
print()
print("Point at (px, 0, 2.0) with source_evidence_ids=['c0','c1']:")
for px in [-1.0, -0.5, 0.0, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]:
    p = ReconstructedPoint((px, 0.0, 2.0), f'p{px}', ['c0', 'c1'])
    r = ReconstructionResult([p], poses, 'success')
    rep = assess_evidence_quality(r, cams)
    print(f"  px={px:+.1f} obs={rep.observed_fraction} gsd={rep.gsd_mm_per_px} tier={rep.detail_tier} unproj={rep.unprojectable_point_ids}")
print()
print("Now test z=1.0 plane:")
for px in [-1.0, -0.5, 0.0, 0.2, 0.4, 1.0]:
    p = ReconstructedPoint((px, 0.0, 1.0), f'p{px}', ['c0'])
    r = ReconstructionResult([p], poses, 'success')
    rep = assess_evidence_quality(r, cams)
    print(f"  px={px:+.1f} z=1.0 obs={rep.observed_fraction} gsd={rep.gsd_mm_per_px} tier={rep.detail_tier} unproj={rep.unprojectable_point_ids}")
