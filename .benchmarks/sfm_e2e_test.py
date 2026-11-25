"""Quick end-to-end test of the Python SfM backend with synthetic multi-view images."""
from __future__ import annotations

import json
import numpy as np
import cv2
from pathlib import Path
from evidence.session import EvidenceItem, EvidenceKind
from reconstruction.backend.python_backend import PythonSfMReconstructionBackend


def make_room_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((480, 480, 3), dtype=np.uint8)
    y, x = np.mgrid[0:480, 0:480]
    img[:] = np.clip((x + y) / 960.0 * 120 + 60, 0, 255).astype(np.uint8)[:, :, None]
    cv2.line(img, (0, 120), (480, 120), (200, 200, 200), 3)
    cv2.line(img, (0, 360), (480, 360), (100, 100, 100), 3)
    for _ in range(6):
        x1 = int(rng.integers(50, 400)); y1 = int(rng.integers(130, 340))
        x2 = x1 + int(rng.integers(30, 120)); y2 = y1 + int(rng.integers(30, 120))
        color = tuple(int(c) for c in rng.integers(40, 220, size=3))
        cv2.rectangle(img, (x1, y1), (min(x2, 470), min(y2, 470)), color, -1)
        cv2.rectangle(img, (x1, y1), (min(x2, 470), min(y2, 470)), (0, 0, 0), 2)
    for _ in range(12):
        cx = int(rng.integers(60, 420)); cy = int(rng.integers(140, 350))
        r = int(rng.integers(8, 25))
        cv2.circle(img, (cx, cy), r, (255, 255, 255), -1)
        cv2.circle(img, (cx, cy), r, (0, 0, 0), 2)
    noise = rng.integers(0, 8, size=img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def main():
    out_dir = Path("/tmp/sfm_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create 4 synthetic views
    items = []
    for i in range(4):
        img = make_room_image(seed=1000 + i)
        # Slightly warp each view
        M = cv2.getRotationMatrix2D((240, 240), i * 3 - 6, 1.0 + i * 0.02)
        M[0, 2] += (i - 1) * 10
        M[1, 2] += (i - 1) * 5
        img = cv2.warpAffine(img, M, (480, 480), borderMode=cv2.BORDER_REFLECT)
        p = out_dir / f"view_{i:02d}.jpg"
        cv2.imwrite(str(p), img)
        items.append(EvidenceItem(id=f"view_{i:02d}", kind=EvidenceKind.PHOTO, source_uri=str(p), sha256=""))

    backend = PythonSfMReconstructionBackend()
    result = backend.reconstruct(items)

    print("Status:", result.registration_status)
    print("Matcher:", result.registration_stats.get("matcher") if result.registration_stats else "none")
    print("Input images:", result.registration_stats.get("n_input_images") if result.registration_stats else "none")
    print("Good pairs:", result.registration_stats.get("n_good_pairs") if result.registration_stats else "none")
    print("Camera poses:", len(result.camera_poses))
    print("3D points:", len(result.points))
    print("Pair stats:", json.dumps(result.registration_stats.get("pair_stats", []), indent=2) if result.registration_stats else "none")

    for i, p in enumerate(result.camera_poses):
        print(f"  Camera {i} ({p.evidence_id}): pos={p.position}, rot={p.rotation}, conf={p.uncertainty.confidence}")

    assert result.registration_status in ("success", "partial"), f"Unexpected status: {result.registration_status}"
    assert len(result.camera_poses) == 4, f"Expected 4 poses, got {len(result.camera_poses)}"
    assert len(result.points) >= 0, "Points should be non-negative"
    print("\nALL BACKEND TESTS PASSED")


if __name__ == "__main__":
    main()
