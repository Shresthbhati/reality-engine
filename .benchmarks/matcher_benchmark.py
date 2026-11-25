"""Matcher benchmark for camera registration.

Creates synthetic multi-view images of a simple "room" (geometric shapes on a
background) and benchmarks available matchers (SIFT, ORB, LightGlue-DISK,
LightGlue-ALIKED) on image pairs. Reports inlier ratio, match count and
runtime per matcher.

Usage:
    python .benchmarks/matcher_benchmark.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import cv2

from reconstruction.matching import available_matchers, get_matcher, MatcherUnavailableError

IMG_SIDE = 480
N_PAIRS = 6


def make_room_image(seed: int, noise: float = 8.0) -> np.ndarray:
    """Synthetic room image: random walls (lines), corners (circles), floor texture."""
    rng = np.random.default_rng(seed)
    img = np.zeros((IMG_SIDE, IMG_SIDE, 3), dtype=np.uint8)
    # Background gradient
    y, x = np.mgrid[0:IMG_SIDE, 0:IMG_SIDE]
    img[:] = (np.clip((x + y) / (2 * IMG_SIDE) * 120 + 60, 0, 255)).astype(np.uint8)[:, :, None]
    # Ceiling / floor line
    cv2.line(img, (0, 120), (IMG_SIDE, 120), (200, 200, 200), 3)
    cv2.line(img, (0, 360), (IMG_SIDE, 360), (100, 100, 100), 3)
    # Walls as random rectangles
    for _ in range(6):
        x1 = int(rng.integers(50, 400)); y1 = int(rng.integers(130, 340))
        x2 = x1 + int(rng.integers(30, 120)); y2 = y1 + int(rng.integers(30, 120))
        color = tuple(int(c) for c in rng.integers(40, 220, size=3))
        cv2.rectangle(img, (x1, y1), (min(x2, 470), min(y2, 470)), color, -1)
        cv2.rectangle(img, (x1, y1), (min(x2, 470), min(y2, 470)), (0, 0, 0), 2)
    # Corners / features
    for _ in range(12):
        cx = int(rng.integers(60, 420)); cy = int(rng.integers(140, 350))
        r = int(rng.integers(8, 25))
        cv2.circle(img, (cx, cy), r, (255, 255, 255), -1)
        cv2.circle(img, (cx, cy), r, (0, 0, 0), 2)
    # Add noise
    noise_img = rng.integers(0, noise, size=img.shape, dtype=np.uint8)
    img = cv2.add(img, noise_img)
    return img


def warp_view(img: np.ndarray, seed: int) -> np.ndarray:
    """Apply a camera-like transform: small rotation + scale + translation + perspective."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    # Random affine: slight rotation, scale, translation
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.9, 1.1)
    tx = int(rng.integers(-30, 30))
    ty = int(rng.integers(-30, 30))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return warped


def main() -> None:
    out_dir = Path(".benchmarks/matcher_benchmark_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create N image pairs (base + warped view)
    pairs = []
    for i in range(N_PAIRS):
        base = make_room_image(seed=1000 + i)
        view = warp_view(base, seed=2000 + i)
        p1 = out_dir / f"pair_{i:02d}_a.png"
        p2 = out_dir / f"pair_{i:02d}_b.png"
        cv2.imwrite(str(p1), base)
        cv2.imwrite(str(p2), view)
        pairs.append((str(p1), str(p2)))

    paths = {}
    for i, (a, b) in enumerate(pairs):
        paths[f"pair_{i:02d}_a"] = Path(a)
        paths[f"pair_{i:02d}_b"] = Path(b)

    # Test each available matcher
    results = {}
    for matcher_name in ("sift", "orb", "lightglue@disk", "lightglue@aliked"):
        try:
            matcher = get_matcher(matcher_name, paths)
        except MatcherUnavailableError as exc:
            results[matcher_name] = {"available": False, "reason": str(exc)}
            continue

        pair_results = []
        for i, (a, b) in enumerate(pairs):
            aid = f"pair_{i:02d}_a"
            bid = f"pair_{i:02d}_b"
            try:
                mr = matcher.match(aid, bid)
                pair_results.append({
                    "pair": i,
                    "matches": mr.matches,
                    "inliers": mr.inliers,
                    "inlier_ratio": round(mr.inlier_ratio, 4),
                    "runtime_s": round(mr.runtime_s, 4),
                })
            except Exception as exc:
                pair_results.append({"pair": i, "error": str(exc)})

        if pair_results:
            inlier_ratios = [r["inlier_ratio"] for r in pair_results if "error" not in r]
            runtimes = [r["runtime_s"] for r in pair_results if "error" not in r]
            results[matcher_name] = {
                "available": True,
                "name": matcher.name,
                "license": matcher.license,
                "non_commercial": matcher.non_commercial,
                "mean_inlier_ratio": round(float(np.mean(inlier_ratios)), 4),
                "min_inlier_ratio": round(float(np.min(inlier_ratios)), 4),
                "mean_runtime_s": round(float(np.mean(runtimes)), 4),
                "pairs": pair_results,
            }

    print(json.dumps(results, indent=2))

    # Write report
    report_path = out_dir / "results.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
