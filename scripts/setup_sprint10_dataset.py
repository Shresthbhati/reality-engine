"""Setup script for scene-disjoint manual component evaluation dataset (Sprint 10 P1)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def setup_disjoint_dataset() -> None:
    eval_dir = Path("datasets/scene_disjoint_evaluation")
    img_dir = eval_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. 5 Real-world exterior architectural images from South Building
    sb_dir = Path("datasets/south_building/images")
    sb_imgs = ["P1180141.JPG", "P1180145.JPG", "P1180149.JPG", "P1180153.JPG", "P1180157.JPG"]
    for name in sb_imgs:
        src = sb_dir / name
        dst = img_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # 2. 5 Multi-view interior capture images from real_room_capture
    rc_dir = Path("datasets/real_room_capture")
    rc_imgs = ["IMG_0000.jpg", "IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg", "IMG_0004.jpg"]
    for name in rc_imgs:
        src = rc_dir / name
        dst = img_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # Manifest metadata
    manifest = {
        "dataset_name": "scene_disjoint_evaluation",
        "version": "1.0.0",
        "description": "Scene-disjoint, multi-view, manually annotated ground-truth dataset spanning indoor and outdoor environments.",
        "splits": {
            "all": sb_imgs + rc_imgs,
            "exterior_disjoint": sb_imgs,
            "interior_disjoint": rc_imgs,
        },
        "target_resolution": {"width": 1280, "height": 960},
        "taxonomy_families": [
            "doors_and_windows",
            "electrical",
            "furniture",
            "plumbing_and_hvac",
            "building_and_exterior",
        ],
        "annotation_protocol": {
            "coordinate_format": "[x1, y1, x2, y2] in pixels",
            "strict_one_to_one_hierarchy": True,
            "multi_view_physical_tracking": True,
            "synthetic_labels_allowed": False,
        },
    }

    with open(eval_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Setup scene_disjoint_evaluation with {len(sb_imgs + rc_imgs)} images.")


if __name__ == "__main__":
    setup_disjoint_dataset()
