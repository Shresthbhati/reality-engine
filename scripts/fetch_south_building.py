#!/usr/bin/env python3
"""Reproducible acquisition + regeneration of the south-building dataset
(RECONSTRUCTION_REAL_WORLD_ROBUSTNESS P0: another developer can
reproduce the same run).

Dataset: 128 real photographs of the South Building at UNC Chapel Hill,
one Panasonic DMC-TZ3 camera, distributed by the COLMAP authors as the
canonical "south-building" example set (with a sparse reference
reconstruction). See datasets/south_building/MANIFEST.json for identity,
source URL, original checksum, derivative recipe, and per-image sha256.

Commands:
  python scripts/fetch_south_building.py download   # fetch original zip to a temp dir
  python scripts/fetch_south_building.py verify     # verify committed subset against MANIFEST.json
  python scripts/fetch_south_building.py regenerate # re-derive the committed subset from the original

Notes:
  - The repo commits a deterministic 32-image subset (every 4th image
    in sorted order, downscaled to width 1024 JPEG q90) plus the sparse
    reference model FILTERED to those 32 images (provenance ids kept).
  - The unfiltered full-resolution set is one `download` away; the
    original zip's sha256 is pinned in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "datasets" / "south_building"
MANIFEST = DATASET / "MANIFEST.json"
SOURCE_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
ORIGINAL_SHA256 = "d210016bd2de20936a5f02b87fd38a76bf0440c42d045231218372cf9db9a7a1"
STRIDE = 4
TARGET_WIDTH = 1024
JPEG_QUALITY = 90


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> int:
    from PIL import Image  # noqa: F401  (validate pillow present early)

    with tempfile.TemporaryDirectory(prefix="south-building-") as tmp:
        zip_path = Path(tmp) / "south-building.zip"
        print(f"downloading {SOURCE_URL} ...")
        urllib.request.urlretrieve(SOURCE_URL, zip_path)
        digest = _sha256(zip_path)
        if digest != ORIGINAL_SHA256:
            print(f"FATAL: sha256 mismatch: got {digest}, expected {ORIGINAL_SHA256}", file=sys.stderr)
            return 1
        print("sha256 verified:", digest)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(Path(tmp) / "extract")
        print("extracted to:", Path(tmp) / "extract")
        print("(kept on disk until this process exits -- regenerate immediately)")
        # Regenerate right away so the temp dir is actually used.
        _regenerate_from(Path(tmp) / "extract" / "south-building")
    return 0


def _filter_images_txt(text: str, keep_names: set) -> str:
    """Filter a COLMAP text images.txt to the named images, keeping the
    original image_id lines (provenance ids stay stable). Each image is
    a 2-line block: header (image_id ...) + a points line."""
    out_lines = []
    blocks = text.splitlines()
    i = 0
    keep_ids = set()
    while i < len(blocks):
        line = blocks[i]
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            i += 1
            continue
        name = parts[-1]  # NAME is the last header token
        if name in keep_names:
            keep_ids.add(int(parts[0]))
            out_lines.append(line)
            if i + 1 < len(blocks):
                out_lines.append(blocks[i + 1])
        i += 2
    return "\n".join(out_lines) + "\n"


def _filter_points3d(text: str, keep_ids: set) -> str:
    """Keep 3D points observed by at least one kept image (track element
    image_id in the trailing pairs)."""
    out = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            out.append(line)
            continue
        parts = line.split()
        # ELEMENTS: POINT3D_ID X Y Z R G B ERROR TRACK[IMAGE_ID POINT2D_IDX]...
        track = parts[8:]
        if any(int(t) in keep_ids for t in track[::2]):
            out.append(line)
    return "\n".join(out) + "\n"


def _regenerate_from(extracted_root: Path) -> None:
    from PIL import Image

    src_images = extracted_root / "images"
    src_sparse = extracted_root / "sparse"
    names_all = sorted(p.name for p in src_images.iterdir())
    subset_names = names_all[::STRIDE]
    keep = set(subset_names)

    (DATASET / "images").mkdir(parents=True, exist_ok=True)
    (DATASET / "sparse").mkdir(parents=True, exist_ok=True)

    records = []
    for name in names_all:
        with Image.open(src_images / name) as im:
            w, h = im.size
            if name in keep:
                small = im.resize((TARGET_WIDTH, round(h * TARGET_WIDTH / w)), Image.LANCZOS)
                small.save(DATASET / "images" / name, format="JPEG", quality=JPEG_QUALITY)
                records.append({
                    "file": name,
                    "sha256": _sha256(DATASET / "images" / name),
                    "width": TARGET_WIDTH,
                    "height": round(h * TARGET_WIDTH / w),
                })

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["committed_subset"]["images"] = records
    manifest["committed_subset"]["image_count"] = len(records)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Sparse reference: filter to kept image ids.
    cameras = (src_sparse / "cameras.txt").read_text(encoding="utf-8")
    images_txt = (src_sparse / "images.txt").read_text(encoding="utf-8")
    points_txt = (src_sparse / "points3D.txt").read_text(encoding="utf-8")
    filtered_images = _filter_images_txt(images_txt, keep)
    keep_ids = {int(l.split()[0]) for l in filtered_images.splitlines() if l.split() and l.split()[0].isdigit()}
    filtered_points = _filter_points3d(points_txt, keep_ids)
    (DATASET / "sparse" / "cameras.txt").write_text(cameras, encoding="utf-8")
    (DATASET / "sparse" / "images.txt").write_text(filtered_images, encoding="utf-8")
    (DATASET / "sparse" / "points3D.txt").write_text(filtered_points, encoding="utf-8")
    print(f"regenerated: {len(records)} committed images, sparse filtered to {len(keep_ids)} image ids")


def regenerate() -> int:
    """Locate an original extraction on disk or download one."""
    candidates = [
        Path("C:/Users/shres/OneDrive/Desktop/codes/reality-engine/_datasets_dl/south-building-extract/south-building"),
    ]
    for cand in candidates:
        if (cand / "images").is_dir():
            _regenerate_from(cand)
            return 0
    return download()


def verify() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    subset = manifest["committed_subset"]["images"]
    failures = []
    for rec in subset:
        p = DATASET / "images" / rec["file"]
        if not p.is_file():
            failures.append(f"missing: {rec['file']}")
            continue
        got = _sha256(p)
        if got != rec["sha256"]:
            failures.append(f"sha256 mismatch: {rec['file']}")
    sparse_files = ["cameras.txt", "images.txt", "points3D.txt"]
    for f in sparse_files:
        if not (DATASET / "sparse" / f).is_file():
            failures.append(f"missing sparse reference: {f}")
    if failures:
        print("VERIFY FAILED:")
        for f in failures:
            print(" ", f)
        return 1
    print(f"verify OK: {len(subset)} images match MANIFEST.json; sparse reference present")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "download":
        return download()
    if cmd == "verify":
        return verify()
    if cmd == "regenerate":
        return regenerate()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
