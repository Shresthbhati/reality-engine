"""Check hand-written label files before they are scored (build-plan task 8.6).

    python -m benchmarks.scene_complete.check [labels_dir] [images_dir]

Catches the mistakes that silently corrupt a benchmark: a typo in a class name (it can never match, so every such object
reads as a miss), a box outside the image or with no area, a duplicate object, an unknown tier or support, a label file
with no image, and an image with no label file. Exit code 0 only when there are no errors; unknown-but-plausible class
names are warnings (the scorer still accepts them).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from benchmarks.scene_complete.run import IMAGE_SUFFIXES, VOCABULARY
from benchmarks.scene_complete.score import TIERS

SOURCES = ("ai", "real", "synthetic")
SUPPORTS = ("floor", "wall", "ceiling", "table", "shelf", "other", "unknown")


def check(labels_dir: Path, images_dir: Path | None = None) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    seen_stems = set()
    sizes: Dict[str, tuple] = {}
    if images_dir is not None and Path(images_dir).exists():
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover
            Image = None  # type: ignore[assignment]
        for p in Path(images_dir).iterdir():
            if p.suffix in IMAGE_SUFFIXES and Image is not None:
                with Image.open(p) as im:
                    sizes[p.stem] = im.size
    for lab in sorted(Path(labels_dir).glob("*.json")):
        seen_stems.add(lab.stem)
        try:
            doc = json.loads(lab.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{lab.name}: not valid JSON ({exc})")
            continue
        if doc.get("source") not in SOURCES:
            errors.append(f"{lab.name}: source must be one of {SOURCES}, got {doc.get('source')!r}")
        if images_dir is not None and sizes and lab.stem not in sizes:
            errors.append(f"{lab.name}: no image named {lab.stem}.* in {images_dir}")
        objs = doc.get("objects")
        if not isinstance(objs, list):
            errors.append(f"{lab.name}: 'objects' must be a list")
            continue
        boxes = set()
        for i, o in enumerate(objs):
            where = f"{lab.name} object {i}"
            cls = o.get("class")
            if not isinstance(cls, str) or cls != cls.strip().lower() or " " in cls:
                errors.append(f"{where}: class must be lowercase_with_underscores, got {cls!r}")
            elif cls not in VOCABULARY:
                warnings.append(f"{where}: class {cls!r} is not in the furniture vocabulary (fine for infrastructure/decor)")
            if o.get("tier", "furniture") not in TIERS:
                errors.append(f"{where}: tier must be one of {TIERS}, got {o.get('tier')!r}")
            if "support" in o and o["support"] not in SUPPORTS:
                errors.append(f"{where}: support must be one of {SUPPORTS}, got {o['support']!r}")
            box = o.get("box")
            if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
                errors.append(f"{where}: box must be [x0, y0, x1, y1] numbers")
                continue
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                errors.append(f"{where}: box has no area ({box})")
            size = sizes.get(lab.stem)
            if size and (x0 < 0 or y0 < 0 or x1 > size[0] or y1 > size[1]):
                errors.append(f"{where}: box {box} outside the {size[0]}x{size[1]} image")
            key = (cls, tuple(box))
            if key in boxes:
                errors.append(f"{where}: duplicate of an earlier object")
            boxes.add(key)
    if sizes:
        for stem in sorted(set(sizes) - seen_stems):
            errors.append(f"image {stem}: no label file {stem}.json")
    return {"errors": errors, "warnings": warnings, "label_files": [str(len(seen_stems))]}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    base = Path(__file__).parent
    res = check(Path(argv[0]) if argv else base / "labels", Path(argv[1]) if len(argv) > 1 else base / "images")
    for w in res["warnings"]:
        print("warning:", w)
    for e in res["errors"]:
        print("ERROR:", e)
    print(f"{res['label_files'][0]} label files, {len(res['errors'])} errors, {len(res['warnings'])} warnings")
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
