"""Turn a folder of room photographs into predictions the scorer can read (build-plan tasks 8.6 / 8.7).

    python -m benchmarks.scene_complete.run <images_dir> [--out <predictions_dir>]
    python -m benchmarks.scene_complete.score <predictions_dir>

Uses the same COCO Mask R-CNN detector the single-image furniture path uses (engine/pipeline/single_image_furniture.py),
so the number the scorer reports is the 2D detection recall that path starts from. Boxes are in ORIGINAL pixel coordinates.
Only classes the benchmark knows (the furniture placer's vocabulary) are written; everything else the detector saw
(person, cup, ...) is counted in ``run_report.json`` so nothing disappears silently. A photo the detector cannot decode
gets NO prediction file, which the scorer counts as a miss, never a skip.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Sequence

from perception.instances.furniture_from_detections import SIZE_PRIORS, _canon

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
#: tier per class; everything else the furniture vocabulary knows is "furniture"
_TIER = {"tv": "appliance", "lamp": "appliance"}
#: classes a label file may use (the placer's canonical names)
VOCABULARY = frozenset(SIZE_PRIORS)

#: (image path) -> [(raw_label, score, (x0, y0, x1, y1) in original pixels)]
Detector = Callable[[Path], Sequence[tuple]]


def tier_of(cls: str) -> str:
    return _TIER.get(cls, "furniture")


def maskrcnn_detector(score_threshold: float = 0.5) -> Detector:
    """The real detector, loaded once. Raises if the model/weights are unavailable (never a silent empty result)."""
    from perception.detection.maskrcnn_backend import MaskRCNNDetector

    model = MaskRCNNDetector(score_threshold=score_threshold, device="cpu")

    def detect(path: Path):
        item = SimpleNamespace(id=path.stem, source_uri=path.resolve().as_uri(), evidence_id=path.stem)
        results = model.detect([item])
        if not results:
            raise RuntimeError(f"detector could not decode {path.name}")
        return [(d.label, float(d.score), tuple(float(v) for v in d.box)) for d in results[0].detections]

    detect.unload = model.unload  # type: ignore[attr-defined]
    return detect


def run(images: Iterable[Path], out_dir: Path, detect: Detector, source: str = "real") -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dropped: Dict[str, int] = {}
    written = failed = kept = 0
    failures: List[dict] = []
    for img in sorted(images):
        try:
            raw = detect(img)
        except Exception as exc:  # noqa: BLE001 -- recorded; the missing file is scored as a miss
            failed += 1
            failures.append({"image": img.name, "error": str(exc)[-300:]})
            continue
        objects = []
        for label, score, box in raw:
            cls = _canon(label)
            if cls is None:
                dropped[label] = dropped.get(label, 0) + 1
                continue
            objects.append({"class": cls, "tier": tier_of(cls), "box": [round(v, 1) for v in box], "score": round(score, 3)})
        (out_dir / f"{img.stem}.json").write_text(
            json.dumps({"image": img.name, "source": source, "objects": objects}, indent=1), encoding="utf-8"
        )
        written += 1
        kept += len(objects)
    report = {"images_written": written, "images_failed": failed, "objects_kept": kept,
              "dropped_labels_not_in_vocabulary": dict(sorted(dropped.items(), key=lambda kv: -kv[1])),
              "failures": failures}
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("images_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "predictions")
    ap.add_argument("--source", choices=["real", "ai", "synthetic"], default="real")
    args = ap.parse_args(argv)
    imgs = [p for p in args.images_dir.iterdir() if p.suffix in IMAGE_SUFFIXES]
    if not imgs:
        print(f"no images in {args.images_dir}")
        return 2
    detect = maskrcnn_detector()
    try:
        report = run(imgs, args.out, detect, source=args.source)
    finally:
        getattr(detect, "unload", lambda: None)()
    print(json.dumps(report, indent=1))
    return 0 if report["images_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
