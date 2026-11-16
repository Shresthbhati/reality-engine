"""Export a job's 2D detections for human review (CVAT / Label Studio / FiftyOne) and score reviewed labels against them.

    python scripts/export_annotation_review.py export --report report.json --images images.json --out review/
    # ... a human corrects review/coco.json in CVAT (or the Label Studio tasks) and exports COCO again ...
    python scripts/export_annotation_review.py score  --report report.json --reviewed corrected_coco.json --out score.json

``report.json`` is the version report (GET /api/worlds/{id}/report) or just the perception facts; ``images.json`` maps evidence id to
an image file. Only images present in the reviewed file are scored; no review means no numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import annotation_review as ar  # noqa: E402


def _perception(report: dict) -> dict:
    node = (report.get("stages", {}).get("perception") if "stages" in report else report) or {}
    if not node.get("detections_2d"):
        raise SystemExit("report has no perception.detections_2d (run a job with the grounded perception stage first)")
    return node


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--report", required=True)
    e.add_argument("--images", required=True, help="JSON {evidence_id: image path}")
    e.add_argument("--out", required=True)
    e.add_argument("--fiftyone", action="store_true", help="also build a FiftyOne dataset (needs fiftyone installed)")
    s = sub.add_parser("score")
    s.add_argument("--report", required=True)
    s.add_argument("--reviewed", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--iou", type=float, default=0.5)
    s.add_argument("--exported", help="the coco.json this review started from; adds precision of the second-opinion / small-fitting proposals")
    a = p.parse_args(argv)

    perception = _perception(json.loads(Path(a.report).read_text(encoding="utf-8")))
    det = perception["detections_2d"]
    if a.cmd == "export":
        images = {k: Path(v) for k, v in json.loads(Path(a.images).read_text(encoding="utf-8")).items()}
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        coco = ar.coco_from_perception(perception, images)      # detections + second-opinion boxes + what the filter dropped
        (out / "coco.json").write_text(json.dumps(coco, indent=1), encoding="utf-8")
        sizes = {im["evidence_id"]: (im["width"], im["height"]) for im in coco["images"]}
        tasks, config = ar.to_label_studio(det, {k: v.as_uri() for k, v in images.items()}, sizes)
        (out / "label_studio_tasks.json").write_text(json.dumps(tasks, indent=1), encoding="utf-8")
        (out / "label_studio_config.xml").write_text(config, encoding="utf-8")
        if a.fiftyone:
            ar.to_fiftyone(coco, Path(next(iter(images.values()))).parent)
        print(f"wrote {len(coco['images'])} images, {len(coco['annotations'])} predicted boxes to {out}")
        return 0
    report = ar.score(det, ar.read_coco(a.reviewed), iou_min=a.iou)
    if getattr(a, "exported", None):                      # the file the review started from: also score the extra sources
        reviewed_coco = json.loads(Path(a.reviewed).read_text(encoding="utf-8"))
        report["extra_sources"] = ar.score_sources(json.loads(Path(a.exported).read_text(encoding="utf-8")), reviewed_coco, iou_min=a.iou)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_label"}), "\n", json.dumps(report["per_label"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
