"""Pre-label images with the taxonomy's open-vocabulary prompts: a DRAFT for human review, never ground truth.

  python scripts/prelabel_images.py OUT_DIR image1.png image2.png ... [--tiles] [--min-score 0.30]

Writes OUT_DIR/draft_inventory.json (records follow benchmarks/reality_benchmark/inventory.py, every record
``review_status: "unreviewed"``, ``instance_id: null``) and one overlay PNG per image. A reviewer keeps, edits or deletes
each record and assigns ``instance_id`` to physical objects that appear in several photographs; only then is it truth.
The same draft doubles as the baseline detector's predictions. Boxes only: no masks, 3D, material or occlusion."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INTERIOR_CATEGORIES = ("furniture", "electrical", "hvac", "plumbing", "opening", "circulation", "vegetation", "structural", "finish", "misc")
MAX_BOX_SHARE = 0.80
NMS_IOU = 0.5
CROSS_CLASS_IOU = 0.8
UNAVAILABLE = {k: "pre-label draft: box only" for k in ("instance_count", "mask", "position_3d", "dimensions", "orientation",
                                                         "material", "visibility", "evidence")}


def _iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _nms(dets):
    """dets: (class_id, score, box, pass). Highest score first; drop same-class overlaps and near-identical other-class boxes."""
    kept = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if any(_iou(k[2], d[2]) > (NMS_IOU if k[0] == d[0] else CROSS_CLASS_IOU) for k in kept):
            continue
        kept.append(d)
    return kept


#: a socket, switch, tap or handle is small: a box over this share of the frame is a tile-sized false positive, not a fitting.
#: Linear classes (wiring, conduit, pipe, trunking) and stairs/vents are exempt: they are legitimately long or wide.
SMALL_FITTING_MAX_SHARE = 0.02
_SMALL_FITTING_WORDS = ("handle", "hinge", "lock", "latch", "socket", "switch", "dimmer", "plug", "port", "plate", "tap", "faucet",
                        "drain", "thermostat", "junction", "breaker", "bulb", "fitting")


def plausible_size(cls, box, size) -> bool:
    """False when ``cls`` is a small fitting and ``box`` is far larger than any fitting could appear."""
    label = cls.label.lower()
    if cls.category in ("electrical", "plumbing", "opening") and cls.small and any(w in label for w in _SMALL_FITTING_WORDS):
        return (box[2] - box[0]) * (box[3] - box[1]) / float(size[0] * size[1]) <= SMALL_FITTING_MAX_SHARE
    return True


def _tiles(w, h, tile=640, overlap=0.25):
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(1, w - tile + step), step)) or [0]
    ys = list(range(0, max(1, h - tile + step), step)) or [0]
    return [(x, y, min(w, x + tile), min(h, y + tile)) for y in ys for x in xs]


def prelabel_image(path, prompt_class, small_prompts, *, tiles, min_score):
    from PIL import Image

    from engine.pipeline.open_vocab_furniture import _grounded_raw

    img = Image.open(path).convert("RGB")
    w, h = img.size
    dets = []
    raw, _ = _grounded_raw(pathlib.Path(path), list(prompt_class))
    dets += [(prompt_class[p], s, b, "coarse") for p, s, b in raw if s >= min_score]
    if tiles and small_prompts:
        with tempfile.TemporaryDirectory() as td:
            for k, (x0, y0, x1, y1) in enumerate(_tiles(w, h)):
                tp = pathlib.Path(td) / f"t{k}.png"
                img.crop((x0, y0, x1, y1)).save(tp)
                traw, _ = _grounded_raw(tp, list(small_prompts))
                dets += [(prompt_class[p], s, (b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0), "fine") for p, s, b in traw if s >= min_score]
    from perception.entities import taxonomy

    out = []
    for cid, s, b, ps in _nms(dets):
        share = (b[2] - b[0]) * (b[3] - b[1]) / float(w * h)
        if share <= MAX_BOX_SHARE and b[2] > b[0] and b[3] > b[1] and plausible_size(taxonomy.get(cid), b, (w, h)):
            out.append((cid, s, b, ps))
    return out, (w, h)


def _emit(stem, path, dets, size, records, out_dir):
    """Append draft records for ``dets`` and write the overlay PNG."""
    from PIL import Image, ImageDraw

    from perception.entities import taxonomy

    img = Image.open(path).convert("RGB")
    dr = ImageDraw.Draw(img)
    for k, (cid, s, b, ps) in enumerate(sorted(dets, key=lambda d: (d[2][1], d[2][0]))):
        cls = taxonomy.get(cid)
        records.append({"image_id": stem, "entity_id": f"draft-{stem}-{k:03d}", "category": cls.category, "subtype": cls.label,
                        "class_id": cid, "bbox": [round(v, 1) for v in b], "confidence": round(float(s), 3), "instance_id": None,
                        "review_status": "unreviewed", "detected_in_pass": ps, "image_size": list(size),
                        "unavailable": dict(UNAVAILABLE), **{m: None for m in UNAVAILABLE}})
        dr.rectangle(b, outline=(255, 40, 40) if cls.small else (40, 200, 60), width=3 if cls.small else 2)
        dr.text((b[0] + 2, b[1] + 1), f"{k}:{cls.label} {s:.2f}", fill=(255, 255, 0))
    img.save(pathlib.Path(out_dir) / f"{stem}_overlay.png")


def refilter(draft_path, out_dir) -> int:
    """Re-apply the size filter to saved drafts (no model run) and redraw the overlays."""
    from perception.entities import taxonomy

    d = json.loads(pathlib.Path(draft_path).read_text(encoding="utf-8"))
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, before = [], len(d["records"])
    for stem, info in d["images"].items():
        dets = [(r["class_id"], r["confidence"], tuple(r["bbox"]), r["detected_in_pass"]) for r in d["records"]
                if r["image_id"] == stem and plausible_size(taxonomy.get(r["class_id"]), r["bbox"], r["image_size"])]
        size = next((r["image_size"] for r in d["records"] if r["image_id"] == stem), (0, 0))
        _emit(stem, info["source"], dets, size, records, out_dir)
        info["records"] = len(dets)
    d["records"], d["refiltered"] = records, f"size filter applied: {before} -> {len(records)} records"
    (out_dir / "draft_inventory.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(d["refiltered"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--refilter", metavar="DRAFT_JSON", help="re-apply the size filter to a saved draft (no model run)")
    ap.add_argument("--tiles", action="store_true", help="also run the small-class prompts on full-resolution tiles")
    ap.add_argument("--min-score", type=float, default=0.30)
    ap.add_argument("--categories", default=",".join(INTERIOR_CATEGORIES))
    args = ap.parse_args(argv)
    if args.refilter:
        return refilter(args.refilter, args.out_dir)
    if not args.images:
        ap.error("give image files, or --refilter DRAFT_JSON")

    from perception.entities import taxonomy

    cats = set(args.categories.split(","))
    prompt_class, small_prompts = {}, []
    for c in taxonomy.all_classes():
        if c.category in cats:
            for p in c.prompts:
                prompt_class.setdefault(p, c.id)
                if c.small:
                    small_prompts.append(p)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, per_image = [], {}
    for path in args.images:
        stem = pathlib.Path(path).stem[:8]
        t0 = time.time()
        dets, (w, h) = prelabel_image(path, prompt_class, sorted(set(small_prompts)), tiles=args.tiles, min_score=args.min_score)
        _emit(stem, path, dets, (w, h), records, out_dir)
        per_image[stem] = {"source": str(path), "records": len(dets), "seconds": round(time.time() - t0, 1)}
        print(f"{stem}: {len(dets)} drafts in {per_image[stem]['seconds']} s", flush=True)
    (out_dir / "draft_inventory.json").write_text(json.dumps(
        {"status": "DRAFT - not ground truth; every record is unreviewed", "model": "grounding-dino-tiny", "min_score": args.min_score,
         "tiles": args.tiles, "images": per_image, "records": records}, indent=1), encoding="utf-8")
    print(f"{len(records)} draft records -> {out_dir / 'draft_inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
