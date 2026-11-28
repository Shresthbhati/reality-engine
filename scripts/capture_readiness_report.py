"""CPU-only capture-readiness report for a folder of photographs (no COLMAP, no GPU).

    python scripts/capture_readiness_report.py DIR [--subset N] [--registered ID,ID,...] [--model SPARSE_TXT_DIR] [--out report.json]

``--registered`` / ``--model`` add the post-registration view (image ids are file stems). Prints the plain-language message and a
per-image diagnosis table; the full JSON goes to --out."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction import capture_readiness as cr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--subset", type=int)
    ap.add_argument("--registered")
    ap.add_argument("--model")
    ap.add_argument("--out")
    a = ap.parse_args()
    files = sorted(p for p in Path(a.dir).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    files = files[: a.subset] if a.subset else files
    rep = cr.assess_readiness([(p.stem, p) for p in files])
    if a.registered is not None or a.model:
        sparse = cr.load_colmap_text_model(a.model) if a.model else None
        reg = [s for s in (a.registered or "").split(",") if s] or (list(sparse["cameras"]) if sparse else [])
        rep = cr.after_registration(rep, reg, sparse)
    cr.store_report(rep, None)
    print(rep["user_message"], f"\nstatus={rep['status']} runtime={rep['timing']['runtime_s']}s components={rep['graph']['components']}")
    for r in rep["images"]:
        d = r["diagnosis"]
        print(f"{r['id']:>12} {r.get('size', '?'):>10} cause={d['cause']!s:<20} pred={d.get('predicted_registration')!s:<10} "
              f"links={d['evidence'].get('n_verified_links')} best={d['evidence'].get('best_inliers')} registered={r.get('registered')}")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
