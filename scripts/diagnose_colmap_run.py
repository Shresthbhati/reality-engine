"""Diagnose a COLMAP session: `PYTHONPATH=. python scripts/diagnose_colmap_run.py <session_dir> [--json out.json]`.

Prints which photos registered, why the others did not, pair/inlier/coverage statistics, triangulation angles, intrinsics, and capture guidance.
"""

from __future__ import annotations

import argparse
import json
import sys

from reconstruction.registration_diagnosis import capture_guidance, diagnose


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rep = diagnose(a.session_dir)
    rep["capture_guidance"] = capture_guidance(rep)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    slim = {k: v for k, v in rep.items() if k != "images"}
    slim["components"] = [{**c, "images": f"{len(c['images'])} images"} for c in rep["components"]]
    json.dump(slim, sys.stdout, indent=1)
    print("\n--- unregistered images")
    for i in rep["images"]:
        if not i["registered"]:
            print(i["name"], "| kp", i["keypoints"], "| verified-nbrs", i["verified_neighbours"], "| link", i["strongest_link_to_registered"], "|", i["why_unregistered"])
    print("--- registered: points observed / mean track err")
    for i in rep["images"]:
        if i["registered"]:
            print(i["name"], i["points3d_observed"], i.get("mean_track_error_px"), "nbrs", i["verified_neighbours"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
