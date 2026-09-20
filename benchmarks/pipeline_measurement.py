#!/usr/bin/env python3
"""Production pipeline measurement over the committed real dataset
(REAL_RECONSTRUCTION_PERCEPTION_CITY_READY increment D).

Measures the full canonical chain on the REAL south-building
photographs through the NEW batch path (reconstruction/batch.py):

    frames -> accepted/rejected (admission) -> reconstruction
    -> registration -> coverage -> uncertainty -> provenance
    -> runtime -> memory (per stage, tracemalloc)

Every stage records measured facts; nothing is fabricated. The
reconstruction backend is the REAL COLMAP binary when available; when
it is not, the record states BACKEND_UNAVAILABLE with the probe detail
and the non-COLMAP stages are still measured. This script is not a
test double and does not silently fall back to one.

Output: a measured record JSON alongside the E2E run records.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASET = REPO / "datasets" / "south_building"
OSM_DATASET = REPO / "datasets" / "city_osm" / "south_building_campus.osm"


def _stage(name: str, fn):
    """Run one stage with tracemalloc; return (result, stage_record)."""
    tracemalloc.start()
    started = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    record = {
        "runtime_s": round(elapsed, 3),
        "peak_python_alloc_mb": round(peak / (1024 * 1024), 3),
    }
    print(f"[{name}] {record}")
    return result, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true",
                        help="enable COLMAP GPU (CUDA build required)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="limit images (subset measurement)")
    parser.add_argument("--skip-colmap", action="store_true",
                        help="measure only the non-COLMAP stages")
    args = parser.parse_args()

    from evidence.importers import import_folder
    from evidence.packages import DeterministicPackageBuilder
    from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
    from reconstruction.batch import run_batch

    measurement: dict = {
        "measurement": "production-pipeline (batch path, real dataset)",
        "dataset": "south-building (committed 32-image REAL subset)",
        "python": platform.python_version(),
        "stages": {},
    }

    # ---- Stage 1: real ingestion -------------------------------------
    def ingest():
        builder = DeterministicPackageBuilder(seed="pipeline-measure")
        report = import_folder(builder, str(DATASET / "images"), on_error="record")
        items = builder.build().to_evidence_items()
        if args.max_images:
            items = items[: args.max_images]
        return builder, report, items

    (builder, import_report, items), s1 = _stage("ingest", ingest)
    measurement["stages"]["ingest"] = {
        **s1,
        "files_imported": len(import_report.imported),
        "duplicates_skipped": len(import_report.duplicates_skipped),
        "failed_paths": len(import_report.failed_paths),
        "frames": len(items),
    }

    # ---- Stage 2: classification + admission -------------------------
    from reconstruction.robustness import classify_evidence_items
    from reconstruction.robustness_admission import admit_for_reconstruction

    def admit():
        classified = classify_evidence_items(items)
        admitted, decision = admit_for_reconstruction(items)
        return classified, admitted, decision

    (classified, admitted, decision), s2 = _stage("admission", admit)
    outcome_counts: dict = dict(classified.counts)  # measured per-item outcomes
    measurement["stages"]["admission"] = {
        **s2,
        "frames": len(items),
        "admitted": len(admitted),
        "excluded": len(items) - len(admitted),
        "exclusion_reasons": decision.to_dict().get("excluded", [])[:8],
        "item_outcomes_measured": outcome_counts,
        "sample_reasons": [a.to_dict() for a in classified.admissions[:3]],
    }

    # ---- Stage 3: reconstruction via the batch path (real backend) ---
    stages = measurement["stages"]
    if args.skip_colmap:
        stages["reconstruction"] = {
            "status": "SKIPPED_BY_FLAG",
            "reason": "--skip-colmap: reconstruction stage not measured",
        }
    else:
        backend = ColmapReconstructionBackend(colmap_binary="colmap", use_gpu=args.gpu)
        probe = getattr(backend, "availability_probe", None)
        ok, detail = probe() if probe else (True, "no probe")
        if not ok:
            measurement["backend"] = "BACKEND_UNAVAILABLE"
            measurement["backend_detail"] = detail
            stages["reconstruction"] = {
                "status": "BACKEND_UNAVAILABLE",
                "reason": detail,
                "note": "non-COLMAP stages above are still measured",
            }
        else:
            def reconstruct():
                return run_batch(
                    items,
                    prefixes=("images",),  # the dataset's single real session
                    backends=[backend],
                    reference_session="images",
                    align=True,
                )

            report, s3 = _stage("reconstruction", reconstruct)
            stages["reconstruction"] = {**s3, **report}
            # Coverage + uncertainty (measured from the batch report).
            s = report["sessions"].get("images", {})
            stages["coverage"] = {
                "points": s.get("points", 0),
                "frames_submitted": s.get("frame_count", 0),
                "registration": report["registration"],
            }
            measurement["provenance"] = {
                "backend": s.get("backend", ""),
                "admission": "reconstruction.robustness_admission.admit_for_reconstruction",
                "batch_path": "reconstruction.batch.run_batch",
                "dataset_manifest": "datasets/south_building/MANIFEST.json",
            }

    # ---- Stage 4: city ingestion boundary (real OSM extract) ---------
    if OSM_DATASET.is_file():
        from evidence.city_import import import_osm_xml
        from evidence.packages import EvidenceSource

        def city():
            b = DeterministicPackageBuilder(seed="pipeline-measure-osm")
            src = EvidenceSource(
                source_id="src-osm-overpass", platform="external_map_data",
                device="OpenStreetMap/Overpass",
                notes="REAL OSM extract (ODbL), see datasets/city_osm/MANIFEST.json",
            )
            return import_osm_xml(b, src, str(OSM_DATASET))

        city_report, s4 = _stage("city_ingestion", city)
        stages["city_ingestion"] = {
            **s4,
            "features": city_report.imported_features,
            "nodes_seen": city_report.nodes_seen,
            "unresolved_node_refs": city_report.unresolved_node_refs,
        }
    else:
        stages["city_ingestion"] = {"status": "FIXTURE_MISSING"}

    measurement["totals"] = {
        "frames": len(items),
        "admitted": len(admitted),
        "outcome_counts_measured": outcome_counts,
    }

    out_dir = DATASET / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out = out_dir / f"pipeline_measurement_{stamp}.json"
    out.write_text(json.dumps(measurement, indent=2, default=str), encoding="utf-8")
    print(f"record written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
