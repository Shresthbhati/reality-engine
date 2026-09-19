"""Large image-set robustness measurement (ROBUSTNESS directive P0).

Drives the REAL orchestrator input gate + admission classifier +
IReconstructionBackend contract at increasing capture sizes (20, 50,
100, 500, 1000 images) and records measured runtime and memory for each
stage. This is a measurement harness, not a test: run it directly.

    python benchmarks/robustness_scale.py [--sizes 20,50,100,500,1000]

BACKEND LABEL: the reconstruction stage below runs the repository's
FakeReconstructionBackend -- an explicit TEST DOUBLE that filters
canned results by evidence id. It exercises the orchestration/
classification/admission code paths faithfully but performs NO real
SfM; these numbers measure the pipeline's bookkeeping scale behavior,
never reconstruction quality. Real-model (COLMAP) validation is a
separate, explicitly-real path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
from typing import Dict, List

# Run-where-located: `python benchmarks/robustness_scale.py` must import
# THIS checkout's packages, not any site-packages copy (a stale
# `reconstruction` install exists on this machine and shadows the repo
# when the CWD is not already on sys.path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evidence.importers import EvidenceKind
from evidence.session import EvidenceItem
from reconstruction.backend.fake import FakeReconstructionBackend
from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint
from reconstruction.orchestrator import ReconstructionOrchestrator
from reconstruction.robustness import classify_evidence_items

#: Deterministic quality mix injected per batch: ~80% sharp, 10% soft
#: (DEGRADED), 5% hard-blurred (REJECTED), 5% metric-less (UNRESOLVED).
SHARP = {"laplacian_variance": 1200.0, "luma_mean": 128.0,
         "clipped_fraction": 0.01, "measured": 1.0}
SOFT = dict(SHARP, laplacian_variance=120.0)
BLURRY = dict(SHARP, laplacian_variance=5.0)


def _quality_for(index: int) -> Dict | None:
    phase = index % 20
    if phase < 16:
        return SHARP
    if phase < 18:
        return SOFT
    if phase == 18:
        return BLURRY
    return None


def build_batch(n: int) -> List[EvidenceItem]:
    return [
        EvidenceItem(
            id=f"img-{i}",
            kind=EvidenceKind.PHOTO,
            source_uri=f"file:///capture/img_{i:05d}.jpg",
            metadata={"quality": _quality_for(i)},
        )
        for i in range(n)
    ]


def measure(n: int) -> dict:
    record: dict = {"image_count": n}

    tracemalloc.start()
    t0 = time.perf_counter()
    batch = build_batch(n)
    record["build_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    report = classify_evidence_items(batch)
    record["classify_s"] = time.perf_counter() - t0
    record["outcomes"] = report.counts

    backend = FakeReconstructionBackend(
        canned_poses=[
            ReconstructedCameraPose(e.id, (float(i % 7), float(i // 7), 0.0), (1.0, 0.0, 0.0, 0.0))
            for i, e in enumerate(batch)
        ],
        canned_points=[
            ReconstructedPoint((0.1 * i, 0.2 * i, 0.3 * i), f"track-{i}", [batch[i].id])
            for i in range(n)
        ],
    )
    orchestrator = ReconstructionOrchestrator([backend])

    t0 = time.perf_counter()
    run = orchestrator.run(batch)
    record["orchestrate_s"] = time.perf_counter() - t0
    record["registration_status"] = run.result.registration_status
    record["poses"] = len(run.result.camera_poses)
    record["points"] = len(run.result.points)
    record["final_status"] = run.diagnostics.final_status

    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    record["peak_python_mb"] = round(peak / (1024 * 1024), 2)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="20,50,100,500,1000")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    print(f"{'images':>7} {'build_s':>9} {'classify_s':>11} "
          f"{'orchestrate_s':>14} {'peak_mb':>8} {'status':>8}  outcomes")
    records = []
    for n in sizes:
        r = measure(n)
        records.append(r)
        outcomes = r["outcomes"]
        print(f"{n:>7} {r['build_s']:>9.4f} {r['classify_s']:>11.4f} "
              f"{r['orchestrate_s']:>14.4f} {r['peak_python_mb']:>8.2f} "
              f"{r['registration_status']:>8}  "
              f"acc={outcomes['accepted']} deg={outcomes['degraded']} "
              f"rej={outcomes['rejected']} unres={outcomes['unresolved']} "
              f"fail={outcomes['failed']}")

    print(json.dumps({"records": records,
                      "backend": "FakeReconstructionBackend (TEST DOUBLE)"},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
