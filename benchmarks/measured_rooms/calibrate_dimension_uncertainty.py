"""Empirically calibrate dimension uncertainty from per-axis error records exported by validate_office7_world.py (dimension_errors.json).

    PYTHONPATH=. python benchmarks/measured_rooms/calibrate_dimension_uncertainty.py <dimension_errors.json> [more.json ...]

Several files (different scenes) are pooled; the split is by SCENE when there are >= 2. The reference is the Kinect depth of a 7-Scenes scene
(an ESTIMATE with centimetre error, not a physical survey). Prints the report and writes dimension_uncertainty_calibration.json next to the
first input. ``state`` stays ``lower_bound_only`` unless the evidence supports calibrated intervals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from engine.multiengine.uncertainty_calibration import ErrorRecord, calibrate


def main(paths: list[Path]) -> int:
    rows = [r for p in paths for r in json.loads(p.read_text(encoding="utf-8"))]
    out = calibrate([ErrorRecord.from_row(r) for r in rows])
    out["reference"] = "Kinect depth of the dataset (estimate, centimetre-level), NOT a physical survey"
    (paths[0].parent / "dimension_uncertainty_calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    brief = {k: v for k, v in out.items() if k not in ("calibration_curve_held_out", "failure_cases")}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main([Path(a) for a in sys.argv[1:]]))
