"""One status model for what has actually been verified, and against what.

Four honest levels, used identically by the API (``GET /api/system/verification``), the CLI (``reality verification``),
the ledger (``.agent/CAPABILITIES.yaml`` is checked against this registry by a test) and the reports:

    IMPLEMENTED                      code exists with unit tests; no data-based verification is claimed
    SYNTHETICALLY VERIFIED           verified on GENERATED data (the synthetic pack, fixtures, fake CLIs)
    REAL-WORLD VERIFIED              verified on REAL data, hardware or a real service
    EXTERNAL VERIFICATION PENDING    not a level: a capability may carry pending items that need something this
                                     environment does not have (a device, a built backend, a database server)

``level`` is the HIGHEST level genuinely reached; ``pending`` lists what is still outstanding and is shown next to it,
verbatim (e.g. "SYNTHETIC RGB-D VERIFIED -- PHYSICAL DEVICE VERIFICATION PENDING"). The rules the tests enforce:

  * a capability below REAL-WORLD can never claim real data; REAL-WORLD must name its ``real_data``;
  * synthetic labels contain the word SYNTHETIC -- a synthetic result is never worded as real;
  * every capability cites evidence files that exist in the repository;
  * ``scope`` says whether it belongs to the working prototype (core) or is optional / deferred.

``runtime_availability()`` is a different question -- 'can THIS machine run it right now' -- answered by probing only
(no model loads, no network, no database): AVAILABLE, UNAVAILABLE (not installed) or FAILED (installed but broken).
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Verification(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"
    SYNTHETICALLY_VERIFIED = "SYNTHETICALLY VERIFIED"
    REAL_WORLD_VERIFIED = "REAL-WORLD VERIFIED"


_ORDER = {Verification.IMPLEMENTED: 0, Verification.SYNTHETICALLY_VERIFIED: 1, Verification.REAL_WORLD_VERIFIED: 2}
EXTERNAL_PENDING = "EXTERNAL VERIFICATION PENDING"


@dataclass(frozen=True)
class Capability:
    id: str
    name: str
    level: Verification
    #: the headline label, in the words the project uses for it
    label: str
    scope: str = "core"                       # core | optional | deferred
    #: outstanding verification that needs something this environment lacks (verbatim phrases)
    pending: Tuple[str, ...] = ()
    #: test files that carry the verification (must exist)
    evidence: Tuple[str, ...] = ()
    #: for REAL-WORLD VERIFIED: the data / hardware / service it was verified against
    real_data: str = ""
    note: str = ""
    ledger: Tuple[str, ...] = ()              # TASKS.yaml ids this capability belongs to

    def __post_init__(self):
        if self.level != Verification.REAL_WORLD_VERIFIED and self.real_data:
            raise ValueError(f"{self.id}: only REAL-WORLD VERIFIED may name real data")
        if self.level == Verification.REAL_WORLD_VERIFIED and not self.real_data:
            raise ValueError(f"{self.id}: REAL-WORLD VERIFIED must name the real data it was verified on")
        if self.level == Verification.SYNTHETICALLY_VERIFIED and "SYNTHETIC" not in self.label.upper() \
                and "FAKE" not in self.label.upper():
            raise ValueError(f"{self.id}: a synthetic verification must be worded as synthetic: {self.label!r}")
        if self.scope not in ("core", "optional", "deferred"):
            raise ValueError(f"{self.id}: unknown scope {self.scope!r}")

    @property
    def external_pending(self) -> bool:
        return bool(self.pending)

    @property
    def status_line(self) -> str:
        return " -- ".join([self.label, *self.pending])

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "level": self.level.value, "label": self.label, "scope": self.scope,
                "external_verification_pending": self.external_pending, "pending": list(self.pending),
                "status_line": self.status_line, "evidence": list(self.evidence), "real_data": self.real_data,
                "note": self.note, "ledger": list(self.ledger)}


SOUTH_BUILDING = "South Building (128 real photographs of a university building), real COLMAP 4.2.0 + MiDaS, CUDA"

CAPABILITIES: Tuple[Capability, ...] = (
    Capability("sparse_reconstruction", "Photos -> registered cameras + sparse geometry (COLMAP)",
               Verification.REAL_WORLD_VERIFIED, "REAL-WORLD VERIFIED (South Building)",
               pending=("INDOOR ROOM / CORRIDOR PHOTO SET NOT AVAILABLE",),
               evidence=("tests/integration/test_progressive_product_journey.py",),
               real_data=SOUTH_BUILDING, ledger=("PROD-01",)),
    Capability("progressive_world", "More evidence -> same world -> new immutable version",
               Verification.REAL_WORLD_VERIFIED, "REAL-WORLD VERIFIED (South Building)",
               pending=("INDOOR ROOM / CORRIDOR PHOTO SET NOT AVAILABLE",),
               evidence=("tests/integration/test_progressive_product_journey.py",
                         "tests/integration/test_continuous_evidence.py", "tests/integration/test_demo_journey.py"),
               real_data=SOUTH_BUILDING, ledger=("PROD-01",)),
    Capability("candidate_arbitration", "Incremental vs full candidate arbitration, frame preservation",
               Verification.REAL_WORLD_VERIFIED, "REAL-WORLD VERIFIED (South Building candidates)",
               evidence=("tests/test_candidate_selection.py", "tests/integration/test_continuous_evidence.py"),
               real_data=SOUTH_BUILDING, ledger=("PROD-01",)),
    Capability("failure_safety", "Crash-safe adoption: stage-synchronised kills, safe retry, no duplicate versions",
               Verification.REAL_WORLD_VERIFIED, "REAL-WORLD VERIFIED (real engine, 8 killed stages)",
               evidence=("tests/test_adoption_transaction.py", "tests/integration/test_reliability_journey.py"),
               real_data="a real uvicorn process running real COLMAP on South Building, killed at 8 pipeline boundaries",
               ledger=("PROD-01",)),
    Capability("dense_reconstruction", "Automatic dense MVS with quality arbitration against the sparse world",
               Verification.REAL_WORLD_VERIFIED, "REAL-WORLD VERIFIED (South Building, CUDA)",
               evidence=("tests/test_dense_arbitration.py", "tests/test_dense_gate.py"),
               real_data=SOUTH_BUILDING, ledger=("P6-01",)),
    Capability("detail_budget", "Scale-aware detail budget (metric GSD vs relative-scale voxel)",
               Verification.SYNTHETICALLY_VERIFIED, "SYNTHETICALLY VERIFIED (thresholds measured on one real dataset)",
               pending=("CALIBRATION ON MORE THAN ONE REAL DATASET PENDING",),
               evidence=("tests/test_detail_scale.py", "tests/test_detail_budget.py"),
               note="thresholds are provisional/empirical, not universal: docs/engineering/DETAIL_CALIBRATION.md",
               ledger=("P7-05",)),
    Capability("rgbd", "RGB-D depth sidecar -> metric points -> world", Verification.SYNTHETICALLY_VERIFIED,
               "SYNTHETIC RGB-D VERIFIED", pending=("PHYSICAL DEVICE VERIFICATION PENDING",),
               evidence=("tests/test_synthetic_rgbd.py", "tests/test_depth_frames.py"), ledger=("P1-03",)),
    Capability("vio", "VIO trajectory -> clock normalisation -> frame -> WorldIR", Verification.SYNTHETICALLY_VERIFIED,
               "SYNTHETIC VIO PIPELINE VERIFIED", pending=("REAL BACKEND / HARDWARE VERIFICATION PENDING",),
               evidence=("tests/test_synthetic_vio.py", "tests/test_trajectory_backend.py"), ledger=("P3-02",)),
    Capability("indoor_architecture", "Rooms, storeys, openings, floors/ceilings from geometry",
               Verification.SYNTHETICALLY_VERIFIED, "SYNTHETIC INDOOR VERIFIED",
               pending=("REAL INDOOR DATASET VERIFICATION PENDING",),
               evidence=("tests/test_synthetic_indoor.py", "tests/test_multistorey_matrix.py"),
               note="known limits are pinned as strict xfail in tests/test_synthetic_indoor.py", ledger=("P7-03",)),
    Capability("export_core", "glTF, USD(A), IFC, CityGML, CityJSON, Blender exports", Verification.REAL_WORLD_VERIFIED,
               "REAL-WORLD VERIFIED (a world reconstructed from real photographs, parsed back with independent readers)",
               pending=("OPENING THE FILES IN BLENDER / USD / BIM VIEWERS NOT VERIFIED",),
               evidence=("tests/test_core_exports.py", "tests/test_export_route.py",
                         "tests/integration/test_demo_journey.py"),
               real_data="the demo journey's world, reconstructed from 20 real South Building photographs; all six exports "
                         "produced through the API and parsed back (json / XML / IfcOpenShell / ast)",
               ledger=("P16-01",)),
    Capability("export_ecosystem", "GeoJSON, Gazebo SDF, Godot, Unreal, SUMO, Habitat writers",
               Verification.SYNTHETICALLY_VERIFIED, "SYNTHETICALLY VERIFIED (structure and axis conventions only)",
               scope="optional", pending=("TARGET RUNTIME ACCEPTANCE (Gazebo, Godot, Unreal, SUMO, habitat-sim) PENDING",),
               evidence=("tests/test_box_exporters.py",), ledger=("P16-02",)),
    Capability("citydb", "3DCityDB export through the official citydb CLI", Verification.SYNTHETICALLY_VERIFIED,
               "SYNTHETICALLY VERIFIED (contract tests against a FAKE citydb CLI)",
               pending=("REAL 3DCITYDB (PostgreSQL/PostGIS) VERIFICATION PENDING",),
               evidence=("tests/test_citydb_exporter.py",), ledger=("P14-01",)),
)


def by_id() -> Dict[str, Capability]:
    return {c.id: c for c in CAPABILITIES}


def report() -> dict:
    """The whole registry as JSON-ready data, with a per-level tally."""
    tally: Dict[str, int] = {v.value: 0 for v in Verification}
    for c in CAPABILITIES:
        tally[c.level.value] += 1
    return {"levels": [v.value for v in Verification], "external_pending_label": EXTERNAL_PENDING,
            "tally": tally, "capabilities": [c.to_dict() for c in CAPABILITIES]}


# ------------------------------------------------------------------------------------------------------ runtime

@dataclass(frozen=True)
class Availability:
    name: str
    status: str                  # AVAILABLE | UNAVAILABLE | FAILED
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _module(name: str, label: str) -> Availability:
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        found = False
    return Availability(label, "AVAILABLE" if found else "UNAVAILABLE",
                        "" if found else f"python module {name!r} is not installed")


def runtime_availability() -> List[Availability]:
    """Probe-only: what this machine can run now. 3DCityDB reports AVAILABLE only when the ``citydb`` CLI starts; the
    database behind it is not contacted (it is known only at export time)."""
    out: List[Availability] = []
    colmap = shutil.which("colmap")
    out.append(Availability("colmap", "AVAILABLE" if colmap else "UNAVAILABLE",
                            colmap or "`colmap` is not on PATH"))
    out.append(_module("ifcopenshell", "ifcopenshell (IFC export)"))
    out.append(_module("torch", "torch (MiDaS / Mask R-CNN)"))
    try:
        from exporters.citydb import probe_citydb

        p = probe_citydb()
        out.append(Availability("3dcitydb (citydb CLI)", p.status, p.detail))
    except Exception as exc:  # noqa: BLE001 -- a probe must report, not raise
        out.append(Availability("3dcitydb (citydb CLI)", "FAILED", f"{type(exc).__name__}: {exc}"))
    try:
        from trajectories.backend.selection import DEFAULT_BACKENDS

        for b in DEFAULT_BACKENDS:
            ok = b.is_available()
            out.append(Availability(f"vio:{b.name}", "AVAILABLE" if ok else "UNAVAILABLE",
                                    "" if ok else f"none of {getattr(b, 'binary_names', ())!r} on PATH"))
    except Exception as exc:  # noqa: BLE001
        out.append(Availability("vio", "FAILED", f"{type(exc).__name__}: {exc}"))
    return out
