"""Production batch reconstruction path (REAL_RECONSTRUCTION_PERCEPTION_
CITY_READY increment A): heterogeneous multi-session image sets become
canonical ReconstructionContract results through the EXISTING chain.

Composition, not reinvention -- every stage here is a canonical module:

    import_folder(on_error="record")             evidence.importers
      -> classify_evidence_items                 reconstruction.robustness
      -> admit_for_reconstruction                reconstruction.robustness_admission
      -> ReconstructionOrchestrator.run          reconstruction.orchestrator (per group)
      -> align_reconstructed_sessions            registration.cross_session
    -> summary report (JSON-sized, provenance-preserving)

No new reconstruction contract is defined here: groups produce ordinary
ReconstructionRun objects (the ReconstructionContract stage surface);
the batch layer only owns grouping, per-group failure isolation, bounded
submission, and cross-session registration.

Batch rules:
  - Session grouping is DECLARED (path components of source_uri), never
    guessed; items matching no declared prefix are recorded under
    "<unassigned>", never silently merged into another group.
  - A group below the orchestrator's minimum (or any backend refusing
    it) fails ALONE with a machine-readable reason; the rest of the
    batch continues. One bad session never loses the batch.
  - Bounded submission: MemoryBudget.max_frames_per_group caps what a
    single orchestrator run may receive; excess frames are recorded as
    deferred_capacity -- a measured fact, not a silent drop.
  - Unavailable backends surface their probe detail verbatim in the
    failure reason; nothing is reconstructed under a fake backend.
  - Outcome vocabulary is honest: "complete" only when every group
    reconstructed and every non-reference session registered; anything
    else is "degraded" with the causes visible in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from reconstruction.orchestrator import (
    EvidenceValidationError,
    ReconstructionOrchestrator,
)
from reconstruction.robustness_admission import (
    AdmissionError,
    admit_for_reconstruction,
)

UNASSIGNED = "<unassigned>"


@dataclass(frozen=True)
class MemoryBudget:
    """Hard submission bounds for one batch run.

    max_frames_per_group caps a single orchestrator run's input so
    per-run memory is bounded by the budget, not by capture size;
    frames beyond the cap are recorded as deferred_capacity per group.
    """

    max_frames_per_group: int = 2000


def _session_of(source_uri: str, prefixes: Sequence[str]) -> Optional[str]:
    """The declared session name if `prefixes` matches a path COMPONENT
    of source_uri (never a bare substring: 'session_a' must not match
    'session_a_backup')."""
    parts = source_uri.replace("\\", "/").split("/")
    for name in prefixes:
        if name in parts:
            return name
    return None


def group_by_session(
    items: Iterable, prefixes: Sequence[str]
) -> Dict[str, List]:
    """Deterministically partition evidence into declared session groups.

    Returns a dict keyed by matched prefix (sorted), plus "<unassigned>"
    ONLY when unassigned items exist -- absence of the key means there
    were none, presence with a non-empty list is a recorded fact.
    """
    prefixes = list(prefixes)
    groups: Dict[str, List] = {name: [] for name in prefixes}
    unassigned: List = []
    for item in items:
        session = _session_of(item.source_uri, prefixes)
        if session is None:
            unassigned.append(item)
        else:
            groups[session].append(item)
    ordered = {name: groups[name] for name in sorted(groups)}
    if unassigned:
        ordered[UNASSIGNED] = unassigned
    return ordered


def _unavailable_reason(backends: List) -> Optional[str]:
    """When every backend's availability probe declines, the exact probe
    detail IS the failure reason (UNAVAILABLE + reason, never a silent
    fallback). Backends without a probe are assumed available and are
    given their real chance by the orchestrator."""
    details: List[str] = []
    probed_any = False
    for backend in backends:
        probe = getattr(backend, "availability_probe", None)
        if probe is None:
            return None  # at least one backend gets its real chance
        probed_any = True
        try:
            ok, detail = probe()
        except Exception as exc:  # noqa: BLE001 - probe failures are data
            ok, detail = False, f"probe raised: {type(exc).__name__}: {exc}"
        if ok:
            return None
        name = getattr(backend, "backend_name", type(backend).__name__)
        details.append(f"{name}: {detail}")
    if probed_any and details:
        return "backend unavailable: " + "; ".join(details)
    return None


def run_batch(
    items: List,
    *,
    prefixes: Sequence[str],
    backends: List,
    reference_session: Optional[str] = None,
    memory_budget: MemoryBudget = MemoryBudget(),
    align: bool = True,
) -> dict:
    """Run the full batch: per-group admission -> orchestration ->
    cross-session registration -> honest summary report.

    Per-group failures (admission refusal, insufficient evidence,
    backend failure, unavailable backend) are isolated: the group is
    recorded failed with its reason and the batch continues. Returns a
    JSON-serializable dict; full ReconstructionRun objects are NOT
    retained (memory stays bounded by the summary, not by world size).
    """
    groups = group_by_session(items, prefixes)
    orchestrator = ReconstructionOrchestrator(backends)

    sessions: Dict[str, dict] = {}
    results: Dict[str, object] = {}
    totals = {"frames_submitted": 0, "points": 0, "deferred_capacity": 0}

    for name in sorted(groups):
        group_items = groups[name]
        entry: dict = {
            "frame_count": len(group_items),
            "status": "failed",
            "reason": "",
            "points": 0,
            "backend": "",
            "deferred_capacity": 0,
        }
        sessions[name] = entry

        # Bounded submission: over-budget frames are recorded, not run.
        if len(group_items) > memory_budget.max_frames_per_group:
            entry["deferred_capacity"] = (
                len(group_items) - memory_budget.max_frames_per_group
            )
            totals["deferred_capacity"] += entry["deferred_capacity"]
            group_items = group_items[: memory_budget.max_frames_per_group]

        try:
            unavailable = _unavailable_reason(backends)
            if unavailable is not None:
                entry["reason"] = unavailable
            else:
                admitted, _decision = admit_for_reconstruction(list(group_items))
                run = orchestrator.run(admitted)
                entry["status"] = run.diagnostics.final_status  # success|partial|failed
                entry["backend"] = run.diagnostics.backend_name
                entry["points"] = len(run.result.points)
                entry["frame_count"] = len(admitted)
                # Measured uncertainty summary (from the points' own
                # Uncertainty records) -- counts and spread, no invented
                # numbers; an uncertainty-less result records that fact.
                confs = [
                    p.uncertainty.confidence for p in run.result.points
                    if p.uncertainty is not None
                ]
                entry["uncertainty"] = (
                    {
                        "points_with_uncertainty": len(confs),
                        "confidence_mean": round(sum(confs) / len(confs), 4),
                        "confidence_min": round(min(confs), 4),
                    }
                    if confs
                    else {"points_with_uncertainty": 0, "note": "no uncertainty recorded by backend"}
                )
                if run.diagnostics.final_status == "failed":
                    entry["reason"] = run.diagnostics.error or "backend reported failure"
                else:
                    results[name] = run.result
        except (AdmissionError, EvidenceValidationError) as exc:
            entry["reason"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - batch isolation: record, continue
            entry["reason"] = f"{type(exc).__name__}: {exc}"

        totals["frames_submitted"] += entry["frame_count"]
        totals["points"] += entry["points"]

    # Cross-session registration over whatever actually reconstructed.
    registration = {
        "reference_session": None,
        "resolved": [],
        "unresolved": [],
        "reasons": {},
    }
    if align and len(results) >= 1:
        successful = sorted(results)
        chosen = reference_session
        note = ""
        if chosen not in results:
            chosen = successful[0]
            note = (
                f"requested reference {reference_session!r} did not "
                f"reconstruct; fell back to {chosen!r}"
            )
        registration["reference_session"] = chosen
        if note:
            registration["reasons"]["<reference>"] = note
        if len(results) >= 2:
            from registration.cross_session import align_reconstructed_sessions

            chain = align_reconstructed_sessions(results, chosen)
            info = chain.to_dict()
            registration["reasons"].update(info["reasons"])
            for session, status in info["status_by_session"].items():
                if session == chosen:
                    continue
                if status in ("aligned", "resolved"):
                    registration["resolved"].append(session)
                else:
                    registration["unresolved"].append(session)
        else:
            registration["reasons"]["<reference>"] = registration["reasons"].get(
                "<reference>",
                "single successful session; no cross-session registration needed",
            )

    all_groups_ok = (
        bool(sessions)
        and all(s["status"] == "success" for s in sessions.values())
    )
    complete = (
        all_groups_ok
        and not registration["unresolved"]
        and not totals["deferred_capacity"]
        and len(sessions) >= 1
    )
    outcome = "complete" if complete else "degraded"

    return {
        "outcome": outcome,
        "sessions": sessions,
        "registration": registration,
        "totals": totals,
    }
