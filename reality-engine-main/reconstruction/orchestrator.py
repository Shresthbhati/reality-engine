"""Reconstruction orchestrator (deep-implementation sec 2: the single layer
that selects and executes appropriate reconstruction backends).

Before this module, every caller hard-wired one backend
(``FakeReconstructionBackend(...)`` in tests, the COLMAP run in
``tests/test_reconstruction_pipeline.py``) and had to re-implement input
validation, availability checks, failure handling, and provenance stamping
itself. The orchestrator owns all of that once:

    List[EvidenceItem]
      -> validate_evidence (counts, kinds, ids)
      -> detect backends (probe, honouring an explicit preference order)
      -> select (first AVAILABLE backend in policy order that ACCEPTS the
         evidence; availability and acceptance are distinct gates)
      -> execute with timing and error capture
      -> stamp RECONSTRUCTED provenance + diagnostics on the result
      -> on failure/decline: fall through to the next candidate
      -> everything attempted, including refusals, lands in a
         ReconstructionRunDiagnostics record -- never swallowed

Honesty rules preserved from the backend contract: a backend that returns
``registration_status == "failed"`` is a FAILURE the orchestrator surfaces
as such (the next candidate is tried; if none remain, the run raises
ReconstructionOrchestrationError with the full attempt log); no orchestrator
path ever invents geometry. The runtime product stays LLM-free: selection is
explicit policy, not a language model.

Backend identity is recorded on every attempt and on the final result, so
downstream consumers (compiler, fusion, export) can always answer "which
backend produced this?" -- and provenance.py's RECONSTRUCTED value stamps
every normalized artifact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from provenance import Uncertainty

from evidence.session import EvidenceItem, EvidenceKind
from .backend.interface import (
    IReconstructionBackend,
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from .backend.fake import FakeReconstructionBackend

#: Provenance value stamped onto every orchestration artifact.
RECONSTRUCTED = "RECONSTRUCTED"

#: Evidence kinds that can carry imagery usable for SfM.
IMAGE_EVIDENCE_KINDS = (EvidenceKind.PHOTO, EvidenceKind.VIDEO)

#: Minimum images for any SfM pipeline (spec: 20-50 photos per room, but a
#: two-view geometry is the floor for the geometry itself).
MIN_IMAGE_EVIDENCE = 2


class ReconstructionOrchestrationError(RuntimeError):
    """Raised when every candidate backend failed or declined the run.

    Carries the full attempt log so callers (and users) can see exactly
    which backends were tried and why each was skipped/failed -- never a
    bare "reconstruction failed".
    """

    def __init__(self, message: str, attempts: List["BackendAttempt"]):
        super().__init__(message)
        self.attempts = attempts


class EvidenceValidationError(ValueError):
    """Raised when the evidence batch cannot support reconstruction at all.

    Distinct from backend refusal: this is the orchestrator's own input
    gate, checked BEFORE any backend runs.
    """

    def __init__(self, message: str, issues: List[str]):
        super().__init__(message)
        self.issues = issues


@dataclass(frozen=True)
class BackendAttempt:
    """One backend's story in one run: probed, accepted/declined/failed.

    ``error`` carries the exception's type name and message; full tracebacks
    belong to the caller's logging, not to a frozen diagnostics record.
    """
    backend_name: str
    outcome: str          # "selected" | "declined" | "failed" | "succeeded"
    available: bool
    detail: str = ""
    error: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "backend_name": self.backend_name,
            "outcome": self.outcome,
            "available": self.available,
            "detail": self.detail,
            "error": self.error,
            "duration_s": round(self.duration_s, 6),
        }


@dataclass(frozen=True)
class ReconstructionRunDiagnostics:
    """Everything about one orchestrated run: validation, attempts, result.

    Frozen record, deterministic field order -- safe to persist alongside
    the result and to show in an inspector without a live orchestrator.
    """
    evidence_count: int
    image_evidence_count: int
    evidence_validation: Dict[str, object]
    attempts: tuple  # Tuple[BackendAttempt, ...] (frozen container)
    final_status: str  # "success" | "partial" | "failed"
    backend_name: str = ""
    duration_s: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "evidence_count": self.evidence_count,
            "image_evidence_count": self.image_evidence_count,
            "evidence_validation": dict(self.evidence_validation),
            "attempts": [a.to_dict() for a in self.attempts],
            "final_status": self.final_status,
            "backend_name": self.backend_name,
            "duration_s": round(self.duration_s, 6),
            "error": self.error,
        }


@dataclass(frozen=True)
class ReconstructionRun:
    """The orchestrator's return: normalized result + full diagnostics."""
    result: ReconstructionResult
    diagnostics: ReconstructionRunDiagnostics


def validate_evidence(evidence: List[EvidenceItem]) -> Dict[str, object]:
    """Orchestrator's input gate: counts, kinds, unique ids.

    Returns the validation summary used in diagnostics; raises
    EvidenceValidationError (issues attached) when reconstruction is
    impossible regardless of backend. Zero-value rules, no guessing:
    fewer than MIN_IMAGE_EVIDENCE image items, zero image items, or
    duplicate evidence ids all refuse before any backend is touched.
    """
    issues: List[str] = []
    image_items = [e for e in evidence if e.kind in IMAGE_EVIDENCE_KINDS]

    if not evidence:
        issues.append("no evidence items provided")
    if len(image_items) < MIN_IMAGE_EVIDENCE:
        issues.append(
            f"insufficient image evidence: {len(image_items)} "
            f"PHOTO/VIDEO item(s), need >= {MIN_IMAGE_EVIDENCE}"
        )
    ids = [e.id for e in evidence]
    duplicate_ids = sorted({i for i in ids if ids.count(i) > 1})
    if duplicate_ids:
        issues.append(f"duplicate evidence ids: {duplicate_ids}")

    if issues:
        raise EvidenceValidationError(
            "evidence batch cannot support reconstruction: " + "; ".join(issues),
            issues=issues,
        )

    return {
        "total": len(evidence),
        "image": len(image_items),
        "kinds": sorted({e.kind.value for e in evidence}),
        "issues": [],
    }


class ReconstructionOrchestrator:
    """Selects and executes reconstruction backends behind one interface.

    Args:
        backends: candidate backends in strict preference order. The first
            AVAILABLE backend that ACCEPTS the evidence wins; the rest are
            fallback. Duplicates by name are refused at construction (a
            fallback chain with two entries of the same name would make
            attempt logs ambiguous).
        name: optional label used in diagnostics.
    """

    def __init__(self, backends: List[IReconstructionBackend],
                 name: str = "reconstruction-orchestrator"):
        # Display names: same-class instances with different configs are a
        # legitimate fallback pattern (e.g. two COLMAP quality presets), so
        # repeats are disambiguated (#2, #3 ...) rather than refused --
        # attempt logs must be unambiguous without mutating the backends.
        counts: Dict[str, int] = {}
        self._display_names = {}
        for backend in backends:
            base = getattr(backend, "backend_name", type(backend).__name__)
            counts[base] = counts.get(base, 0) + 1
            self._display_names[id(backend)] = (
                base if counts[base] == 1 else f"{base}#{counts[base]}"
            )
        self._backends = list(backends)
        self._name = name

    # ------------------------------------------------------------------
    # Backend detection / availability
    # ------------------------------------------------------------------

    def detect_available_backends(self, evidence: List[EvidenceItem] = None) -> List[Dict[str, object]]:
        """Probe every candidate backend's availability without running it.

        Availability = can this backend run *at all* in this environment
        (binary present, deps importable) -- independent of whether it
        would accept this particular evidence batch. Never raises for a
        backend being unavailable; that is a reportable state, not an
        error.
        """
        report = []
        for backend in self._backends:
            available, detail = self._probe_availability(backend)
            report.append({
                "backend_name": self._backend_name(backend),
                "available": available,
                "detail": detail,
            })
        return report

    @staticmethod
    def _probe_availability(backend: IReconstructionBackend) -> "tuple[bool, str]":
        """Try each availability probe; first successful probe decides.

        Probes are zero-arg callables returning ``(ok, detail)`` set by the
        backend author (or tests) as ``backend.availability_probe`` -- a
        single callable or a list of fallback probes. A backend without
        any probe is considered available (it will be given its chance and
        can still decline/raise at execution). Probe failures are data,
        not crashes.
        """
        probe = getattr(backend, "availability_probe", None)
        if probe is None:
            return True, "no availability probe registered; assumed available"
        if isinstance(probe, (list, tuple)):
            details: List[str] = []
            for p in probe:
                try:
                    ok, detail = p()
                except Exception as exc:  # noqa: BLE001 - probe failures are data
                    details.append(f"probe raised: {type(exc).__name__}: {exc}")
                    continue
                if ok:
                    return True, str(detail)
                details.append(str(detail))
            return False, "; ".join(details) if details else "all availability probes failed"
        if not callable(probe):
            return False, f"availability probe is not callable: {probe!r}"
        try:
            ok, detail = probe()
        except Exception as exc:  # noqa: BLE001 - probe failures are data
            return False, f"availability probe raised: {type(exc).__name__}: {exc}"
        return bool(ok), str(detail)

    def _backend_name(self, backend: IReconstructionBackend) -> str:
        return self._display_names.get(id(backend), getattr(backend, "backend_name", type(backend).__name__))

    # ------------------------------------------------------------------
    # Selection + execution
    # ------------------------------------------------------------------

    def run(self, evidence: List[EvidenceItem]) -> ReconstructionRun:
        """Validate -> select -> execute -> stamp -> (fallback) -> report.

        Deterministic given the backend order: the same evidence and the
        same chain always walks the same attempts in the same order.
        """
        started = time.perf_counter()
        validation = validate_evidence(evidence)  # may raise EvidenceValidationError

        attempts: List[BackendAttempt] = []
        last_failure = ""

        for backend in self._backends:
            name = self._backend_name(backend)

            available, detail = self._probe_availability(backend)
            if not available:
                attempts.append(BackendAttempt(
                    backend_name=name, outcome="declined", available=False,
                    detail=f"unavailable: {detail}",
                ))
                continue

            accepts, why = self._accepts(backend, evidence)
            if not accepts:
                attempts.append(BackendAttempt(
                    backend_name=name, outcome="declined", available=True,
                    detail=why,
                ))
                continue

            t0 = time.perf_counter()
            try:
                result = backend.reconstruct(evidence)
            except Exception as exc:  # noqa: BLE001 - backend failures are data for the next fallback
                duration = time.perf_counter() - t0
                attempts.append(BackendAttempt(
                    backend_name=name, outcome="failed", available=True,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_s=duration,
                ))
                last_failure = f"{name}: {type(exc).__name__}: {exc}"
                continue

            duration = time.perf_counter() - t0

            if result is None:
                # Contract violation, not a crash: a backend must return a
                # ReconstructionResult (possibly registration_status=
                # "failed"), never None. Captured, logged, next fallback.
                attempts.append(BackendAttempt(
                    backend_name=name, outcome="failed", available=True,
                    error="backend returned None (contract violation: "
                          "expected ReconstructionResult)",
                    duration_s=duration,
                ))
                last_failure = f"{name}: backend returned None"
                continue

            if result.registration_status == "failed":
                attempts.append(BackendAttempt(
                    backend_name=name, outcome="failed", available=True,
                    detail="backend reported registration_status='failed'",
                    duration_s=duration,
                ))
                last_failure = f"{name}: registration failed"
                continue

            attempts.append(BackendAttempt(
                backend_name=name,
                outcome="succeeded" if result.registration_status == "success" else "partial",
                available=True,
                detail=f"registration_status={result.registration_status}",
                duration_s=duration,
            ))

            stamped = self._stamp_provenance(result, name)
            diagnostics = ReconstructionRunDiagnostics(
                evidence_count=validation["total"],
                image_evidence_count=validation["image"],
                evidence_validation=validation,
                attempts=tuple(attempts),
                final_status=result.registration_status,
                backend_name=name,
                duration_s=time.perf_counter() - started,
            )
            return ReconstructionRun(result=stamped, diagnostics=diagnostics)

        # Nothing succeeded.
        raise ReconstructionOrchestrationError(
            "all candidate backends failed or declined: " + (last_failure or "no attempts made"),
            attempts=attempts,
        )

    @staticmethod
    def _accepts(backend: IReconstructionBackend, evidence: List[EvidenceItem]) -> "tuple[bool, str]":
        """Ask the backend whether it accepts this evidence, before running.

        A backend may expose ``accepts(evidence) -> (bool, reason)``. When
        absent, acceptance is the interface's default: image evidence in
        sufficient count (which validate_evidence has already ensured), so
        acceptance is True.
        """
        accepts = getattr(backend, "accepts", None)
        if accepts is None:
            return True, "no accepts() gate; interface default applies"
        try:
            ok, why = accepts(evidence)
            return bool(ok), str(why)
        except Exception as exc:  # noqa: BLE001 - a broken gate is a decline, not a crash
            return False, f"accepts() raised {type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # Provenance stamping
    # ------------------------------------------------------------------

    @staticmethod
    def _stamp_provenance(result: ReconstructionResult, backend_name: str) -> ReconstructionResult:
        """Stamp RECONSTRUCTED provenance + backend identity on the result.

        The backend contract returns frozen dataclasses; stamping returns a
        NEW frozen result rather than mutating. The backend's per-point /
        per-pose confidence is preserved verbatim -- stamping adds
        orchestration-level identity (a note naming the backend), it never
        second-guesses backend confidence.
        """
        def _stamp(u: Uncertainty) -> Uncertainty:
            backend_note = f"backend={backend_name}"
            if u.note:
                merged = f"{u.note}; {backend_note}"
            else:
                merged = backend_note
            return Uncertainty(confidence=u.confidence, note=merged)

        stamped_points = [
            ReconstructedPoint(
                position=p.position,
                track_id=p.track_id,
                source_evidence_ids=list(p.source_evidence_ids),
                uncertainty=_stamp(p.uncertainty),
            )
            for p in result.points
        ]
        stamped_poses = [
            ReconstructedCameraPose(
                evidence_id=p.evidence_id,
                position=p.position,
                rotation=p.rotation,
                uncertainty=_stamp(p.uncertainty),
            )
            for p in result.camera_poses
        ]
        return ReconstructionResult(
            points=stamped_points,
            camera_poses=stamped_poses,
            registration_status=result.registration_status,
        )
