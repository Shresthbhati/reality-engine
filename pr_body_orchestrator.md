## Summary

Implements the **Reconstruction Orchestrator** — the single layer that selects and executes appropriate reconstruction backends (spec §2, "the single layer that selects and executes appropriate reconstruction backends"). This replaces the previous pattern where every caller hard-wired one backend and had to re-implement input validation, availability checks, failure handling, and provenance stamping.

## Changes

- **`reconstruction/orchestrator.py`** — Core orchestrator implementing:
  - **Evidence validation gate** (`validate_evidence`): checks counts, kinds, unique IDs before any backend runs; raises `EvidenceValidationError` with attached issues when reconstruction is impossible regardless of backend
  - **Availability probing** (`detect_available_backends`): probes every candidate backend's ability to run *at all* in this environment (binary present, deps importable) without executing reconstruction; zero-arg probes registered as `backend.availability_probe`
  - **Acceptance gate** (`_accepts`): asks backend whether it accepts this specific evidence batch via optional `accepts(evidence) -> (bool, reason)` method; distinct from availability
  - **Preference-ordered selection with fallback**: first AVAILABLE backend that ACCEPTS the evidence wins; on decline/failure/registration-failed, falls through to next candidate
  - **Full attempt-log diagnostics** (`ReconstructionRunDiagnostics`): every backend tried, including refusals, lands in a frozen record — never swallowed
  - **Provenance stamping**: stamps `RECONSTRUCTED` provenance + backend identity on the result; preserves backend per-point/pose confidence verbatim
  - **Deterministic**: same evidence + same chain = same attempts in same order; no clocks/RNG in selection logic
  - **Duplicate backend support**: two same-class instances (e.g., two COLMAP quality presets) are a legitimate fallback chain; attempt logs disambiguate as `stub` / `stub#2`

- **`tests/test_reconstruction_orchestrator.py`** — 26 tests covering:
  - Evidence validation (empty, insufficient images, duplicate IDs)
  - Selection: availability vs acceptance as distinct gates, fallback on decline/failure/registration-failed
  - Detection: availability probes, probe lists, raising probes reported not raised
  - Provenance stamping + diagnostics serialization/roundtrip
  - Deterministic attempt ordering (excluding wall-clock duration)
  - COLMAP backend integration: probe returns bool+detail, `accepts` declines below 2-view floor
  - End-to-end: orchestrator → room pipeline → WorldIR (hand-computed expectations)

## Design Decisions

- **Honesty rules preserved from backend contract**: a backend returning `registration_status == "failed"` is a FAILURE the orchestrator surfaces; the next candidate is tried; if none remain, raises `ReconstructionOrchestrationError` with full attempt log — no bare "reconstruction failed"
- **Runtime product stays LLM-free**: selection is explicit policy, not a language model
- **Backend identity recorded on every attempt and final result**: downstream consumers (compiler, fusion, export) can always answer "which backend produced this?"
- **Frozen dataclasses**: safe to persist alongside result and show in inspector without live orchestrator

## Testing

```bash
# Orchestrator tests
pytest tests/test_reconstruction_orchestrator.py -v
# 26 passed

# Full suite (excluding pre-existing room inference failures)
pytest tests/ --ignore=tests/test_room_inference.py -v
# 902 passed, 2 skipped
```

## Related Work

This builds on the previously merged:
- MiDaS depth backend (PR #3)
- SAM segmentation backend (PR #4)
- Media ingestion pipeline (photos, video, LAS)
- COLMAP reconstruction backend
- Plane detection + room inference

## Next Steps

- Add OpenMVG/Open3D as additional reconstruction backends behind the same interface
- Implement `ITrackBackend` for 2D→3D instance lifting
- Wire orchestrator into the full capture→reconstruction→WorldIR pipeline