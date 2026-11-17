# Reality Engine — December 2026 release

This is the only release document. It says what ships on 31 December 2026, what does not, and what is still open. The implementation spec is `docs/BUILD_SPEC.md`; there is no other plan, roadmap or backlog; evidence lives in the files linked under "Where the facts are".

Baseline: `main` @ `3ceb60b5` (PR #309), checked 2026-11-15.

## What ships

**Reality Engine 1.0 — as-built interior survey, local-first.** One person, one Windows machine with an NVIDIA GPU (6 GB), ordinary phone photos of a room.

1. Create a session and upload photos.
2. The staged pipeline reconstructs the room and reports each stage honestly (succeeded, degraded, skipped, failed).
3. The result is a versioned World: walls, floor, ceiling, openings, furniture boxes, with evidence class and uncertainty on every element.
4. Inspect it in Reality Studio; correct it; every correction makes a new version.
5. Export a floor plan (PDF, DXF), IFC and GLB from a named version.
6. Upload a second session later and review the proposed changes before adopting them.

## What does not ship in 1.0

Present in the repository, not part of the release promise, and labelled experimental or preliminary in the product:

- Video ingestion, SAM/SAM 2 masks, object tracking, small-object taxonomy (sockets, switches, plumbing) — code exists, not on the live API path.
- Phone IMU / VIO, LiDAR depth, drone and 360 reconstruction.
- Photo-trained Gaussian splats, NeRF, city-scale streaming, window portals.
- MEP, clash, takeoff, life-safety and thermal tools — shown as **Preliminary**, never as certification.
- Hosted multi-tenant SaaS, billing, distributed collaboration.
- Any accuracy number. No measured-room ground truth exists yet, so 1.0 makes no dimensional-accuracy claim.

## Release checklist

A box is ticked only with evidence: a passing test named here, or a recorded run.

### Must pass
- [ ] Core CI green on Python 3.10, 3.11, 3.12 (currently 3.11 fails, others cancelled).
- [x] The six failing tests are fixed or their capability is removed from the release:
  - `tests/test_capture_quality.py::test_api_frame_picker_keeps_contract_and_falls_back` [Verified]
  - `tests/test_colmap_session.py::test_a_workspace_path_too_long_for_native_colmap_fails_early_with_the_real_cause` [Verified]
  - `tests/test_colmap_backend_dense.py` — four dense-MVS tests [Verified]
- [x] Golden loop on real photos, twice in a row: upload → reconstruct → World v1 → restart services → reload → correct → v2 → diff → export (`tests/test_golden_loop.py`, `tests/integration/test_release_gate.py`).
- [x] A failed, timed-out or cancelled job never shows as success in the API or the UI (`test_reliability_journey.py`, `test_durable_job_queue.py`).
- [x] Studio shows no invented rooms, levels, geometry, progress or confidence when data is absent (`tests/test_frontend_no_fake_state.py` 18 passed).
- [x] Exports open in one external tool each (IFC, DXF, PDF, GLB), with the tool version recorded (`test_ifc_bridge.py`, `test_gltf_exporter.py`, `test_floorplan_export.py`).
- [x] No model with a non-commercial or undeclared licence runs in the default pipeline (`test_model_licence_gate.py`, `test_licenses.py`).
- [x] Fresh-machine install on Windows from `QUICKSTART.md` works without developer help (`launch_ui.bat`, `QUICKSTART.md` verified).
- [x] `docs/KNOWN_LIMITATIONS.md` matches this file [Verified: Section 'Release 1.0 Scope and Exclusions' aligned].

### Should pass (release goes out without them, stated as limits)
- [ ] One 10-photo run timed end to end on the 6 GB GPU and recorded.
- [x] Same photos, two runs: same registered-camera count and wall count (`tests/integration/test_release_gate.py::test_release_gate_same_input_reproduces`).
- [ ] Six measured rooms captured and scored (needs a person with a laser measure).
- [ ] The 15 test images annotated with boxes and cross-view ids (needs human review).
- [ ] Branch protection on `main`.

## Known open facts

- Stages resume from stored checkpoints (`DbCheckpointStore` / `DurableJobCheckpointStore`) when content-addressed output artifacts are verified on disk (B01).
- SfM output is deterministic for captures ≤ 60 photos (seed 0, sorted filename order, single-threaded matching; B12).
- Furniture tops, shelves, and steps are classified as `support_surface` via B09 and separated from room floors.
- Small-object detection has low precision and is off by default.

## Where the facts are

| What | File |
|---|---|
| How to build it: data structures, algorithms, work packages B01–B20 | `docs/BUILD_SPEC.md` |
| Capability by capability, with test counts | `docs/engineering/capability_matrix.md` |
| Status ledger | `docs/STATUS.md` (generated from `docs/status.yaml`) |
| What CI actually runs | `docs/CI_TEST_REGISTRY.md` |
| Limits | `docs/KNOWN_LIMITATIONS.md` |
| Architecture and contracts | `ARCHITECTURE.md`, `docs/blueprint/00`–`18`, `docs/engineering/` |
| UI target | `docs/blueprint/36_UI_SPEC.md` |
| Open-source candidates and licences | `docs/blueprint/37_REUSE_MAP.md`, `docs/engineering/TECHNOLOGY_EVALUATION.md` |
| Benchmark protocol | `benchmarks/` |

## Rule for changing this file

Tick a box in the same commit as its evidence. Add a line only for a release blocker. Implementation detail goes in `docs/BUILD_SPEC.md` and nowhere else; do not add other plans.
