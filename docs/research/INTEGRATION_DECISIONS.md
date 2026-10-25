# Integration Decisions — Reality Engine

Records the RETAIN / WRAP / ADAPT / REIMPLEMENT / REPLACE / DEFER / REJECT decision for each investigated capability, the rationale, and the licensing basis. Grounded in `docs/research/capability-fusion-matrix.csv` and `docs/research/TECHNOLOGY_DECOMPOSITION.md`. Pinned commit SHAs in `docs/research/REPOSITORY_INVENTORY.md`.

## Decision index

| Capability | Decision | Rationale (one line) |
|---|---|---|
| REC-001 COLMAP SfM/MVS + dense MVS | RETAIN (WRAP as subprocess) | Already the wrapped baseline behind `IReconstructionBackend`; BSD license; colmap binary already installed on this machine; no demonstrated advantage to switch |
| REG-001 Open3D point-cloud registration | DEFER | open3d python wheel not installable here (PyPI unreachable); RE's scipy/cKDTree ICP + point-to-plane already real and tested |
| DEP-001 Depth (MiDaS / Depth Anything V2) | DEFER | Model checkpoints not cached, cannot download (PyPI blocked), no GPU; Depth Anything V2 per-checkpoint licensing unverified |
| SEG-001 Segmentation (SAM) | DEFER | Backend code exists but blocked at runtime (no SAM checkpoint, no GPU); SAM2 video tracking unused by photo pipeline |
| REA-001 REA reverse-engineering platform | ADAPT (patterns only, no code copy) | Source-level research instrument; provider registry/evidence-ledger/unknown-ownership patterns inform RE contracts; live binary analysis blocked (no Ghidra/IDA/Hopper; Node version) |
| WIR-001 WorldIR canonical model | RETAIN + evolve (REIMPLEMENT within RE) | WorldIR v1 stable; add schema migrations / content-addressed hashing / temporal validity, no provider model leakage |
| WS-001 WorldStore versioning | RETAIN + extend (REIMPLEMENT within RE) | Immutable versions + lineage + integrity already real; add diff integration / candidate revisions |
| EXP-001 Exporters (glTF/USDA/CityJSON/CityGML) | RETAIN | 13 modules with real-geometry paths; add round-trip tests; IFC/USD deferred |
| SPA-001 Spatial indexing | RETAIN | Accelerated grid+BVH implemented and benchmarked (100k entities) |
| CF-001 Frame graph + pose-graph | RETAIN frame graph; DEFER pose-graph (GTSAM) | Frame graph real; multi-session alignment not a defined slice; GTSAM not installed |
| PER-ARCH Perception provider architecture | RETAIN + ADAPT REA patterns | IDepthBackend/ISegmentationBackend + availability probe already honest; adopt REA registry/evidence patterns |
| BIM-001 IFC/USD (IfcOpenShell/OpenUSD) | DEFER | BIM not a current slice; license mix (LGPL/MIT vs Apache 2.0) to verify |
| SLAM-001 ORB_SLAM3/VINS-Fusion/OpenVINS | REJECT | GPLv3 + heavy build + no defined robotics slice in RE |
| SLAM-001 GTSAM pose-graph | DEFER | BSD, could support Layer 4 multi-session alignment later; not installed |
## Decision details

### REC-001 — COLMAP: RETAIN (WRAP as subprocess)
COLMAP (BSD, `fa08906`) is wrapped by `reconstruction.backend.colmap_backend.ColmapReconstructionBackend`, which shells out to the `colmap` CLI and parses COLMAP-text sparse output. The binary is already installed at `C:\Users\shres\tools\colmap-extracted\bin\colmap.exe` and on PATH. BSD is permissive — wrapping as a subprocess creates no linking/derivative-work concern. No alternative (Open3D/Meshroom/ODM) has demonstrated a measurable advantage on RE's inputs/hardware, so COLMAP remains the baseline per the mission's "do not replace without comparative testing" rule. `tests/test_colmap_backend.py` plus calibration/registration tests pass (94).

### REG-001 — Open3D registration: DEFER
Open3D (MIT, `b6c5e19`) offers colored ICP, RANSAC/FPFH global registration, and point-cloud processing, but its python wheel cannot be installed in this environment (PyPI unreachable, `pip install open3d` times out). RE's `registration.registration.py` already implements point-to-point and point-to-plane ICP with measured `ResidualStats`/`RegistrationCovariance` and honest `AttemptRecord`s, tested via `tests/test_registration.py`. Revisit if global registration for poor initial alignment becomes a proven gap and open3d becomes installable.

### DEP-001 / SEG-001 — Depth & Segmentation: DEFER
`MiDaSDepthBackend` and `SAMSegmentationBackend` are written and honest (raise `*UnavailableError` rather than fabricate), but runtime inference is blocked: checkpoints aren't cached and can't be downloaded (PyPI unreachable); torch is CPU-only; no GPU. Depth Anything V2's checkpoint licensing is fragmented (per TECHNOLOGY_REGISTRY.md). Do not auto-download weights; do not claim integration merely because the repo is cloned. Revisit when checkpoints and GPU are available and licensed for use.

### REA-001 — REA: ADAPT patterns only
REA (MIT, `b22b0ef`) is a Node.js MCP/CLI reverse-engineering tool. Its `AnalysisProviderRegistry` (deterministic selection, no fallback), `SessionProviderRouter`, `EvidenceLedger`, and `Unknown` ownership align with RE's existing provider/evidence honesty conventions. REA is a dev-time research instrument; it is not embedded in RE. Live binary analysis is blocked (no Ghidra/IDA/Hopper installed; Node 26.8.1 is below REA's 26.12.1 minimum so the full `npm ci && build` was not attempted).

### WIR-001 / WS-001 — WorldIR / WorldStore: RETAIN, evolve in place
WorldIR v1 (`world_ir/world_v1.py`) and WorldStore (`worldstore/store.py`) are real, tested, immutable, content-addressed, lineage-tracked, integrity-verified. Target evolution = schema migrations, content-addressed hashing, temporal validity, tighter diff integration. Implemented within RE; no external dependency.

### BIM-001 / SLAM-001 / geospatial: DEFER or REJECT
IFC/USD (IfcOpenShell/OpenUSD) deferred (BIM not a slice; license mix). Geospatial (Cesium/3D Tiles/GDAL/PROJ) deferred. ORB_SLAM3/VINS-Fusion/OpenVINS REJECTED (GPLv3 + no robotics slice). GTSAM deferred (BSD pose-graph candidate for Layer 4 later).

## Licensing attestation
No code was copied from any candidate repository into RE. All integration decisions keep candidate repos in `codes/` as separate projects; RE depends on them only as optional subprocesses/external tools where noted. BSD/MIT/Apache-2.0 candidates (COLMAP, Open3D, REA code) present no copyleft linking concern when used as separate processes or pattern references only. GPLv3 candidates (ORB_SLAM3, VINS-Fusion, OpenVINS) are rejected for RE integration to avoid copyleft contamination. Per-checkpoint neural-weight licenses (MiDaS, Depth Anything V2, SAM) were not verified because no weights were obtained; they remain blocked pending review.

## Open items
- Revisit Open3D/GTSAM/depth/segmentation when hardware (GPU), network (PyPI), and checkpoints are available and licensed.
- Add round-trip export tests for glTF/USDA/CityJSON/CityGML.
- If a BIM/geospatial slice is chartered, verify IfcOpenShell (LGPL/MIT) and OpenUSD (Apache 2.0) license compatibility before any adapter is written.

