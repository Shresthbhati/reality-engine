# Reconstruction Backend Decision (Goal C)

Evaluation only — no code in this pass, per the directive's own research-loop
rule (§30/§37): decide and justify before writing an adapter.

## Candidates considered

| Backend | License | Output rep | Maturity | Integration cost |
|---|---|---|---|---|
| **COLMAP** | BSD (permissive) | Sparse SfM point cloud + camera poses; MVS dense cloud/mesh via separate stage | Very mature, the de facto SfM reference implementation, actively maintained | External binary/CLI or pycolmap bindings; no native Python physics/WorldIR integration — everything crosses a process boundary |
| **OpenDroneMap (ODM)** | AGPL-3.0 | Orthomosaic, DSM, textured mesh, georeferenced point cloud — drone/aerial-survey shaped | Mature, project is literally "drone mapper" (this repo's own dir name) — output already matches the domain | AGPL is viral if this project's code links against it as a library; ODM is designed as a standalone pipeline (Docker/CLI), not an embeddable library — same process-boundary cost as COLMAP, but license is the blocker |
| **Meshroom / AliceVision** | MPL-2.0 | Mesh + textures, node-graph pipeline | Mature, GUI-first tooling, less scriptable than COLMAP | Heaviest — GPU-dependent node graph, not designed for headless embedding |
| **OpenSfM** | BSD-2 | Sparse SfM + optional dense (via OpenMVS) | Maintained but smaller community than COLMAP; Python-native (Mapillary) | Best language fit (pure Python core) but weaker single-image/edge-case robustness than COLMAP |

## Decision: COLMAP, behind a subprocess adapter — not vendored, not started yet

**Why COLMAP over ODM despite ODM matching the domain name:** ODM's AGPL-3.0
would force this repo's license if reconstruction code links against it
in-process; COLMAP's BSD license imposes no such constraint. Both are
external-process integrations anyway (neither is a pip-installable library
with a clean Python API), so ODM's "closer to drone workflows" advantage is
mostly cosmetic — COLMAP still requires our own georeferencing/DSM
post-processing on top of its point cloud either way, which is exactly the
kind of thing `evidence/promote.py`-style adapter code would do.

**Why COLMAP over OpenSfM:** OpenSfM is the better long-term language fit
(pure Python, easier to introspect and patch), but COLMAP's robustness and
ecosystem maturity make it the safer default backend to build the adapter
interface against first. OpenSfM stays a plausible second backend once the
adapter abstraction (`IReconstructionBackend`, following the existing
`IPhysicsBackend` precedent) exists — swapping backends behind that interface
is exactly what Goal J (extensibility) is for.

**Representation compatibility:** COLMAP's sparse output (camera poses +
3D points with track IDs) maps directly onto `Observation`/`Measurement`
provenance-tagged records the same way `evidence/promote.py` already does for
manual measurements — each reconstructed point becomes an `OBSERVED`→
`RECONSTRUCTED`-provenance entity, not an invented one. Dense MVS output
(meshes) maps onto `Geometry`/`Material`. No schema changes to WorldIR are
needed; existing `Provenance.RECONSTRUCTED` already anticipated this case.

## What this decision does NOT authorize

No COLMAP dependency is added, no binary is invoked, no adapter code exists.
The next step is the adapter *interface* only (method signatures: capture a
`Session`'s photo evidence in, typed poses+points out, provenance attached) —
still no COLMAP call inside it — followed by a real integration once the
interface is reviewed.
