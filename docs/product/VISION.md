# Reality Engine — Product Vision (one page)

Merged from the five vision documents of 2026-10-18 (Product Vision/Constitution, Spatial UI/UX, World Factory,
Photo-to-World, Complete Vision). Those remain background reading; **this page and `docs/implementation/ROADMAP.md` are
the scope of record.**

## The promise
Give Reality Engine images of a place (or a description of one). It reconstructs what the evidence supports, builds what you
ask it to create, says what remains uncertain, and hands you an **editable, versioned 3D world** you can inspect and export.

## Three modes, one world model (WorldIR)
| Mode | Input | Output status in WorldIR |
|---|---|---|
| **Reconstruct** | photos, video, plans, measurements | OBSERVED / RECONSTRUCTED / MEASURED, with evidence + uncertainty |
| **Create** | dimensions, a brief, templates, procedural rules | PROCEDURAL / GENERATED (never presented as real) |
| **Complete** | a partial reconstruction | reconstructed core + clearly labelled GENERATED proposals the user accepts or rejects |

One scale ladder: object → room → apartment → building → site → district → city → world. Same data model, same editor.

## What the product is made of
Evidence & ingest → reconstruction (COLMAP, depth, fusion, mesh) → architectural interpretation → **World Factory**
(procedural creation) → WorldIR + WorldStore (versions, diffs, provenance) → validation → Reality Studio → exporters
(glTF, USD, IFC, CityGML/JSON, Blender, GIS, Godot, Unreal, ROS, SUMO, Habitat).

## Non-negotiables (acceptance criteria, not style)
1. No success without verified success; failed, partial, cancelled stay distinct.
2. Observed ≠ inferred ≠ generated; unknown stays unknown; no fabricated geometry, progress or thumbnails.
3. Every important claim traces to evidence or to the operation that produced it.
4. Scale is explicit (unknown / estimated / measured); never print metres for unknown scale.
5. Edits are validated, versioned and reversible; stale parents are rejected, not overwritten.
6. Local change → local regeneration (dependency tracking), not a rebuild of the world.

## Non-goals (do not promise)
Perfect reconstruction from arbitrary photos; hidden-geometry recovery; structural/fire/regulatory certification;
complete construction-ready BIM; one-prompt AAA cities; multi-tenant cloud and pricing (later).

## How "done" is judged
The acceptance journey, automated: real photos → reconstruct → WorldIR → commit → restart → correct → new version → diff →
export. Creation journey: describe/enter parameters → generate → edit one parameter → only dependents regenerate → save →
reload → export.

## Where things are tracked
Roadmap `docs/implementation/ROADMAP.md` · backlog `PENDING_IMPLEMENTATION.md` (P6b Studio parity, P9 World Factory) · plan
`docs/superpowers/plans/2026-10-18-world-factory-and-studio-master-plan.md` · ledger `.agent/TASKS.yaml`.
