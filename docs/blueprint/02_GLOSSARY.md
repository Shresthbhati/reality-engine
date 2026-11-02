# 02 — Glossary

One word, one meaning. If code uses a different name for the same thing, the
mapping is in [20_REUSE_MAP.md](20_REUSE_MAP.md).

## Core nouns

| Term | Meaning |
|------|---------|
| **World** | The persistent digital twin of one place (a room, apartment, house, property). Has an ID, a name and a version history. |
| **WorldIR** | The authoritative structured data of one world version (see [11](11_WORLDIR_CONTRACT.md)). Not a mesh file. |
| **Version** | An immutable snapshot of a WorldIR. Every change creates a new version with a parent. |
| **Session (capture session)** | One batch of uploads (photos/videos) taken together. Can exist without a world and attach to one later. |
| **Evidence** | A source file: original photo, video, depth map, GPS track. Immutable once stored. |
| **Image** | One photo or one extracted video frame. Has metadata, a quality score and (optionally) a camera pose. |
| **Camera** | Intrinsics + pose for one image, in world coordinates. |
| **Observation** | *Something a model or algorithm saw in one image*: a mask/box, label, depth, features, model + version. Lives in the **ledger**. |
| **Entity** | *A persistent physical thing in the world* (a sofa, a wall). Has a stable ID, class, geometry, confidence, and links to many observations. |
| **EntityComponent** | A part of an entity (sofa cushion, window sash). Belongs to its parent; becomes a separate entity only if evidence says so. |
| **Surface** | A geometric surface of the architecture (a wall face, a floor region). |
| **Opening** | A hole in a surface that is a door, window or passage. |
| **Space** | A bounded volume: room, corridor, stairwell, outdoor zone. |
| **Relationship** | A typed edge between two entities (`supported_by`, `mounted_on`, `part_of`, …) with evidence. |
| **Measurement** | A dimension/distance with value, unit, uncertainty and **scale source**. |
| **CaptureRequest** | The extra photo the system needs: target region, suggested camera poses, instruction text. |
| **Ledger (observation ledger)** | Append-only store of all observations. The bridge between photos and the world. |

## Association decisions (outcomes of matching an observation)

| Decision | Meaning |
|----------|---------|
| `SAME_ENTITY_CONFIRMED` | Geometry + appearance agree: same physical object. Merge evidence. |
| `SAME_SURFACE_CONFIRMED` | The same architectural surface seen again. |
| `DIFFERENT_ENTITY` | Similar but physically distinct. Keep both. |
| `PHYSICAL_INTERSECTION` | Two geometries occupy incompatible space → investigate the conflict. |
| `IMAGE_OCCLUSION` | One blocks the other in this view; both may exist. |
| `PARTIAL_OVERLAP` | Only part is visible; keep as partial evidence. |
| `AMBIGUOUS` | Could be either; keep separate until more evidence arrives. |
| `INSUFFICIENT_EVIDENCE` | Not enough to decide; the observation stays unassociated. |

## Evidence states (per surface / property)

| State | Meaning |
|-------|---------|
| `OBSERVED` | Directly supported by image evidence. |
| `PARTIALLY_OBSERVED` | Some support, incomplete. |
| `OCCLUDED` | Hidden by something else in every view so far. |
| `UNOBSERVED` | No view covers it. |
| `INFERRED` | Estimated from geometry/structure (e.g. the wall behind the sofa continues). |
| `PRIOR_GENERATED` | Proposed by a learned/template prior (e.g. the unseen back of the sofa). |
| `CONFLICTED` | Evidence sources disagree; both are kept. |
| `UNKNOWN` | Cannot currently be determined. |

## Confidence vector (per entity, each 0–1 or null)

`classification, identity, position, orientation, dimensions, geometry,
completeness, material, colour, topology, cross_view`.
Never collapse these into one number for display without showing the parts on request.

## Scale states (per world / measurement)

| State | Meaning |
|-------|---------|
| `relative` | Shape is right up to an unknown scale. Units shown as "units", not metres. |
| `estimated` | Scaled from a weak source (known object size, metric depth model). Wide uncertainty. |
| `calibrated` | Scaled from a reliable source (user tape measurement, depth sensor, calibrated rig). |

## Maturity labels

See [README](README.md#maturity-labels-used-everywhere).
