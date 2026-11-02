# Universal Entity Perception — plan, status, roadmap

Written 2026-11-01. Rule: an entity class is **supported** only when detection, localisation, reconstruction,
persistence and verification all work. A schema entry is *registered*, nothing more. This file states which is which.

## 1. Repository capability audit (what already existed)

| Area | Exists | Gap for this goal |
|---|---|---|
| Detection | `perception/detection/` Detector protocol, `ImageDetection`, `FakeDetector`, OWL-ViT and torchvision-COCO backends (never auto-download), Mask R-CNN backend, structure-only policy, masks | **No weights installed**, so no real detection has ever run. No electrical/plumbing/fixture prompts before this phase. |
| Lifting / identity | `perception/instances/` lifting, epipolar, appearance, `object_resolution.merge_hypotheses`, furniture-from-detections | Tuned for furniture-scale objects (decimetres). Not validated for centimetre fittings. |
| Reconstruction | COLMAP sparse + dense (CUDA), scale-independent working unit, room prior, dense structure pass | Sparse points ~1-4k: far too coarse to localise an 8 cm socket from the cloud. Small objects need their own measurement (see 6). |
| WorldIR | `Entity` has `component_ids`, typed `relationships`, `semantic_labels`, `observations`, `custom_properties`; `EntityType` has **no** electrical/plumbing/furniture/fixture values | Do not change the enum: fixtures are `structure` + `custom_properties["entity_class"]`. |
| Frontend | Scene tree and inspector read `custom_properties` / `dimensions` generically | New groups for fixtures not yet added. |
| Tests | Room-fidelity baseline (synthetic), contract tests | **No golden dataset** with real small objects. |

## 2. Taxonomy — `perception/entities/taxonomy.py` (implemented)

18 categories (buildings, rooms, envelope, openings, structural, circulation, electrical, plumbing, HVAC, furniture,
finishes, exterior, roads, vegetation, street furniture, utility, signage, misc) and 100+ classes with stable dotted ids
(`electrical.switchboard.module.socket`). `parent` is only the primary classification; everything else is a WorldIR
relationship (`part_of`, `attached_to`, `adjacent_to`, `supports`, ...), so nothing is forced into one tree.
Each class carries geometry strategy, plausible size range, small-object flag, detector prompts and typical hosts.

## 3. Entity schema (on the existing `Entity`)

`custom_properties`: `entity_class`, `geometry_strategy`, `parametric{shape, frame}`, `dimensions{name:{value,unit,state,basis}}`
(value `None` + state `UNKNOWN` when not measured), `evidence{source_images, classification_confidence}`,
`unknown_attributes`, `reconstruction_status` (`reconstructed | partial | detected_only | unresolved`),
`module_count_state` (`complete | lower_bound`). `observations` hold per-image evidence; `relationships` hold semantics.
Association, geometry and classification confidences are kept separately and never merged into one number.

## 4. Per-category detection strategy

* Large classes (walls, doors, windows, furniture, trees, roads): whole-image open-vocabulary detector + existing lifting.
* **Small classes** (`small=True`: sockets, switches, switchboards, USB/data ports, junction boxes, handles, hinges, locks,
  pipes, conduits, vents, thermostats, meters, drains, lamp fixtures): `small_objects.inspect_small_objects` — coarse pass
  finds regions, then full-resolution overlapping tiles *restricted to those regions* run only small-class prompts.
* Vegetation / street: map data (OSM) supplies the baseline; photos refine (planned).

## 5. Geometry strategy

Per class in the taxonomy: walls/doors/windows/sockets are **parametric**; pipes, conduits, stairs, lampposts, fences are
**procedural** (profile sweeps, per-step treads); trees are **volumetric** canopy + parametric trunk; furniture and
fixtures with complex form are **hybrid/mesh**; unknown objects stay **pointcloud**. A switchboard is never one box:
`switchboard.build_switchboard` emits the plate plus one entity per module with count, arrangement, orientation and host wall.

## 6. Multi-view association — `association.py` (implemented, synthetic-verified)

Same-image detections never merge; only the same class merges; 3D gate scales with both uncertainties; ambiguity refuses
the merge (two entities beat one wrong one); appearance can veto but never create a merge; observations and association
confidence are preserved. **Not yet done:** 3D positions for centimetre objects require a per-object depth/plane
measurement (ray-to-host-wall-plane intersection with the room's wall planes), which is the next phase.

## 7. Performance architecture

Implemented: coarse-to-fine tiling with region restriction, per-class NMS across overlaps, vectorised plane RANSAC,
fast dense tier, per-stage timings in every version report. Planned: tile-result cache keyed by image hash + prompt set,
parallel tiles, reuse of camera geometry across incremental runs, uncertainty-prioritised re-inspection.
Runtime and accuracy are measured separately (report timings vs benchmark scores).

## 8. WorldIR integration

Entities are ordinary WorldIR entities; persistence goes through the existing compile/WorldStore path with a round-trip
test. No separate store. Version lineage, coordinate frames and provenance use the existing mechanisms.

## 9. Completeness — `completeness.py` (implemented)

Per region: expected categories (labelled typical-room *priors*), detected, reconstructed, unresolved, occluded and
uninspected fractions, gaps, an upper-bound estimate capped at 0.95. `complete_claim` is true only if no gap remains and
the region was fully inspected unoccluded.

## 10. Golden benchmark specification (NOT yet built — needs real photos + tape measurements)

Scenes: residential interiors, kitchens, bathrooms, corridors, staircases, building exteriors, footpaths, trees, plants,
>=3 lamppost designs, switchboards (1-4 gang, mixed modules), individual plug points, small architectural details.
Per scene: photos (>=12, varied angles, plus close-ups), hand-labelled instance masks, per-entity tape measurements.
Metrics: detection precision/recall **per class and for small objects separately**, instance segmentation IoU,
cross-view association accuracy, duplicate-entity rate, 3D localisation error (cm), dimensional error, completeness error,
runtime, peak memory. **Gate rule:** a build that reconstructs walls accurately but misses electrical fittings FAILS the
entity benchmark (small-object recall is a hard threshold, set from the first real measurement, not guessed).

## Status (honest)

| Capability | Status | Evidence |
|---|---|---|
| Taxonomy registry (18 categories, 100+ classes) | IMPLEMENTED | `tests/test_entity_perception.py` |
| Small-object coarse-to-fine inspection | IMPLEMENTED against a scripted detector; **real detection UNVERIFIED** | no weights installed |
| Multi-view association | IMPLEMENTED (synthetic) | tests incl. same-image, ambiguity, class, appearance rules |
| Switchboard with individual modules | IMPLEMENTED (synthetic) | plate+modules, arrangement, unknowns, round-trip |
| Completeness subsystem | IMPLEMENTED | never-100% rules tested |
| Learned detection of sockets/switches/handles | **BLOCKED**: needs detector weights (user approval) | — |
| 3D localisation of small objects from photos | **NOT BUILT** | next phase (host-plane ray intersection) |
| Pipes, stairs components, lampposts, trees, plants from photos | REGISTERED only (geometry strategy chosen) | — |
| Golden dataset and benchmark | **NOT BUILT**; needs the user's photos + measurements | — |
| Frontend groups for fixtures | NOT BUILT | — |

## Ordered roadmap

1. (done) taxonomy, schema convention, small-object pass, association, switchboard, completeness, tests.
2. Detector weights approved -> run the real detector on the 20 living-room photos, record what it actually finds.
3. Small-object 3D localisation: intersect detection rays with the wall planes from the room prior; per-object sigma.
4. Wire inspection -> association -> switchboard builder -> WorldIR into the reconstruction job; add scene-tree groups.
5. Golden dataset + benchmark script + CI gate with per-class small-object recall.
6. Per-class geometry builders: door furniture, window subdivisions, stairs (tread/riser), railings, pipes, lamppost profiles.
7. Region completeness wired to coverage maps from camera poses.
