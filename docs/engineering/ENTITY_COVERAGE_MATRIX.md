# Entity Coverage Matrix (KiloCode RCPAS Workstream 3)

Source of truth: `config/ontology.yaml` (canonical ontology, version 1.0.0).
This matrix classifies every ontology class by implementation status:

- **Implemented** — class exists in the ontology AND detection code can emit
  it (perception/detection_adapter.py resolves labels via
  perception/detection/ontology_grounding.ground_label to canonical ontology
  nodes; small classes are reserved a prompt budget so they are actually
  queried).
- **Integrated** — emitted detections flow through the evidence pipeline
  (engine_multiengine/evidence_fusion.py, WorldIR, WorldStore persistence)
  and are queryable in the persisted world.
- **Evaluated** — detection performance measured against ground truth on the
  committed real_room_capture / room_capture datasets (precision, recall,
  IoU, duplicate rate, cross-view identity consistency).
- **Unverified** — present in the ontology but not yet measured against real
  data; treated as unverified, not claimed.

Classes with `small: true` (sockets, switches, door handles, hinges, vents,
etc.) are explicitly reserved a prompt budget in
perception/detection/ontology_grounding.production_queries so they are not
crowded out by furniture and structure.

## Coverage matrix

| Category | Class (ontology path) | Implemented | Integrated | Evaluated | Notes |
|----------|----------------------|-------------|------------|-----------|-------|
| Architecture | architecture/wall | Yes | Yes | No | `small: false`, fixed |
| Architecture | architecture/floor | Yes | Yes | No | `small: false`, fixed |
| Architecture | architecture/ceiling | Yes | Yes | No | `small: false`, fixed |
| Architecture | architecture/door | Yes | Yes | No | `small: false`, fixed |
| Architecture | architecture/window | Yes | Yes | No | `small: false`, fixed |
| Architecture | architecture/passage | Yes | Yes | No | `small: false`, fixed |
| Architecture | architecture/column | Yes | Yes | No | `small: false`, fixed |
| Architecture | architecture/skirting | Yes | Yes | No | `small: false`, fixed |
| Electrical | fixtures/electrical/switch | Yes | Yes | No | `small: true`, fixed |
| Electrical | fixtures/electrical/socket | Yes | Yes | No | `small: true`, fixed |
| Electrical | fixtures/hvac/vent | Yes | Yes | No | `small: false`, fixed |
| Electrical | fixtures/hvac/radiator | Yes | Yes | No | `small: false`, fixed |
| Electrical | fixtures/hvac/ac_unit | Yes | Yes | No | `small: false`, fixed |
| Interior | furniture/seating/sofa | Yes | Yes | No | `small: false`, movable |
| Interior | furniture/seating/sofa/sectional | Yes | Yes | No | `small: false`, movable |
| Interior | furniture/seating/armchair | Yes | Yes | No | `small: false`, movable |
| Interior | furniture/seating/chair | Yes | Yes | No | `small: false`, movable |
| Interior | furniture/tables/coffee_table | Yes | Yes | No | `small: false`, movable |
| Interior | furniture/tables/side_table | Yes | Yes | No | `small: false`, movable |
| Interior | furniture/tables/dining_table | Yes | Yes | No | `small: false`, movable |
| Interior | furniture/tables/desk | Yes | Yes | No | `small: false`, movable |
| Interior | furniture/sleeping/bed | Yes | Yes | No | `small: false`, movable |
| Interior | furniture/storage/cabinet | Yes | Yes | No | `small: false`, movable |
| Interior | furniture/storage/shelf | Yes | Yes | No | `small: false`, movable |
| Interior | furniture/storage/media_console | Yes | Yes | No | `small: false`, movable |
| Interior | furniture/storage/wardrobe | Yes | Yes | No | `small: false`, movable |
| Interior | decor/floor/rug | Yes | Yes | No | `small: false`, movable |
| Interior | decor/window/curtain | Yes | Yes | No | `small: false`, movable |
| Interior | decor/window/blind | Yes | Yes | No | `small: false`, movable |
| Interior | decor/soft/cushion | Yes | Yes | No | `small: false`, movable |
| Interior | decor/soft/throw | Yes | Yes | No | `small: false`, movable |
| Interior | decor/wall/wall_art | Yes | Yes | No | `small: false`, movable |
| Interior | decor/wall/mirror | Yes | Yes | No | `small: false`, movable |
| Interior | decor/object/vase | Yes | Yes | No | `small: false`, movable |
| Interior | decor/lighting/lamp | Yes | Yes | No | `small: false`, movable |
| Building services | fixtures/lighting/ceiling_light | Yes | Yes | No | `small: false`, fixed |
| Building services | fixtures/lighting/wall_light | Yes | Yes | No | `small: false`, fixed |
| Building services | fixtures/electrical/switch | Yes | Yes | No | `small: true`, fixed |
| Building services | fixtures/electrical/socket | Yes | Yes | No | `small: true`, fixed |
| Building services | fixtures/hvac/vent | Yes | Yes | No | `small: false`, fixed |
| Building services | fixtures/hvac/radiator | Yes | Yes | No | `small: false`, fixed |
| Building services | fixtures/hvac/ac_unit | Yes | Yes | No | `small: false`, fixed |
| Outdoor | *(none in current ontology)* | — | — | — | Not yet represented in ontology |

## Small-object coverage (small: true)

| Class | min_pixels | Status |
|-------|-----------|--------|
| fixtures/electrical/switch | 100 | Implemented, integrated, unevaluated |
| fixtures/electrical/socket | 100 | Implemented, integrated, unevaluated |
| fixtures/hvac/vent | 150 | Implemented, integrated, unevaluated |
| decor/soft/cushion | 300 | Implemented, integrated, unevaluated |
| decor/object/vase | 200 | Implemented, integrated, unevaluated |
| fixtures/electrical/switch (module) | 100 | Implemented, integrated, unevaluated |
| fixtures/electrical/socket (module) | 100 | Implemented, integrated, unevaluated |

## Classes with no ground-truth coverage

All classes above are **unverified** (no evaluated detection results against
real ground truth). The committed datasets (room_capture, south_building)
provide camera poses and scene structure but not per-class object bounding
box ground truth. Detection precision/recall, IoU, duplicate-entity rate and
cross-view identity consistency for every class remain **unmeasured** and
must not be reported as achieved.

## Total class count

| Status | Count |
|--------|-------|
| Implemented (in ontology + detectable) | ~60 |
| Integrated (flows through evidence pipeline) | ~60 |
| Evaluated (measured against real ground truth) | 0 |
| Unverified (ontology present, not measured) | ~60 |

## Methodology note

Coverage is measured against the canonical ontology
(`config/ontology.yaml`), not against what a detector happens to output.
A class is "implemented" only when it appears in the ontology with prompts
and can be queried; "integrated" only when its detections flow through the
evidence pipeline to WorldIR; "evaluated" only when measured against real
ground truth on a committed dataset. No class in the current ontology has
reached "evaluated" status.
