# 17 — Ontology (what things can be)

The full wish-list of classes in the master spec is large on purpose. We do
**not** build a detector per class. We build **one** ontology file plus a
generic pipeline, and add classes as data (P9).

## Format

One versioned file: `config/ontology.yaml` (new). Each node:

```yaml
- path: furniture/seating/sofa/sectional    # unique, slash-separated
  label: Sectional sofa
  parent: furniture/seating/sofa
  prompts: ["sectional sofa", "L-shaped sofa"]   # for open-vocabulary detectors
  geometry: parametric:sofa                      # parametric:<template> | mesh | plane | proxy_box
  components: [frame, seat_cushion, back_cushion, armrest, leg, chaise]
  mobility: movable
  min_pixels: 2000            # below this, don't claim the class (resolution-aware)
  typical_dims_m: {w: [1.6, 3.5], d: [0.8, 2.0], h: [0.7, 1.0]}   # sanity-check prior only
  attach: [rests_on: architecture/floor]
```

`typical_dims_m` is used **only** to flag suspicious results and as a weak
scale hint — never to set dimensions (P4, P10).

## v1 class set (what the living-room demo needs)

| Group | Classes |
|-------|---------|
| Architecture | wall, floor, ceiling, door (opening + leaf), window (opening + frame + glazing), passage, column, skirting |
| Furniture | sofa (+sectional), armchair, chair, coffee_table, side_table, dining_table, desk, bed, cabinet, shelf, media_console, wardrobe |
| Soft / decor | rug, curtain, blind, cushion, throw, wall_art, mirror, vase, lamp (floor/table) |
| Electronics | tv, monitor, speaker |
| Fixtures | ceiling_light, wall_light, switch, socket, vent, radiator, ac_unit |
| Plants | potted_plant (+pot), plant (no pot) |
| Fallback | unknown_object (keeps geometry + evidence, no forced class) |

## Growth order (later phases)

1. Kitchen + bathroom fixtures and appliances.
2. Fine details (handles, hinges, multi-gang plates) — needs the S17 high-res pass.
3. Exterior & site (facade, roof parts, paths, fences, trees).
4. Street infrastructure (poles, signs, kerbs, drains) — no vehicles, no people.
5. Regional/design attributes (style, material family) as **attributes** on
   existing classes, never as new classes (`style` ≠ `material` ≠ `class`).

## Adding a class (checklist)

1. Add the node to `config/ontology.yaml` with prompts and a geometry kind.
2. Add 3+ test images where it appears to `datasets/`, with expected labels.
3. Run the perception benchmark; record precision/recall for the class.
4. No pipeline code change should be needed. If one is, the pipeline is too
   class-specific — fix that instead (P9).
