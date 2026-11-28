# Reality Engine — Complete Entity Taxonomy (Phase 1)

**Canonical Source:** `config/ontology.yaml` (parsed by `perception/ontology.py`, `Ontology` / `OntologyNode`).  
**Fallback Policy:** Unmapped labels resolve strictly to `fallback/unknown_object` (`resolved=False`). A taxonomy entry is **not** a claim of implemented detection.

---

## Capability Status Definitions

| Status | Meaning |
|---|---|
| `implemented_and_tested` | Executed in production/integration pipeline and verified by automated tests with real or ground-truth assertions. |
| `implemented_unvalidated` | Queries/geometry paths exist, but lack independent annotated ground-truth validation on real photos. |
| `partially_supported` | Detected in 2D or reconstructed in limited geometric form, but missing 3D lifting, relationships, or edge cases. |
| `not_implemented` | Defined in taxonomy with scale/evidence contracts; no dedicated detector or reconstructor active yet. |
| `not_applicable` | Abstract grouping or container node; not directly detected as a leaf instance. |

---

## Domain A: Architecture

| Class | Canonical Path | Parent | Geometry | Scale (m) | Required Evidence | Instance ID? | Part Of | Capability Status |
|---|---|---|---|---|---|---|---|---|
| Building | `building` | `null` | `envelope` | 3.0–200.0 | Multi-view / OSM / storeys | Yes | `null` | `partially_supported` |
| Floor (Storey) | `building.storey` | `building` | `plane` | 2.2–6.0 | Multi-view / floor-ceiling planes | Yes | `building` | `partially_supported` |
| Room | `room` | `building.storey` | `space` | 1.5–30.0 | Bounding wall planes + floor | Yes | `building.storey` | `implemented_and_tested` |
| Corridor | `room.corridor` | `room` | `space` | 0.9–50.0 | Elongated wall pair + floor | Yes | `building.storey` | `partially_supported` |
| Wall | `architecture/wall` | `null` | `plane` | 0.5–30.0 | Multi-view plane / RGB-D | Yes | `building.storey` | `implemented_and_tested` |
| Floor surface | `architecture/floor` | `null` | `plane` | 0.5–50.0 | Low-elevation point band | Yes | `room` | `implemented_and_tested` |
| Ceiling | `architecture/ceiling` | `null` | `plane` | 0.5–50.0 | High-elevation point band | Yes | `room` | `implemented_and_tested` |
| Column | `structural.column` | `null` | `parametric` | 0.2–10.0 | Vertical multi-view / RGB | Yes | `building.storey` | `implemented_unvalidated` |
| Beam | `structural.beam` | `null` | `parametric` | 0.2–20.0 | Ceiling-adjacent horizontal | Yes | `building.storey` | `implemented_unvalidated` |
| Structural opening | `opening.structural` | `null` | `parametric` | 0.4–4.0 | RGB multi-view + wall plane gap | Yes | `architecture/wall` | `partially_supported` |
| Door | `architecture/door` | `null` | `parametric:door` | 0.7–2.4 | RGB multi-view + wall opening | Yes | `architecture/wall` | `partially_supported` |
| Door frame | `opening.door.frame` | `architecture/door` | `parametric` | 0.7–2.4 | High-res RGB + wall boundary | Yes | `architecture/door` | `implemented_unvalidated` |
| Window | `architecture/window` | `null` | `parametric:window` | 0.5–3.0 | RGB multi-view + glazing/gap | Yes | `architecture/wall` | `partially_supported` |
| Window frame | `opening.window.frame` | `architecture/window` | `parametric` | 0.5–3.0 | High-res RGB + wall boundary | Yes | `architecture/window` | `implemented_unvalidated` |
| Staircase | `circulation.stairs` | `null` | `parametric` | 1.0–15.0 | Multi-view steps + elevation | Yes | `building` | `implemented_unvalidated` |
| Landing | `circulation.stairs.landing` | `circulation.stairs` | `plane` | 0.8–5.0 | Horizontal step transition | Yes | `circulation.stairs` | `not_implemented` |
| Railing | `circulation.railing` | `null` | `parametric` | 0.5–15.0 | Thin linear barrier + stairs/balcony | Yes | `circulation.stairs` | `implemented_unvalidated` |
| Roof | `exterior.roof` | `building` | `surface` | 2.0–100.0 | Exterior multi-view / aerial | Yes | `building` | `implemented_unvalidated` |
| Facade | `exterior.facade` | `building` | `surface` | 2.0–100.0 | Exterior multi-view | Yes | `building` | `partially_supported` |
| Balcony | `exterior.balcony` | `building` | `surface` | 1.0–15.0 | Exterior multi-view + railing | Yes | `exterior.facade` | `not_implemented` |
| Entrance | `architecture/entrance` | `null` | `parametric` | 0.9–4.0 | Exterior/interior door + facade | Yes | `exterior.facade` | `not_implemented` |
| Architectural recess | `opening.recess` | `null` | `parametric` | 0.3–4.0 | Depth recess in wall plane | Yes | `architecture/wall` | `not_implemented` |

## Domain B: Furniture & Interior Objects

| Class | Canonical Path | Parent | Geometry | Scale (m) | Required Evidence | Instance ID? | Part Of | Capability Status |
|---|---|---|---|---|---|---|---|---|
| Table | `furniture/tables/dining_table` | `furniture` | `proxy_box` / `parametric` | 0.5–3.5 | RGB multi-view + floor support | Yes | `room` | `partially_supported` |
| Chair | `furniture/seating/chair` | `furniture` | `proxy_box` / `parametric` | 0.4–1.2 | RGB multi-view | Yes | `room` | `partially_supported` |
| Sofa | `furniture/seating/sofa` | `furniture` | `proxy_box` / `parametric` | 1.2–3.5 | RGB multi-view | Yes | `room` | `partially_supported` |
| Bed | `furniture/sleeping/bed` | `furniture` | `proxy_box` / `parametric` | 1.0–2.4 | RGB multi-view | Yes | `room` | `partially_supported` |
| Cabinet | `furniture/storage/cabinet` | `furniture` | `proxy_box` | 0.4–2.5 | RGB multi-view | Yes | `room` | `partially_supported` |
| Wardrobe | `furniture/storage/wardrobe` | `furniture` | `proxy_box` | 0.6–3.0 | RGB multi-view | Yes | `room` | `partially_supported` |
| Shelf | `furniture/storage/shelf` | `furniture` | `proxy_box` | 0.3–2.5 | RGB multi-view | Yes | `room` | `partially_supported` |
| Desk | `furniture/tables/desk` | `furniture` | `proxy_box` | 0.8–2.5 | RGB multi-view | Yes | `room` | `partially_supported` |
| Counter | `furniture/counter` | `furniture` | `proxy_box` | 0.6–4.0 | RGB multi-view | Yes | `room` | `implemented_unvalidated` |
| Appliance | `furniture/appliance` | `furniture` | `proxy_box` | 0.3–2.0 | RGB multi-view | Yes | `room` | `implemented_unvalidated` |
| Storage unit | `furniture/storage_unit` | `furniture` | `proxy_box` | 0.4–3.0 | RGB multi-view | Yes | `room` | `implemented_unvalidated` |
| Furniture components | `furniture.components` | `furniture` | `subpart` | 0.05–2.0 | Close-up / segmented crop | No | `furniture` | `not_implemented` |

## Domain C: Electrical & Mechanical Fixtures

| Class | Canonical Path | Parent | Small? | Scale (m) | Required Evidence | Instance ID? | Part Of | Capability Status |
|---|---|---|---|---|---|---|---|---|
| Electrical socket / Plug point | `fixtures/electrical/socket` | `null` | Yes | 0.06–0.20 | High-res tile + wall mount | Yes | `architecture/wall` | `implemented_unvalidated` |
| Switch | `fixtures/electrical/switch` | `null` | Yes | 0.06–0.20 | High-res tile + wall mount | Yes | `architecture/wall` | `implemented_unvalidated` |
| Switchboard | `electrical.switchboard` | `null` | Yes | 0.15–0.80 | High-res tile + wall mount | Yes | `architecture/wall` | `implemented_unvalidated` |
| Distribution panel | `electrical.distribution_panel` | `electrical.switchboard` | Yes | 0.20–1.20 | High-res tile + wall mount | Yes | `architecture/wall` | `implemented_unvalidated` |
| Light fixture | `fixtures/lighting/light_fixture` | `null` | Yes | 0.05–1.50 | RGB multi-view | Yes | `architecture/ceiling` / `wall` | `implemented_unvalidated` |
| Ceiling fan | `fixtures/hvac/ceiling_fan` | `null` | No | 0.80–1.80 | RGB multi-view + ceiling | Yes | `architecture/ceiling` | `implemented_unvalidated` |
| Vent | `hvac.vent` | `null` | Yes | 0.10–0.80 | High-res tile + surface mount | Yes | `architecture/wall` / `ceiling` | `implemented_unvalidated` |
| Air-conditioning unit | `hvac.ac_unit` | `null` | No | 0.50–1.50 | RGB multi-view | Yes | `architecture/wall` | `implemented_unvalidated` |
| Duct | `hvac.duct` | `null` | No | 0.20–10.0 | RGB multi-view | Yes | `building.storey` | `implemented_unvalidated` |
| Pipe | `plumbing.pipe` | `null` | Yes | 0.02–10.0 | High-res tile / multi-view | Yes | `architecture/wall` | `implemented_unvalidated` |
| Radiator | `hvac.radiator` | `null` | No | 0.40–2.00 | RGB multi-view + wall | Yes | `architecture/wall` | `implemented_unvalidated` |
| Plumbing fixture | `plumbing.fixture` | `null` | Yes | 0.05–1.50 | High-res tile | Yes | `room` | `implemented_unvalidated` |
| Smoke detector | `fixtures/safety/smoke_detector` | `null` | Yes | 0.08–0.20 | High-res tile + ceiling | Yes | `architecture/ceiling` | `not_implemented` |
| Fire alarm | `fixtures/safety/fire_alarm` | `null` | Yes | 0.08–0.25 | High-res tile + wall | Yes | `architecture/wall` | `not_implemented` |
| Cable tray | `utility.cable_run` | `null` | Yes | 0.05–10.0 | High-res tile | Yes | `architecture/ceiling` | `not_implemented` |

## Domain D: Fine Architectural Components

| Class | Canonical Path | Parent | Small? | Scale (m) | Required Evidence | Instance ID? | Part Of | Capability Status |
|---|---|---|---|---|---|---|---|---|
| Door handle | `opening.door.handle` | `architecture/door` | Yes | 0.05–0.30 | High-res tile + door host | Yes | `architecture/door` | `implemented_unvalidated` |
| Door hinge | `opening.door.hinge` | `architecture/door` | Yes | 0.03–0.20 | High-res tile + door frame | Yes | `architecture/door` | `implemented_unvalidated` |
| Lock | `opening.door.lock` | `architecture/door` | Yes | 0.03–0.15 | High-res tile + door leaf | Yes | `architecture/door` | `implemented_unvalidated` |
| Window latch | `opening.window.latch` | `architecture/window` | Yes | 0.03–0.15 | High-res tile + window frame | Yes | `architecture/window` | `implemented_unvalidated` |
| Window sill | `opening.window.sill` | `architecture/window` | Yes | 0.30–3.00 | RGB multi-view + window | Yes | `architecture/window` | `implemented_unvalidated` |
| Skirting | `envelope.skirting` | `architecture/wall` | Yes | 0.05–10.0 | Floor-wall junction + high-res | No | `architecture/wall` | `not_implemented` |
| Molding | `envelope.molding` | `architecture/wall` | Yes | 0.02–5.00 | High-res tile + wall/ceiling | No | `architecture/wall` | `not_implemented` |
| Trim | `envelope.trim` | `architecture/wall` | Yes | 0.02–4.00 | High-res tile + opening | No | `architecture/wall` | `not_implemented` |
| Grille | `opening.grille` | `null` | Yes | 0.10–2.50 | High-res tile | Yes | `architecture/window` | `not_implemented` |
| Handrail | `circulation.stairs.handrail` | `circulation.stairs` | No | 0.50–10.0 | Linear rail + stairs | Yes | `circulation.stairs` | `not_implemented` |
| Fastener | `hardware.fastener` | `null` | Yes | 0.01–0.15 | Macro crop | No | `null` | `not_implemented` |

## Domain E: Outdoor & Landscape Entities

| Class | Canonical Path | Parent | Scale (m) | Required Evidence | Instance ID? | Capability Status |
|---|---|---|---|---|---|---|
| Tree / Trunk / Canopy / Branch | `vegetation.tree`, `.trunk`, `.canopy`, `.branch` | `vegetation.tree` | 0.05–30.0 | Outdoor multi-view | Yes | `implemented_unvalidated` (tree) / `not_implemented` (subparts) |
| Shrub / Hedge / Plant / Flower bed | `vegetation.shrub`, `.hedge`, `plants/indoor/plant`, `vegetation.flower_bed` | `null` | 0.2–10.0 | Outdoor/indoor RGB | Yes | `implemented_unvalidated` |
| Grass / Soil | `vegetation.grass`, `exterior.soil` | `exterior.terrain` | 0.2–500.0 | Outdoor ground plane | No | `not_implemented` |
| Footpath / Road / Curb | `road.footpath`, `road.carriageway`, `road.kerb` | `null` | 0.1–500.0 | Outdoor ground + edge | Yes | `implemented_unvalidated` |
| Streetlight / Lamppost / Signpost / Bollard / Bench | `street.lamppost`, `signage.sign`, `street.bollard`, `street.bench` | `null` | 0.3–12.0 | Outdoor multi-view | Yes | `implemented_unvalidated` |
| Drain / Manhole | `utility.drainage_component`, `road.manhole` | `null` | 0.1–3.0 | Outdoor ground tile | Yes | `not_implemented` |
| Fence / Gate / Boundary wall | `exterior.fence`, `exterior.gate`, `exterior.boundary_wall` | `null` | 0.5–100.0 | Outdoor multi-view | Yes | `implemented_unvalidated` |

## Domain F: Building & Infrastructure Systems

| Class | Canonical Path | Parent | Geometry | Capability Status |
|---|---|---|---|---|
| Exterior wall | `exterior.wall` | `building` | `surface` | `partially_supported` |
| Facade opening | `exterior.facade_opening` | `exterior.facade` | `parametric` | `not_implemented` |
| Window / Door / Roof / Stair assembly | `opening.window`, `opening.door`, `exterior.roof`, `circulation.stair_assembly` | `building` | `parametric` | `partially_supported` / `not_implemented` |
| Drainage / Lighting / Street infrastructure | `utility.drainage_component`, `street.lighting_infrastructure`, `road.*` | `null` | `proxy_box` | `not_implemented` |
