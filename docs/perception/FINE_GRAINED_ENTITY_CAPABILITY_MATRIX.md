# Fine-Grained Entity Capability Matrix

Status legend (strictly enforced by `perception/ontology.py`):
- `implemented_and_tested`: Grounded in `config/ontology.yaml`, detected/lifted in code, and covered by automated tests.
- `implemented_unvalidated`: Grounded in ontology and prompts, runs in pipeline, lacks annotated GT AP/IoU validation.
- `partially_supported`: Detected at coarse/parent level or requires higher-res/tiling; sub-parts incomplete.
- `not_implemented`: Not yet supported end-to-end.

## Domain A: Interior Architecture
| Entity Class | Ontology Grounding | Pipeline Support | Status |
| :--- | :--- | :--- | :--- |
| Walls, Floors, Ceilings | `architecture/wall`, `floor`, `ceiling` | Grounding DINO + Mask2Former/SAM2 + Planar lifting | `implemented_and_tested` |
| Doors & Windows | `architecture/door`, `window` | Grounding DINO + Mask R-CNN + Depth | `implemented_and_tested` |
| Door/Window Frames, Openings | `opening.door.frame`, `opening.opening` | Grounding DINO tiled prompts | `implemented_unvalidated` |
| Columns, Beams, Stairs, Railings | `architecture/column`, `circulation.stair_assembly`, `circulation.railing.glass` | Grounding DINO + Real-photo verified | `implemented_unvalidated` |
| Skirting, Cornices, Mouldings, Panels | `architecture/skirting`, `wall_panel` | Grounding DINO tiled prompts | `partially_supported` |

## Domain B: Electrical & MEP Details
| Entity Class | Ontology Grounding | Pipeline Support | Status |
| :--- | :--- | :--- | :--- |
| Switches, Sockets, Plug Points | `fixtures/electrical/switch`, `socket` | Small-object tiled Grounding DINO (`tile_scale=2`) | `implemented_and_tested` |
| Switchboards & Modular Inserts | `electrical.switchboard.module.*`, `electrical.junction_box` | Grounding DINO tiled + Real-photo verified | `implemented_unvalidated` |
| Light Fixtures, Ceiling Fans | `fixtures/lighting/*`, `ceiling_fan` | Grounding DINO + Scene-graph ceiling prior | `implemented_unvalidated` |
| Smoke Detectors, Fire Alarms, Sprinklers | `fixtures/safety/*` | Open-vocab prompts | `partially_supported` |
| Conduits, Cable Trays, Pipes, Valves, Meters | `mep/*` | Open-vocab prompts; thin/occluded geometry weak | `partially_supported` |

## Domain C: Furniture & Interior Objects
| Entity Class | Ontology Grounding | Pipeline Support | Status |
| :--- | :--- | :--- | :--- |
| Tables, Chairs, Beds, Sofas, Desks | `furniture/*` | Mask R-CNN + Grounding DINO + SAM2 | `implemented_and_tested` |
| Cabinets, Shelves, Storage, Appliances | `furniture/storage/*`, `appliances/*` | Mask R-CNN + Grounding DINO | `implemented_and_tested` |
| Door/Drawer Components, Handles, Hinges | `hardware/handle`, `hinge` | Tiled Grounding DINO + `part_of` scene graph | `partially_supported` |

## Domain D: Outdoor Environments
| Entity Class | Ontology Grounding | Pipeline Support | Status |
| :--- | :--- | :--- | :--- |
| Building Facades, Roofs, Balconies | `outdoor/building_facade`, `roof` | Grounding DINO + South Building verified | `implemented_unvalidated` |
| Roads, Footpaths, Kerbs, Markings | `outdoor/road`, `footpath` | Grounding DINO + Depth | `partially_supported` |
| Streetlights, Traffic Signs, Utility Poles | `outdoor/streetlight`, `traffic_sign` | Tiled Grounding DINO | `partially_supported` |
| Trees, Shrubs, Grass, Planters, Fences, Gates | `vegetation/*`, `outdoor/fence` | Grounding DINO + SAM2 | `implemented_unvalidated` |

## Domain E: Small & Difficult Entities
| Entity Class | Min Pixel Target | Pipeline Support | Status |
| :--- | :--- | :--- | :--- |
| Plug points, switches, handles, hinges | `>= 16×16` px in tiled crop | `tile_scale=2` + `small_share=0.4` + `MaskStore` | `implemented_and_tested` |
| Fasteners, thin wires, transparent glass | `< 16×16` px or specular | Challenging failure regime documented | `partially_supported` |
