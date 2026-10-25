# Phase D slice: completion mode (backend)

Goal: given a world whose shell was RECONSTRUCTED/OBSERVED and whose interior is unknown, propose labelled
`GENERATED` interiors inside the measured footprint, WITHOUT touching the committed world until accepted.

## Shell and footprint (honest input)
- Shell = non-generated entities of type FLOOR/WALL/ROOM with provenance RECONSTRUCTED or OBSERVED and a geometry
  bounding box. Footprint = axis-aligned bounding rectangle of their x/z extent (world Y-up), floor level = top of
  the lowest FLOOR (else y-min of the shell).
- No such entities, or an extent < 2 x 2 -> 422 stating why. Axis-aligned bounding rectangle only: an L-shaped or
  rotated shell is approximated by its bbox (reported as `footprint.method = "axis_aligned_bbox"`).
- Units: `model_units` unless the world has a metric/authored measurement (any `Measurement.scale_state` in
  metric/authored); the grammar itself works in whatever unit the geometry is in, so in model units the room-area
  program is only meaningful if the user supplies areas in model units^2. Reported in `footprint.unit`.

## Proposals
- Table `completion_proposals` (new, `create_all`, same pattern as `world_intents`): id, world_id, base_version_id,
  status (proposed|accepted|rejected), alt_index, seed, spec (GeneratedSpec dict), summary (footprint, counts),
  accepted_version_id, timestamps.
- Program = user rooms (`ApartmentRoomIn`-shaped) or built-in template `studio`/`one_bed`. The apartment
  `layout_solver` is run with `footprint = (w, d)`; 1-3 distinct plans (`max_alternatives`) -> one proposal each.
  Deterministic in (shell, program, seed).
- Creating a proposal never commits. Generated entities are computed against a copy of HEAD only for the diff.

## API (`/api/worlds/{id}/completion/...`)
- `POST proposals` -> 201 list; `GET proposals`; `GET proposals/{pid}/diff` (added/changed/removed ids, provenance
  counts before/after, `generated_label`); `POST proposals/{pid}/accept`; `POST proposals/{pid}/reject`.
- Accept re-derives the shell; footprint changed since proposal -> 409 stale. Commit goes through
  `_generate_and_commit` (HEAD-moved retry); entities keep `GENERATED`, each gets
  `custom_properties.completion_proposal = pid`, spec stored by `spec_store`. Name collision -> 409.

## Not in this slice
UI, non-rectangular footprints, multi-storey, furniture, door/window alignment with the reconstructed shell
openings, re-proposal on stale (user must create a new proposal), real-data accuracy (Phase E).
