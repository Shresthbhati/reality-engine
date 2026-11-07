# Measured rooms (task H-3.2 data, task 12.2 report)

The engine is only as accurate as it can be shown to be against a real, measured room. This folder holds the measurements and the
tool that compares them with what the engine built.

**Status: tooling done, data missing.** No real room has been measured yet. The accuracy of the engine on real rooms is unknown.

## What to measure (about 20 minutes per room)

Six rooms, chosen to stress different things: a small room, a large room, an L-shaped room, a room with a sloped ceiling, a room with
many windows, a room with a stair. Use a laser measure if you have one (a tape is fine; say which in `method`).

For each room, measure and write down:

- every **wall length**, along the floor, from corner to corner (not around furniture)
- the **ceiling height** (floor to ceiling, in the middle of the room)
- the **floor area** (or the two side lengths of a rectangle; the area is `length x width`)
- every **window and door**: width, height, and the height of the sill above the floor (a door's sill is 0)

Then take the photographs the way a customer would, and keep them next to the measurements.

## File layout

    benchmarks/measured_rooms/rooms/<room-name>/
        truth.json          the measurements, below
        photos/             the photographs of that room (do not commit them if they are large or not yours to publish)

## `truth.json` (schema `measured_room/1`)

    {
      "schema": "measured_room/1",
      "room": "living room, flat 3B",
      "method": "laser",
      "ceiling_height_m": 2.55,
      "floor_area_m2": 17.3,
      "walls": [ {"id": "north", "length_m": 4.12}, {"id": "east", "length_m": 4.20} ],
      "openings": [
        {"kind": "window", "width_m": 1.20, "height_m": 1.40, "sill_m": 0.90},
        {"kind": "door",   "width_m": 0.90, "height_m": 2.05, "sill_m": 0.0}
      ]
    }

Metres everywhere. `kind` is `window` or `door`. Leave out anything you did not measure; the report only checks what is there.

## Running the report

1. Build the room in the app from its photographs. Note the world id.
2. Save the engine's world: `GET /api/worlds/<id>/worldir` to a file, for example `world.json`.
3. Compare:

       python -m benchmarks.measured_rooms.report benchmarks/measured_rooms/rooms/<room>/truth.json world.json
       python -m benchmarks.measured_rooms.report truth.json world.json --calibrate     # separate scale error from shape error
       python -m benchmarks.measured_rooms.report truth.json world.json --json          # machine-readable

The exit code is 0 only when every element is inside tolerance and no wall was missed.

## How to read it

- **Result** is PASS / FAIL against the tolerance, or MISS when the engine never produced that wall or opening (a missed wall is
  not skipped: it counts).
- **Tolerances are PROPOSED**: wall length, ceiling height and opening width within 2 cm; room area within 1 %. They are for rooms up to
  6 x 6 m and are to be confirmed with architecture and interior-design firms (task H-0.1).
- **Stated +/- / Inside it**: the engine says how sure it is of each number. "Inside it = NO" means the engine was over-confident,
  which is a finding in itself. The summary line gives the share of numbers whose truth fell inside the stated range.
- **Scale**: while the world is in model units (before you press Set scale) the engine's numbers are compared as if they were
  metres, and the report says so. `--calibrate` fits one scale factor from the measured walls; whatever error remains after that is shape
  error (wrong angles, wrong proportions), which is what the plane-fitting fixes in T-3.3 have to remove.
- The room area the engine reports from a single photograph is the visible part only (a lower bound), and is marked as such.

## What this is not

It does not measure the engine on photographs of the wrong room, on generated images, or on a room whose measurements were taken
after furniture was moved. Photograph and measure the same room on the same day.
