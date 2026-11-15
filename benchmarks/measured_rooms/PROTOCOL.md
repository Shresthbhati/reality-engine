# Measured-rooms dataset protocol

How the six real rooms that anchor every accuracy claim are chosen, captured, measured and stored.
The ground-truth file format is `ground_truth.schema.json` in this directory (E205). Nothing here is
real data yet: capture is task E208 and needs a person with a phone and a laser meter.

## Rooms

Six rooms. Each room ticks at least two of:

- plain painted walls
- a closed window
- a glass door or a mirror
- a non-rectangular plan
- a sloped or stepped ceiling
- heavy furniture occlusion
- sockets and switches on two walls
- low light

## Consent

Written permission from the occupier, stored outside the repo and referenced by id
(`capture.consent_reference`). No people, documents or screens in frame.

## Capture

- One ordinary phone, default camera app.
- 40–80 photos per room.
- Walk the perimeter facing the centre, then stand at the centre facing each wall.
- Include every corner, floor to ceiling.
- Capture two of the rooms a second time, independently, on a different day.

## Ground truth

Measured with a laser distance meter:

- every wall length
- ceiling height at two points
- each opening's width, height and sill
- each socket and switch: height above the floor and offset from the nearest corner

Measure twice, by two people where possible, and record both sets of numbers.

## Files

```
benchmarks/measured_rooms/<room_id>/
    photos/
    ground_truth.json
    capture.json
    README.md
```

Photos are stored with Git LFS, or outside the repo with their checksums listed in `capture.json`.

## Splits

Rooms 1–4 are development rooms. Rooms 5–6 are the blind hold-out: their ground truth is never
opened by anyone who is changing reconstruction code.

## Retention

Delete a room's photos and data on the occupier's request within 7 days.
