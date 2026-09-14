# Depth Sidecar Ingestion (RGB-D)

Backlog: P1.4 — PARTIAL. Decision 020 (ACCEPTED): 16-bit PNG first.

## Goal

External depth becomes a first-class sensor stream — parsed, scaled,
calibrated, provenance-tracked — not merely "a `depth/` directory was
seen". RGB-D capture then feeds unprojection directly, bypassing
mono-depth estimation.

## Canonical representation

```python
DepthFrame:            # immutable dataclass, provenance-carrying
    timestamp          # sensor-local; P1.6 normalizes to global
    source_id          # coherently linked to the session SourceRecord
    frame_id           # links to the RGB frame
    width, height
    depth_values       # raw array, never mutated post-load
    depth_dtype        # uint16 for PNG; float32 for EXR
    depth_scale        # depth_m = raw * scale  (MUST be explicit)
    invalid_value      # 0 for most devices; masked, never averaged
    units              # "meter"
    camera_intrinsics_ref
    confidence_ref     # optional
    coordinate_frame   # camera frame id
    provenance         # OBSERVED
```

## Format ladder

| Format | Status | Notes |
|---|---|---|
| 16-bit PNG | First | lossless integer, inspectable, Pillow-native |
| EXR | Second | float depth where sub-mm precision matters |
| device raw | Later | explicit per-device adapters only |

## Critical rules

- `depth_meters = raw_value * depth_scale` — raw integers are never
  assumed to be meters.
- Invalid values are masked, not zero-filled or averaged.
- Dimensions must match the paired RGB frame or the frame is rejected.
- Corrupt/unparseable files raise a named error, never silent skips.

## Required tests

PNG round-trip · scale application · invalid masking · missing scale
metadata rejection · dimension mismatch rejection · corrupt file ·
calibration association · provenance round-trip.

## Evidence to advance status

Canonical `DepthFrame` + PNG decoder with the rules above + a real RGB-D
capture ingested end-to-end into the pipeline (unprojection uses the
sidecar instead of mono-depth).
