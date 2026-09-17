## Summary

Implements SAM (Segment Anything Model, Meta AI) segmentation backend behind the `ISegmentationBackend` interface — the first concrete segmentation implementation for Reality Engine.

## Changes

- **`perception/segmentation/sam_backend.py`** — Real SAM segmentation backend implementing `ISegmentationBackend`
  - Supports ViT-B, ViT-L, ViT-H model types with configurable `SamAutomaticMaskGenerator` parameters
  - Auto-detects CUDA, falls back to CPU; optional local checkpoint path for air-gapped environments
  - Lazy model loading via `torch.hub` (downloads on first use)
  - Produces class-agnostic masks with `label="mask_N"` and `predicted_iou` as confidence
  - Carries `model_name="sam"` and `model_version` in `SegmentationResult` for provenance (spec §48)
  - Raises `SegmentationBackendUnavailableError` when perception deps missing — never fakes output

- **`perception/segmentation/__init__.py`** — Exports backend + interface + error types

- **`tests/test_sam_backend.py`** — 20 tests (19 passing, 1 skipped without deps):
  - Interface contract (ABC enforcement)
  - SegmentedRegion/SegmentationResult validation (confidence bounds, mask structure)
  - Missing-dependency error paths (torch, torchvision, numpy)
  - Device selection logic (CUDA auto-detection, explicit override)
  - Model type validation
  - Mask generator config storage
  - Provenance fields verification

## Design Decisions

- Follows same pattern as `MiDaSDepthBackend` and `ColmapReconstructionBackend`: honest error handling, no fabricated output
- Chosen per `docs/TECHNOLOGY_REGISTRY.md`: SAM (original) over SAM 2 for photo-only pipeline (SAM2's video/temporal tracking unused)
- SAM has Apache 2.0 license (code + checkpoints) — cleanest licensing story
- Addresses Capability Matrix Goal O: "Perception adapter interfaces" — now has first concrete segmentation implementation
- Pairs with MiDaS depth backend (already merged) for 2D→3D instance lifting via `ITrackBackend` (next)

## Testing

```bash
# All unit tests (no perception deps required)
pytest tests/test_sam_backend.py -v
# 19 passed, 1 skipped (integration test requires: pip install -e .[perception])

# Full suite (excluding pre-existing room inference failures)
pytest tests/ -v --ignore=tests/test_room_inference.py
# 780 passed, 2 skipped
```

## Limitations

- SAM produces *class-agnostic* masks (no semantic labels). Downstream mapping to Reality Engine ontology (`EntityType.WALL`, `DOOR`, etc.) is separate work.
- Next step: Implement `ITrackBackend` for 2D→3D instance lifting (appearance-embedding re-id + geometric consistency)
- Real inference not tested in CI (deps not installed). Skipped integration test validates full pipeline when deps present.