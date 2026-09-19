## Summary

First real perception backend for Reality Engine — implements MiDaS (Intel ISL) depth estimation behind the `IDepthBackend` interface.

## Changes

- **`perception/depth/midAS_backend.py`** — Real MiDaS depth backend implementing `IDepthBackend`
  - Supports 4 model types: `MiDaS_small`, `DPT_Large`, `DPT_Hybrid`, `DPT_Small`
  - Auto-detects CUDA, falls back to CPU; optional local checkpoint path for air-gapped environments
  - Lazy model loading via `torch.hub` (downloads on first use)
  - Produces `DepthMap` with `unit="relative"` (honest — MiDaS outputs relative depth)
  - Raises `DepthBackendUnavailableError` when perception deps missing — never fakes output

- **`pyproject.toml`** — Added optional `perception` dependency group:
  ```toml
  perception = ["torch>=2.0", "torchvision>=0.15", "numpy>=1.24"]
  ```

- **`tests/test_midas_backend.py`** — 17 tests (16 passing, 1 skipped without deps):
  - Interface contract (ABC enforcement)
  - DepthMap validation (shape, bounds, unit field)
  - Missing-dependency error paths (torch, torchvision, numpy)
  - Device selection logic (CUDA auto-detection, explicit override)
  - Model type validation

## Design Decisions

- Follows same pattern as `ColmapReconstructionBackend`: honest error handling, no fabricated output
- Chosen per `docs/TECHNOLOGY_REGISTRY.md`: MiDaS has MIT license (code + checkpoints) — cleanest licensing story
- Addresses `docs/REALITY_ENGINE_AUDIT.md` item 4: "install and adapt one real depth or segmentation backend"
- Relates to Capability Matrix Goal O: "Perception adapter interfaces" — now has first concrete implementation

## Testing

```bash
# All unit tests (no perception deps required)
pytest tests/test_midas_backend.py -v

# Full suite
pytest tests/ -v
# 741 passed, 1 skipped (integration test requires: pip install -e .[perception])
```

## Limitations

- MiDaS produces *relative* depth (not metric). Fusion with COLMAP's metric sparse points requires a scale-recovery step — not yet built.
- Next step: Implement `ISegmentationBackend` (SAM) per `TECHNOLOGY_REGISTRY.md`, then build depth-segmentation fusion for 3D instance lifting.