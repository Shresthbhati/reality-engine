"""Real-data indoor & outdoor multi-engine perception validation (Phases 7-10).

Runs the production multi-engine perception stack (`run_multi_engine_for_job`)
on actual physical photographs from `datasets/real_room_capture` (indoor) and
`datasets/south_building/images` (outdoor), measuring detections by class,
small-object detections, segmentation masks, two-view geometry, runtime, and
peak VRAM without synthetic substitution.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from perception.engine_adapters import register_all_engines
from perception.engine_orchestrator import MultiEngineOrchestrator
from perception.mask_persistence import persist_masks
from world_ir.mask_store import MaskStore


def _run_dataset(name: str, paths: list[Path], domain: str, tile_scale: int = 2) -> dict:
    t0 = time.perf_counter()
    orch = MultiEngineOrchestrator()
    report = orch.run_all(
        [str(p) for p in paths],
        context={
            "run_id": f"val_{name}",
            "domain": domain,
            "tile_scale": tile_scale,
            "tile_overlap": 0.2,
            "query_budget": 40,
            "small_share": 0.4,
        },
    )
    mask_store = MaskStore()
    stamped = persist_masks(list(report.observations), mask_store)

    by_engine = Counter(o.model_id.engine_id for o in stamped)
    det_classes = Counter(
        o.metadata.get("ontology_path") or o.label
        for o in stamped
        if o.model_id.engine_id == "detection"
    )
    small_classes = Counter(
        o.metadata.get("ontology_path") or o.label
        for o in stamped
        if o.model_id.engine_id == "detection" and o.metadata.get("is_small")
    )
    return {
        "dataset": name,
        "domain": domain,
        "image_count": len(paths),
        "images": [p.name for p in paths],
        "engines_run": dict(report.engines_run),
        "unavailable": list(report.unavailable),
        "n_observations": len(stamped),
        "observations_by_engine": dict(by_engine),
        "masks_persisted": len(mask_store),
        "top_detection_classes": dict(det_classes.most_common(20)),
        "small_object_classes": dict(small_classes),
        "vram_peak_mb": report.vram_peak_mb,
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "engine_executions": [
            {
                "engine_id": e.engine_id,
                "status": e.status,
                "device": e.device,
                "wall_time_s": round(e.wall_time_s, 2),
                "peak_vram_mb": e.peak_vram_mb,
                "n_observations": e.n_observations,
                "error": e.error,
            }
            for e in report.engine_executions
        ],
    }


def main() -> int:
    register_all_engines()
    indoor_imgs = sorted(Path("datasets/real_room_capture").glob("IMG_*.jpg"))[:4]
    outdoor_imgs = sorted(Path("datasets/south_building/images").glob("P*.JPG"))[:4]

    results = {
        "indoor": _run_dataset("real_room_capture_4", indoor_imgs, domain="indoor", tile_scale=2),
        "outdoor": _run_dataset("south_building_4", outdoor_imgs, domain="outdoor", tile_scale=2),
    }
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
