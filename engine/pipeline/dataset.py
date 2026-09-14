"""Capture-dataset manifest loader (one owner of the manifest contract).

Reads a dataset folder laid out as:

    <dataset>/
      manifest.json    images[], measured_baselines[], intrinsics_px, image_size
      images/<file>

Used by both `scripts/run_vertical_slice.py` (flagship runner) and the
`reality compile` CLI command, so the two entry points cannot drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

from evidence.session import EvidenceItem, EvidenceKind
from provenance import Provenance
from reconstruction.scale import ScaleReference


class DatasetError(ValueError):
    pass


def _uri(path: Path) -> str:
    return path.resolve().as_uri()


def load_capture_dataset(dataset: Path):
    """Dataset folder -> (evidence items, scale references, intrinsics,
    image size). Raises DatasetError for a missing/invalid manifest --
    never silently returns an empty dataset."""
    manifest_path = dataset / "manifest.json"
    if not manifest_path.exists():
        raise DatasetError(f"no manifest.json in {dataset}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"invalid manifest.json in {dataset}: {exc}") from exc

    images_dir = dataset / "images"
    items: List[EvidenceItem] = []
    for entry in manifest.get("images", []):
        img_path = images_dir / entry["file"]
        if not img_path.exists():
            raise DatasetError(f"manifest references missing image: {img_path}")
        data = img_path.read_bytes()
        items.append(
            EvidenceItem(
                id=entry["file"].rsplit(".", 1)[0],
                kind=EvidenceKind.PHOTO,
                source_uri=_uri(img_path),
                sha256=hashlib.sha256(data).hexdigest(),
                metadata={
                    "file": entry["file"],
                    "dataset": manifest.get("dataset", "unknown"),
                },
                provenance=Provenance.OBSERVED,
            )
        )
    if not items:
        raise DatasetError(f"manifest lists no images in {dataset}")

    refs = [
        ScaleReference(
            evidence_id_a=b["evidence_id_a"],
            evidence_id_b=b["evidence_id_b"],
            distance_m=float(b["distance_m"]),
            method=manifest.get("scale_reference", {}).get(
                "method", "manual_measurement"
            ),
        )
        for b in manifest.get("measured_baselines", [])
    ]

    intr: Optional[dict] = manifest.get("intrinsics_px")
    intrinsics = (
        tuple(float(intr[k]) for k in ("fx", "fy", "cx", "cy")) if intr else None
    )
    size = manifest.get("image_size", [1280, 960])
    return items, refs, intrinsics, (int(size[0]), int(size[1]))
