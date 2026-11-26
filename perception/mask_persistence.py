"""Persist ObservationRecord segmentation masks into the run's MaskStore and
stamp each record with its persisted ``mask_ref`` row index.

Masks are large image-plane bool arrays; they are stored run-length-encoded
(one ``.npz`` per job, never a PNG per mask) in
``world_ir.mask_store.MaskStore``. ``ObservationRecord`` keeps the in-memory
array (``mask``) plus the persisted index (``mask_ref``), so downstream
consumers (association, fusion, lifting) can use the array without re-decoding
from the store, and a reloaded run can recover the mask bytes from ``mask_ref``.

This is the production connection the A4 run was missing: SAM 2 computed masks
and the adapter stored only ``mask_area_px`` in metadata, so the mask pixels
were discarded before reaching MaskStore, ObservationTable, or any consumer.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from perception.engine_evidence import ObservationRecord
from world_ir.mask_store import MaskStore


def persist_masks(
    records: Sequence[ObservationRecord],
    store: MaskStore,
) -> list[ObservationRecord]:
    """Encode every record's in-memory mask into ``store`` and return records
    with ``mask_ref`` stamped to the persisted row.

    Records without a mask are returned unchanged (``mask_ref`` stays None).
    The returned list is new (records are frozen dataclasses); records keep
    their in-memory ``mask`` array so downstream consumers need not re-decode.
    """
    out: list[ObservationRecord] = []
    for rec in records:
        if rec.mask is None:
            out.append(rec)
            continue
        arr = np.asarray(rec.mask, dtype=bool)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])
        row = store.add(arr)
        out.append(replace(rec, mask_ref=int(row)))
    return out


def save_mask_store(store: MaskStore, path: str | Path) -> None:
    """Persist the mask store to disk atomically (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=str(path.parent))
    os.close(fd)  # close the mkstemp handle so Windows can replace/remove the file
    try:
        store.save(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_mask_store(path: str | Path) -> MaskStore:
    """Load a mask store previously written with :func:`save_mask_store`."""
    return MaskStore.load(path)


__all__ = ["persist_masks", "save_mask_store", "load_mask_store"]
