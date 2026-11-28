"""Columnar store of every observation in a job (BUILD_SPEC B02 / 2.2).

Struct-of-arrays so grouping by image / entity is a ``np.unique`` call, not a Python loop. Converts to and from
``world_ir.observations.Observation`` so nothing else changes type, and emits the benchmark inventory contract
with every unmeasured field ``None`` plus a stated reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from world_ir.observations import DepthSource, Observation, Visibility

_VIS = {Visibility.FULL: 0, Visibility.PARTIAL: 1, Visibility.OCCLUDED: 2, Visibility.NOT_OBSERVED: 3}
_VIS_INV = {v: k for k, v in _VIS.items()}
_VIS_NAME = {0: "visible", 1: "partial", 2: "occluded", 3: "not_observed"}
_SCHEMA = {  # name -> (dtype, trailing shape)
    "obs_id": ("<U32", ()), "image_id": ("<U32", ()), "class_path": ("<U96", ()), "parent_obs": ("int32", ()),
    "bbox_px": ("float32", (4,)), "score": ("float32", ()), "mask_ref": ("int32", ()), "emb_row": ("int32", ()),
    "world_pos": ("float32", (3,)), "sigma_m": ("float32", ()), "visibility": ("int8", ()), "entity_row": ("int32", ()),
    "model": ("<U48", ()), "attrs_json": ("<U2000", ()),
}


def _group(keys: np.ndarray) -> dict:
    if len(keys) == 0:
        return {}
    uniq, inv = np.unique(keys, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    cuts = np.cumsum(np.bincount(inv))[:-1]
    return {u.item() if hasattr(u, "item") else u: rows for u, rows in zip(uniq, np.split(order, cuts))}


@dataclass
class ObservationTable:
    obs_id: np.ndarray
    image_id: np.ndarray
    class_path: np.ndarray
    parent_obs: np.ndarray
    bbox_px: np.ndarray
    score: np.ndarray
    mask_ref: np.ndarray
    emb_row: np.ndarray
    world_pos: np.ndarray
    sigma_m: np.ndarray
    visibility: np.ndarray
    entity_row: np.ndarray
    model: np.ndarray
    attrs_json: np.ndarray
    by_image: dict = field(default_factory=dict, repr=False)
    by_entity: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.reindex()

    def __len__(self) -> int:
        return len(self.obs_id)

    def reindex(self) -> None:
        self.by_image = _group(self.image_id)
        placed = np.flatnonzero(self.entity_row >= 0)
        g = _group(self.entity_row[placed])
        self.by_entity = {int(k): placed[v] for k, v in g.items()}

    @classmethod
    def empty(cls, n: int = 0) -> "ObservationTable":
        cols = {}
        for k, (dt, sh) in _SCHEMA.items():
            cols[k] = np.zeros((n, *sh), dtype=dt)
        for k in ("parent_obs", "mask_ref", "emb_row", "entity_row"):
            cols[k][:] = -1
        cols["world_pos"][:] = np.nan; cols["sigma_m"][:] = np.nan; cols["visibility"][:] = -1
        return cls(**cols)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{k: getattr(self, k) for k in _SCHEMA})

    @classmethod
    def load(cls, path: str | Path) -> "ObservationTable":
        z = np.load(path, allow_pickle=False)
        return cls(**{k: z[k] for k in _SCHEMA})

    @classmethod
    def from_observations(cls, obs: Sequence[Observation]) -> "ObservationTable":
        t = cls.empty(len(obs))
        for i, o in enumerate(obs):
            t.obs_id[i], t.image_id[i], t.class_path[i], t.model[i] = o.observation_id, o.image_id, o.label, o.model
            t.world_pos[i], t.sigma_m[i], t.score[i] = o.world_pos, o.sigma_m, (o.confidence if o.confidence is not None else float("nan"))
            if o.bbox_px:
                t.bbox_px[i] = o.bbox_px
            if o.visibility is not None:
                t.visibility[i] = _VIS[o.visibility]
        t.reindex()
        return t

    def to_observations(self) -> list[Observation]:
        out = []
        for i in range(len(self)):
            v = int(self.visibility[i])
            out.append(Observation(
                observation_id=str(self.obs_id[i]), image_id=str(self.image_id[i]), label=str(self.class_path[i]),
                world_pos=tuple(float(x) for x in self.world_pos[i]), sigma_m=float(self.sigma_m[i]),
                depth_source=DepthSource.RELATIVE, model=str(self.model[i]), confidence=(None if self.score[i] != self.score[i] else float(self.score[i])),
                bbox_px=tuple(float(x) for x in self.bbox_px[i]), visibility=_VIS_INV.get(v),
            ))
        return out

    def to_inventory(self) -> list[dict]:
        """Benchmark contract rows; anything not measured is None with an ``unavailable`` reason."""
        recs = []
        # RE-PERC-000022: a count is DISTINCT ENTITIES of the class in the photo after association, never detections
        entities_of: dict = {}
        for i in range(len(self)):
            if self.entity_row[i] >= 0:
                entities_of.setdefault((str(self.image_id[i]), str(self.class_path[i])), set()).add(int(self.entity_row[i]))
        for i in range(len(self)):
            path = str(self.class_path[i]).split("/")
            lifted = bool(np.all(np.isfinite(self.world_pos[i])))
            vis_val = _VIS_NAME.get(int(self.visibility[i]))
            r = {
                "image_id": str(self.image_id[i]),
                "entity_id": f"ent-{int(self.entity_row[i])}" if self.entity_row[i] >= 0 else f"obs-{self.obs_id[i]}",
                "category": path[0], "subtype": path[-1], "bbox": [float(x) for x in self.bbox_px[i]],
                "confidence": float(min(max(self.score[i], 0.0), 1.0)),
                "position_3d": [float(x) for x in self.world_pos[i]] if lifted else None,
                "visibility": None if vis_val == "not_observed" else vis_val,
                "instance_count": (len(entities_of[(str(self.image_id[i]), str(self.class_path[i]))])
                                   if self.entity_row[i] >= 0 else None),
                "mask": int(self.mask_ref[i]) if self.mask_ref[i] >= 0 else None,
                "dimensions": None, "orientation": None, "material": None,
                "evidence": [str(self.obs_id[i])],
            }
            why = {k: "not measured by this pipeline" for k in ("dimensions", "orientation", "material")}
            if r["instance_count"] is None:
                why["instance_count"] = "not associated to an entity yet (raw detections are never counted)"
            if r["mask"] is None:
                why["mask"] = "no mask produced"
            if r["position_3d"] is None:
                why["position_3d"] = "not lifted to 3D"
            if r["visibility"] is None:
                why["visibility"] = "not observed" if vis_val == "not_observed" else "not assessed"
            r["unavailable"] = why
            recs.append(r)
        return recs

    def to_rows(self) -> list[dict]:
        rows = []
        for i in range(len(self)):
            pos = [float(x) for x in self.world_pos[i]] if np.all(np.isfinite(self.world_pos[i])) else None
            rows.append({
                "obs_id": str(self.obs_id[i]),
                "image_id": str(self.image_id[i]),
                "class_path": str(self.class_path[i]),
                "parent_obs": int(self.parent_obs[i]),
                "bbox_px": [float(x) for x in self.bbox_px[i]],
                "score": float(self.score[i]),
                "mask_ref": int(self.mask_ref[i]),
                "emb_row": int(self.emb_row[i]),
                "world_pos": pos,
                "sigma_m": float(self.sigma_m[i]) if np.isfinite(self.sigma_m[i]) else None,
                "visibility": _VIS_NAME.get(int(self.visibility[i])),
                "entity_row": int(self.entity_row[i]),
                "model": str(self.model[i]),
                "attrs_json": str(self.attrs_json[i]),
            })
        return rows
