"""Correction provenance, version integrity and entity lineage (Sprint 2, workstream 6).

A correction is only trustworthy if it is (1) made against a version the caller actually saw, (2) applied to a COPY,
validated and committed atomically, and (3) auditable afterwards: who, what, from which version, which values changed,
what evidence was kept, and which derived artifacts (exports, floor plans, calibrations) no longer match the world.

This module is pure (no HTTP, no DB). The API routes and the store-level transaction below share it.

* :class:`CorrectionRecord` -- the typed audit record, persisted in ``world.metadata["correction_log"]``.
* :func:`require_base_version` -- mandatory, never silent, base-version check.
* :func:`stamp_correction` -- diff base vs corrected world and append the record.
* :func:`commit_correction_atomically` -- copy -> mutate -> validate -> save; the committed head is untouched on failure.
* lineage helpers -- split/merge/reclassify/rebuild keep ``parent -> child`` edges in ``world.metadata["entity_lineage"]``.
* :func:`classify_artifacts` -- flags derived artifacts built from an older version as stale, by version id.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

LOG_KEY = "correction_log"
LINEAGE_KEY = "entity_lineage"

#: custom_properties keys that hold recomputed geometric values
_RECOMPUTED_KEYS = frozenset({
    "length_m", "width_m", "height_m", "depth_m", "area_m2", "floor_area_m2", "thickness_m",
    "floor_dimension_u_m", "floor_dimension_v_m", "dimensions", "bounding_box",
})
_MAX_VALUE_CHARS = 4000


class BaseVersionRequired(ValueError):
    """The caller did not say which version it was looking at."""


class StaleBaseVersion(ValueError):
    """The caller's base version is not the current version."""

    def __init__(self, base: str, current: str):
        super().__init__(
            f"stale: this correction was made against version {base} but the current version is {current}; "
            "reload and retry")
        self.base, self.current = base, current


class CorrectionRejected(ValueError):
    """The corrected copy failed validation or evidence preservation; nothing was committed."""


def require_base_version(provided: Optional[str], current: Optional[str]) -> str:
    """Return the verified base version id. Missing -> BaseVersionRequired; not the current head -> StaleBaseVersion."""
    if not provided or not str(provided).strip():
        raise BaseVersionRequired(
            "a base version is required: send the id of the version you were looking at "
            f"(expected_version_id / parent_version_id). The current version is {current}")
    if current is not None and provided != current:
        raise StaleBaseVersion(provided, current)
    return provided


@dataclass
class FieldChange:
    entity_id: str
    field: str
    old: Any = None
    new: Any = None


@dataclass
class CorrectionRecord:
    record_id: str
    operation: str                       # reclassify | split | merge | dimension | commit | calibrate ...
    actor_id: str                        # user id, or the system component name for actor_kind == "system"
    actor_kind: str                      # "user" | "system"
    created_at: float
    base_version_id: str
    new_version_id: str
    entity_ids: List[str] = field(default_factory=list)
    changes: List[FieldChange] = field(default_factory=list)          # old -> new, per entity field
    lineage: List[Dict[str, Any]] = field(default_factory=list)       # {from_ids, to_id, relation}
    changed_observation_ids: List[str] = field(default_factory=list)  # added / removed / modified observation ids
    preserved_observation_ids: List[str] = field(default_factory=list)  # present and byte-identical after the correction
    evidence_untouched: bool = True      # no original observation was removed or modified
    recomputed_values: List[Dict[str, Any]] = field(default_factory=list)   # dimensions / areas that changed
    stale_artifacts: List[Dict[str, Any]] = field(default_factory=list)     # derived artifacts that no longer match
    note: Optional[str] = None

    def missing_fields(self) -> List[str]:
        """Names of required audit fields that are empty (a complete record returns [])."""
        out = [k for k in ("record_id", "operation", "actor_id", "actor_kind", "base_version_id", "new_version_id")
               if not getattr(self, k)]
        if self.actor_kind not in ("user", "system"):
            out.append("actor_kind")
        if not self.created_at:
            out.append("created_at")
        if not self.entity_ids:
            out.append("entity_ids")
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CorrectionRecord":
        d = dict(d)
        d["changes"] = [FieldChange(**c) for c in d.get("changes") or []]
        return CorrectionRecord(**d)


def _brief(v: Any) -> Any:
    raw = json.dumps(v, sort_keys=True, default=str)
    if len(raw) <= _MAX_VALUE_CHARS:
        return json.loads(raw)
    return {"_truncated": True, "sha256": hashlib.sha256(raw.encode()).hexdigest(), "chars": len(raw)}


def _entity_dict(e: Any) -> dict:
    return e.to_dict()


def _observations(world: Any) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for e in (world.entities or {}).values():
        for o in e.observations or []:
            out[o.id] = o.to_dict()
    return out


_VOLATILE = frozenset({"user_decision", "user_corrections"})


def diff_entities(base: Any, new: Any) -> tuple[List[FieldChange], List[dict]]:
    """Field-level old->new changes between two worlds, and the recomputed geometric values among them."""
    changes: List[FieldChange] = []
    recomputed: List[dict] = []
    be, ne = base.entities or {}, new.entities or {}
    for eid in sorted(set(be) | set(ne)):
        if eid not in ne:
            changes.append(FieldChange(eid, "entity", "present", "removed"))
            continue
        if eid not in be:
            changes.append(FieldChange(eid, "entity", None, "added"))
            continue
        a, b = _entity_dict(be[eid]), _entity_dict(ne[eid])
        for k in sorted(set(a) | set(b)):
            if k in ("custom_properties", "observations"):
                continue
            if a.get(k) != b.get(k):
                changes.append(FieldChange(eid, k, _brief(a.get(k)), _brief(b.get(k))))
        ca, cb = a.get("custom_properties") or {}, b.get("custom_properties") or {}
        for k in sorted(set(ca) | set(cb)):
            if ca.get(k) == cb.get(k):
                continue
            if k not in _VOLATILE:
                changes.append(FieldChange(eid, f"custom_properties.{k}", _brief(ca.get(k)), _brief(cb.get(k))))
            if k in _RECOMPUTED_KEYS:
                recomputed.append({"entity_id": eid, "quantity": k, "old": _brief(ca.get(k)), "new": _brief(cb.get(k))})
    return changes, recomputed


def stamp_correction(
    base: Any,
    new: Any,
    *,
    operation: str,
    actor_id: str,
    actor_kind: str = "user",
    base_version_id: str,
    new_version_id: str,
    entity_ids: Sequence[str] = (),
    note: Optional[str] = None,
    lineage: Sequence[dict] = (),
    artifacts: Iterable[dict] = (),
    created_at: Optional[float] = None,
) -> CorrectionRecord:
    """Build the audit record for ``base -> new`` and append it to ``new.metadata['correction_log']``.

    ``artifacts`` are derived artifacts known to be built from ``base_version_id`` (each ``{artifact_id, kind,
    built_from_version_id}``); every one is stale as of ``new_version_id``. A calibrated-scale determination carried by
    ``base`` is also stale when the correction changed geometry."""
    changes, recomputed = diff_entities(base, new)
    bo, no = _observations(base), _observations(new)
    changed_obs = sorted({i for i in set(bo) | set(no) if bo.get(i) != no.get(i)})
    preserved = sorted(i for i in bo if i in no and bo[i] == no[i])
    stale = [dict(a) for a in artifacts if a.get("built_from_version_id") != new_version_id]
    cal = (getattr(base, "metadata", None) or {}).get("scale_calibration")
    geometry_changed = bool(recomputed) or any(c.field in ("transform", "geometry_ids") for c in changes)
    if isinstance(cal, dict) and cal.get("id") and geometry_changed:
        stale.append({"artifact_id": cal["id"], "kind": "scale_calibration", "built_from_version_id": base_version_id})
    ids = list(dict.fromkeys(list(entity_ids) or [c.entity_id for c in changes]))
    rec = CorrectionRecord(
        record_id=f"corr-{uuid.uuid4().hex[:12]}", operation=operation, actor_id=str(actor_id or ""),
        actor_kind=actor_kind, created_at=float(created_at if created_at is not None else time.time()),
        base_version_id=base_version_id, new_version_id=new_version_id, entity_ids=ids, changes=changes,
        lineage=[dict(x) for x in lineage], changed_observation_ids=changed_obs, preserved_observation_ids=preserved,
        evidence_untouched=not any(i in bo and (i not in no or bo[i] != no[i]) for i in bo),
        recomputed_values=recomputed, stale_artifacts=stale, note=note or None)
    for edge in lineage:
        for src in edge.get("from_ids") or []:
            add_lineage_edge(new, src, edge["to_id"], edge["relation"], record_id=rec.record_id)
    new.metadata[LOG_KEY] = list(new.metadata.get(LOG_KEY) or []) + [rec.to_dict()]
    return rec


def correction_log(world: Any) -> List[CorrectionRecord]:
    return [CorrectionRecord.from_dict(d) for d in (getattr(world, "metadata", None) or {}).get(LOG_KEY) or []]


# --------------------------------------------------------------------------------------------------------- lineage

def add_lineage_edge(world: Any, from_id: str, to_id: str, relation: str, *, record_id: Optional[str] = None) -> None:
    """Record that ``to_id`` descends from ``from_id`` (split / merge / reclassify / rebuild / carried). Idempotent."""
    if from_id == to_id and relation in ("split", "merge"):
        return
    edges = list(world.metadata.get(LINEAGE_KEY) or [])
    edge = {"from_id": from_id, "to_id": to_id, "relation": relation}
    if record_id:
        edge["record_id"] = record_id
    if not any(e["from_id"] == from_id and e["to_id"] == to_id and e["relation"] == relation for e in edges):
        edges.append(edge)
    world.metadata[LINEAGE_KEY] = edges


def lineage_successors(world: Any, entity_id: str) -> List[str]:
    """Where an id went: itself when it is still in the world, else the transitive descendants that are."""
    ents = world.entities or {}
    edges = (getattr(world, "metadata", None) or {}).get(LINEAGE_KEY) or []
    out: List[str] = []
    seen: set = set()
    stack = [entity_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in ents:
            out.append(cur)
        for e in edges:
            if e["from_id"] == cur:
                stack.append(e["to_id"])
    return sorted(set(out))


def lineage_ancestors(world: Any, entity_id: str) -> List[str]:
    edges = (getattr(world, "metadata", None) or {}).get(LINEAGE_KEY) or []
    out: set = set()
    stack = [entity_id]
    while stack:
        cur = stack.pop()
        for e in edges:
            if e["to_id"] == cur and e["from_id"] not in out:
                out.add(e["from_id"])
                stack.append(e["from_id"])
    return sorted(out)


# --------------------------------------------------------------------------------------------------- stale artifacts

def classify_artifacts(artifacts: Iterable[dict], current_version_id: str) -> List[dict]:
    """Tag each derived artifact ``stale`` when it was built from a version other than the current one."""
    out = []
    for a in artifacts:
        built = a.get("built_from_version_id")
        out.append({**a, "current_version_id": current_version_id, "stale": bool(built) and built != current_version_id,
                    "stale_unknown": not built})
    return out


# ----------------------------------------------------------------------------------------------- atomic transaction

def world_content_hash(world: Any) -> str:
    """Canonical content hash of a world (volatile ``modified_at`` excluded)."""
    d = world.to_dict()
    d.pop("modified_at", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


def commit_correction_atomically(
    store: Any,
    *,
    expected_base_version_id: Optional[str],
    mutate: Callable[[Any], Optional[dict]],
    operation: str,
    actor_id: str,
    actor_kind: str = "user",
    note: Optional[str] = None,
    artifacts: Iterable[dict] = (),
    source_session_ids: Optional[List[str]] = None,
    new_version_id: Optional[str] = None,
):
    """Apply ``mutate`` to a COPY of the head, validate, stamp the audit record and save as a new immutable version.

    Raises BaseVersionRequired / StaleBaseVersion / CorrectionRejected (validation or evidence loss). Any exception
    from ``mutate`` or the store propagates after the store has been left exactly as found: the head version, its
    artifact hash and its content are unchanged and a half-written version is withdrawn (quarantined)."""
    from world_ir.validation import validate_world_ir
    from world_ir.world_v1 import WorldIR

    versions = store.list_versions()
    head = versions[-1] if versions else None
    if head is None:
        raise CorrectionRejected("the store has no version to correct")
    base_id = require_base_version(expected_base_version_id, head.version_id)
    base = store.load_version(base_id)
    base_hash = head.artifact_hash
    # the copy: a JSON round trip, because to_dict()/from_dict() can share nested dicts with the source
    work = WorldIR.from_dict(json.loads(json.dumps(base.to_dict())))
    vid = new_version_id or f"v-{uuid.uuid4().hex[:12]}"
    saved = False
    try:
        extra = mutate(work) or {}
        errors = [str(i) for i in validate_world_ir(work).errors]
        if errors:
            raise CorrectionRejected("corrected world failed validation: " + "; ".join(errors)[:1200])
        work.modified_at = time.time()
        rec = stamp_correction(
            base, work, operation=operation, actor_id=actor_id, actor_kind=actor_kind, base_version_id=base_id,
            new_version_id=vid, entity_ids=extra.get("entity_ids") or (), note=note, lineage=extra.get("lineage") or (),
            artifacts=artifacts)
        if rec.missing_fields():
            raise CorrectionRejected(f"incomplete correction record: {rec.missing_fields()}")
        stored = store.save_version(work, parent=base_id, version_id=vid, source_session_ids=source_session_ids)
        saved = True
        if store.verify_version(vid) or world_content_hash(store.load_version(vid)) != world_content_hash(work):
            raise CorrectionRejected("the saved version failed read-back verification")
    except BaseException:
        if saved or (store._versions_dir / f"{vid}.json").exists():
            try:
                store.quarantine_version(vid)
            except Exception:  # noqa: BLE001 -- the failure being reported matters more
                pass
        raise
    return stored, rec
