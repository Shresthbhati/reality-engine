"""Cross-view entity association with a decision ledger.

``perception.instances.global_association.cluster_observations`` already merges candidate pairs strongest-first with union-find
under hard constraints (one object cannot appear twice in one image; a part only joins groups whose wholes share an entity). It
returns entity indices only -- *why* two detections were or were not merged is lost. This module reproduces that exact merge
order (pinned by a randomized equivalence test) and records EVERY pair decision with its reason, plus one extra refusal the
original has no input for: incompatible semantic labels.

Nothing is averaged or silently dropped: a refused pair stays in the ledger with the constraint that refused it, and an accepted
pair can be written to the evidence store as a ``same_entity`` relation so identity is traceable to the raw observations.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from engine.multiengine.evidence_store import EvidenceStore

LabelCompat = Callable[[str, str], bool]

MERGED = "merged"
REFUSED = "refused"


class AssociationError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    observation_id: str
    source_id: str  # the image/video the detection came from
    label: str
    parent_observation_id: str | None = None  # set when this detection is a PART of another detection


@dataclass(frozen=True)
class PairScore:
    a: str  # observation ids
    b: str
    score: float  # > 0 means "plausibly the same object"; what produced it belongs in ``basis``
    basis: Mapping[str, Any] = field(default_factory=dict)  # e.g. {"epipolar_px": 1.2, "reid_cosine": 0.91}


@dataclass(frozen=True)
class Decision:
    a: str
    b: str
    score: float
    outcome: str  # "merged" | "refused"
    reason: str
    basis: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssociatedEntity:
    entity_key: str  # stable: a hash of the sorted member observation ids
    members: tuple[str, ...]
    labels: tuple[str, ...]  # distinct labels among the members, sorted
    sources: tuple[str, ...]  # distinct images/videos, sorted


@dataclass(frozen=True)
class AssociationResult:
    entities: tuple[AssociatedEntity, ...]
    decisions: tuple[Decision, ...]

    def entity_of(self, observation_id: str) -> AssociatedEntity:
        for e in self.entities:
            if observation_id in e.members:
                return e
        raise AssociationError(f"unknown observation {observation_id!r}")

    def decisions_for(self, observation_id: str) -> list[Decision]:
        return [d for d in self.decisions if observation_id in (d.a, d.b)]

    def refused(self, reason: str | None = None) -> list[Decision]:
        return [d for d in self.decisions if d.outcome == REFUSED and (reason is None or d.reason == reason)]


def entity_key(member_ids: Iterable[str]) -> str:
    blob = "|".join(sorted(member_ids))
    return "ent_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def ontology_label_compat(ontology: Any | None = None) -> LabelCompat:
    """Labels are compatible when they resolve to the same ontology class, one is an ancestor of the other, or either is the
    ontology's unknown-object fallback (an unclassified detection cannot contradict a classified one)."""
    from perception.ontology import FALLBACK_PATH, get_default_ontology

    onto = ontology or get_default_ontology()

    def compat(la: str, lb: str) -> bool:
        na, nb = onto.resolve_or_unknown(la), onto.resolve_or_unknown(lb)
        if FALLBACK_PATH in (na.path, nb.path) or na.path == nb.path:
            return True
        return nb.path in onto.ancestors(na.path) or na.path in onto.ancestors(nb.path)

    return compat


def associate(
    candidates: Sequence[Candidate],
    scores: Iterable[PairScore],
    *,
    labels_compatible: LabelCompat | None = None,
    store: EvidenceStore | None = None,
) -> AssociationResult:
    """Group detections of the same physical object across views.

    Pairs are taken strongest-first; ties are broken by candidate order (``min row``, ``max row``), never by input order, so the
    result does not depend on how the caller sorted the pairs. ``labels_compatible`` adds a semantic refusal; leave it ``None`` to
    reproduce ``cluster_observations`` exactly. With ``store``, each accepted pair is recorded as a ``same_entity`` relation.
    """
    index: dict[str, int] = {}
    for i, c in enumerate(candidates):
        if c.observation_id in index:
            raise AssociationError(f"duplicate candidate observation {c.observation_id!r}")
        index[c.observation_id] = i
    n = len(candidates)
    parent = list(range(n))
    size = [1] * n
    imgs = [{c.source_id} for c in candidates]
    for c in candidates:
        if c.parent_observation_id and c.parent_observation_id not in index:
            raise AssociationError(
                f"candidate {c.observation_id!r} names unknown parent {c.parent_observation_id!r}"
            )
    part_of = [index[c.parent_observation_id] if c.parent_observation_id else -1 for c in candidates]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pending: list[tuple[float, int, int, PairScore]] = []
    decisions: list[Decision] = []
    for s in scores:
        for oid in (s.a, s.b):
            if oid not in index:
                raise AssociationError(f"pair score references unknown observation {oid!r}")
        if s.a == s.b:
            raise AssociationError(f"pair score {s.a!r} pairs an observation with itself")
        if not s.score > 0:
            decisions.append(Decision(s.a, s.b, s.score, REFUSED, "non_positive_score", s.basis))
            continue
        lo, hi = sorted((index[s.a], index[s.b]))
        pending.append((-s.score, lo, hi, s))
    pending.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, _lo, _hi, s in pending:
        ia, ib = index[s.a], index[s.b]
        ra, rb = find(ia), find(ib)

        def refuse(reason: str, s: PairScore = s) -> None:
            decisions.append(Decision(s.a, s.b, s.score, REFUSED, reason, s.basis))

        if ra == rb:
            refuse("already_same_entity")
            continue
        if imgs[ra] & imgs[rb]:
            refuse("same_image_conflict")
            continue
        pa, pb = part_of[ia], part_of[ib]
        if (pa >= 0) != (pb >= 0):
            refuse("whole_part_mismatch")
            continue
        if pa >= 0 and find(pa) != find(pb):
            refuse("part_parent_mismatch")
            continue
        if labels_compatible is not None and not labels_compatible(candidates[ia].label, candidates[ib].label):
            refuse("label_conflict")
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        imgs[ra] |= imgs[rb]
        decisions.append(Decision(s.a, s.b, s.score, MERGED, "accepted", s.basis))
        if store is not None:
            store.link(s.a, s.b, "same_entity")

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    entities = []
    for members in groups.values():
        ids = tuple(sorted(candidates[i].observation_id for i in members))
        entities.append(
            AssociatedEntity(
                entity_key=entity_key(ids),
                members=ids,
                labels=tuple(sorted({candidates[i].label for i in members})),
                sources=tuple(sorted({candidates[i].source_id for i in members})),
            )
        )
    entities.sort(key=lambda e: e.members)
    return AssociationResult(tuple(entities), tuple(decisions))
