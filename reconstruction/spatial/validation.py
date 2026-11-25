"""Global validation after optimization: one of five reconstruction states, with every reason that applies.

States (highest priority first; `flags` lists ALL that apply, `state` is the most severe):
  invalid          a structural error: the evidence cannot be right as stated (duplicate wall, walls crossing with
                   no junction, self-intersecting room, opening past a fully observed host, stair within one
                   storey, untransformable frame, ...)
  uncertain        no structural error, but the result cannot be trusted at the stated precision: surfaced
                   contradictions, a position sigma above `sigma_limit_m`, entities with no sigma at all, or
                   observations the optimizer rejected as outliers
  incomplete       nothing is wrong but no space is enclosed yet (boundaries were not observed)
  valid_partial    at least one enclosed room, with parts still unobserved
  valid_complete   every observed wall end resolved, at least one room, nothing unresolved
A missing wall is never a contradiction: it can only make the result incomplete / partial.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .model import Kind, SpatialEntity
from .multiview import MultiViewResult
from .topology import Severity, TopologyReport


class ReconState(str, Enum):
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    INCOMPLETE = "incomplete"
    VALID_PARTIAL = "valid_partial"
    VALID_COMPLETE = "valid_complete"


@dataclass
class GlobalValidation:
    state: ReconState
    flags: list[str]
    reasons: list[str]
    components: list[list[str]]
    checks: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"state": self.state.value, "flags": self.flags, "reasons": self.reasons,
                "components": self.components, "checks": self.checks}


def wall_components(entities: Sequence[SpatialEntity], junctions: Sequence[tuple[str, str, str, str]]) -> list[list[str]]:
    """Connected components of the wall graph (edges = evidenced junctions) plus openings joined to their host."""
    ids = [e.entity_id for e in entities if e.kind is not Kind.STAIR]
    ix = {k: i for i, k in enumerate(ids)}
    rows, cols = [], []
    for a, _, b, _ in junctions:
        if a in ix and b in ix:
            rows.append(ix[a])
            cols.append(ix[b])
    for e in entities:
        if e.host_id and e.host_id in ix and e.entity_id in ix:
            rows.append(ix[e.entity_id])
            cols.append(ix[e.host_id])
    n = len(ids)
    if n == 0:
        return []
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    k, lab = connected_components(g, directed=False)
    comps: dict[int, list[str]] = {}
    for i, c in enumerate(lab):
        comps.setdefault(int(c), []).append(ids[i])
    return sorted((sorted(v) for v in comps.values()), key=lambda v: (-len(v), v[0]))


def classify_reconstruction(
    entities: Sequence[SpatialEntity], topology: TopologyReport, junctions: Sequence[tuple[str, str, str, str]] = (),
    *, optimization: MultiViewResult | None = None, sigma_limit_m: float = 0.15,
) -> GlobalValidation:
    flags: list[str] = []
    reasons: list[str] = []
    errs = topology.by_severity(Severity.STRUCTURAL_ERROR)
    if errs:
        flags.append("invalid")
        reasons += [f"{i.code}: {i.message}" for i in errs[:5]]
    unc: list[str] = []
    contra = topology.by_severity(Severity.CONTRADICTION)
    if contra:
        unc.append(f"{len(contra)} unresolved contradiction(s)")
    walls = [e for e in entities if e.kind is Kind.WALL]
    wide = [e.entity_id for e in walls if (e.beliefs["position"].sigma or 0.0) > sigma_limit_m]
    nosig = [e.entity_id for e in walls if e.beliefs["position"].sigma is None]
    if wide:
        unc.append(f"position sigma above {sigma_limit_m} m: {wide[:4]}")
    if nosig:
        unc.append(f"no position sigma: {nosig[:4]}")
    if optimization and optimization.outliers:
        unc.append(f"{len(optimization.outliers)} observation(s) rejected as multi-view outliers")
    if unc:
        flags.append("uncertain")
        reasons += unc
    inc = topology.by_severity(Severity.INCOMPLETE)
    inc = [i for i in inc if i.code != "single_view_entity"] or inc
    if inc:
        flags.append("incomplete" if not topology.rooms else "valid_partial")
        reasons.append(f"{len(inc)} unobserved/unresolved element(s), e.g. {inc[0].code}")
    comps = wall_components(entities, junctions)
    checks = {"structural_errors": len(errs), "contradictions": len(contra), "incomplete_issues": len(inc),
              "rooms": len(topology.rooms), "wall_components": len(comps),
              "multiview_outliers": len(optimization.outliers) if optimization else None,
              "max_position_sigma_m": max((e.beliefs["position"].sigma or 0.0 for e in walls), default=None)}
    if "invalid" in flags:
        state = ReconState.INVALID
    elif "uncertain" in flags:
        state = ReconState.UNCERTAIN
    elif "incomplete" in flags:
        state = ReconState.INCOMPLETE
    elif "valid_partial" in flags:
        state = ReconState.VALID_PARTIAL
    elif topology.rooms:
        state = ReconState.VALID_COMPLETE
        flags.append("valid_complete")
    else:
        state = ReconState.INCOMPLETE
        flags.append("incomplete")
        reasons.append("no enclosed space has been reconstructed")
    return GlobalValidation(state, flags, reasons, comps, checks)
