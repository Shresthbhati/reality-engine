"""Reconstruction completeness as SEPARATE measurements -- there is deliberately no single score.

Each axis is a ratio over what the evidence can actually support, or None (with a reason) when it cannot be
measured. A missing entity is never counted as reconstructed: ratios are over entities that exist, and the
`unknowns` list states in words what remains unknown. `entity_detection` is measurable only against a reference
inventory (ground truth in a benchmark, a floor plan in production); without one it is None, not 1.0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .association import AssociationResult, line_through
from .model import Kind, SpatialEntity, Status
from .topology import Severity, TopologyReport


def _q(values: Sequence[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if len(values) else None


@dataclass
class CompletenessReport:
    geometric_coverage: dict[str, object]
    surface_completeness: dict[str, object]
    entity_detection: dict[str, object]
    topological: dict[str, object]
    semantic: dict[str, object]
    evidence: dict[str, object]
    uncertainty: dict[str, object]
    unresolved_contradictions: int
    unknowns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"geometric_coverage": self.geometric_coverage, "surface_completeness": self.surface_completeness,
                "entity_detection": self.entity_detection, "topological": self.topological, "semantic": self.semantic,
                "evidence": self.evidence, "uncertainty": self.uncertainty,
                "unresolved_contradictions": self.unresolved_contradictions, "unknowns": list(self.unknowns)}


def _coverage_from_points(e: SpatialEntity, pts: np.ndarray, cell: float, band: float) -> float | None:
    fit = line_through(e.a, e.b)
    t = (pts[:, 0] - fit.c[0]) * fit.u[0] + (pts[:, 1] - fit.c[1]) * fit.u[1]
    d = -(pts[:, 0] - fit.c[0]) * fit.u[1] + (pts[:, 1] - fit.c[1]) * fit.u[0]
    m = (np.abs(d) <= band) & (t >= fit.t_lo) & (t <= fit.t_hi)
    nt = max(1, int(math.ceil((fit.t_hi - fit.t_lo) / cell)))
    if e.z0 is not None and e.z1 is not None and e.z1 > e.z0:
        nz = max(1, int(math.ceil((e.z1 - e.z0) / cell)))
        mm = m & (pts[:, 2] >= e.z0) & (pts[:, 2] <= e.z1)
        ti = np.clip(((t[mm] - fit.t_lo) / cell).astype(int), 0, nt - 1)
        zi = np.clip(((pts[mm, 2] - e.z0) / cell).astype(int), 0, nz - 1)
        return len({(a, b) for a, b in zip(ti, zi)}) / (nt * nz)
    ti = np.clip(((t[m] - fit.t_lo) / cell).astype(int), 0, nt - 1)
    return len(set(ti.tolist())) / nt


def assess_completeness(
    association: AssociationResult, entities: Sequence[SpatialEntity], topology: TopologyReport, *,
    n_observations: int | None = None, points_xyz: np.ndarray | None = None,
    reference_matches: tuple[int, int] | None = None, cell_m: float = 0.25, band_m: float = 0.15,
) -> CompletenessReport:
    unknowns: list[str] = []
    ents = list(entities)
    walls = [e for e in ents if e.kind is Kind.WALL]
    openings = [e for e in ents if e.kind in (Kind.DOOR, Kind.WINDOW, Kind.OPENING)]

    # 1. geometric coverage -- needs reconstructed points; otherwise not measurable
    if points_xyz is not None and len(points_xyz) and walls:
        cov = {w.entity_id: _coverage_from_points(w, np.asarray(points_xyz, float), cell_m, band_m) for w in walls}
        vals = [v for v in cov.values() if v is not None]
        geo = {"measured": True, "per_wall": cov, "min": min(vals), "median": _q(vals, 0.5),
               "basis": f"fraction of wall-surface cells (cell {cell_m:.2f} m) holding a reconstructed point"}
        for wid, v in sorted(cov.items(), key=lambda kv: (kv[1] is None, kv[1]))[:3]:
            if v is not None and v < 0.5:
                unknowns.append(f"{wid}: only {v:.0%} of its surface has reconstructed points")
    else:
        geo = {"measured": False, "reason": "no reconstructed points supplied"}

    # 2. surface completeness: how much of the known wall length is bounded by observed terminations
    tot = sum(w.length for w in walls)
    bounded = sum(w.length for w in walls if w.end_a_observed and w.end_b_observed)
    surf = {"measured": bool(walls), "wall_length_m": tot, "fully_bounded_length_m": bounded,
            "fully_bounded_fraction": (bounded / tot) if tot > 0 else None,
            "walls_with_lower_bound_extent": sorted(w.entity_id for w in walls if not (w.end_a_observed and w.end_b_observed)),
            "basis": "length of walls whose BOTH ends were seen as terminations / total wall length (others are lower bounds)"}
    for w in walls:
        if not (w.end_a_observed and w.end_b_observed):
            unknowns.append(f"{w.entity_id}: true extent unknown (end not observed); {w.length:.2f} m is a lower bound")

    # 3. entity detection coverage -- only against a reference inventory
    if reference_matches is None:
        det = {"measured": False, "reason": "no reference inventory supplied", "detected": len(ents)}
    else:
        n_exp, n_match = reference_matches
        det = {"measured": True, "expected": n_exp, "matched": n_match, "recall": n_match / n_exp if n_exp else None,
               "detected": len(ents)}

    # 4. topological completeness
    ends = [(w, tag) for w in walls for tag in "ab"]
    dangling = {(a, b) for a, b in topology.dangling}
    n_end = len(ends)
    n_unobs = sum(1 for w, tag in ends if not (w.end_a_observed if tag == "a" else w.end_b_observed))
    n_free_obs = sum(1 for (wid, tag) in dangling
                     if next(w for w in walls if w.entity_id == wid) and (
                         next(w for w in walls if w.entity_id == wid).end_a_observed if tag == "a"
                         else next(w for w in walls if w.entity_id == wid).end_b_observed))
    n_join = n_end - len(dangling)
    hosted = sum(1 for o in openings if o.host_id is not None)
    open_gaps = [i for i in topology.issues if i.code == "open_boundary_observed_gap"]
    topo = {"wall_ends": n_end, "ends_with_junction": n_join, "ends_unobserved": n_unobs,
            "ends_observed_free": n_free_obs, "junction_fraction": (n_join / n_end) if n_end else None,
            "rooms_enclosed": len(topology.rooms), "open_boundary_gaps": len(open_gaps),
            "openings": len(openings), "openings_hosted": hosted,
            "opening_host_fraction": (hosted / len(openings)) if openings else None}
    for o in openings:
        if o.host_id is None:
            unknowns.append(f"{o.entity_id}: host wall unknown")
    for i in open_gaps:
        unknowns.append(i.message)

    # 5. semantic completeness
    firm = [e for e in ents if e.beliefs["semantic"].status in (Status.SEMANTICALLY_INFERRED, Status.DIRECTLY_OBSERVED)]
    sem = {"entities": len(ents), "classified_with_support": len(firm),
           "fraction": (len(firm) / len(ents)) if ents else None,
           "conflicting_class": sorted(e.entity_id for e in ents if e.beliefs["semantic"].status is Status.CONFLICTING)}

    # 6. evidence coverage
    multi = [e for e in ents if len(e.image_ids) >= 2]
    used = {x for e in ents for x in e.evidence_ids}
    ev = {"entities_with_multiview": len(multi), "entities": len(ents),
          "multiview_fraction": (len(multi) / len(ents)) if ents else None,
          "min_views": min((len(e.image_ids) for e in ents), default=None),
          "observations_used": len(used),
          "observations_total": n_observations,
          "observation_use_fraction": (len(used) / n_observations) if n_observations else None}

    # 7. uncertainty (raw metres, not folded into a score)
    sig = [e.beliefs["position"].sigma for e in walls if e.beliefs["position"].sigma is not None]
    endsig = [s for e in walls for s in (e.end_sigma("a"), e.end_sigma("b")) if s is not None]
    unc = {"position_sigma_m": {"median": _q(sig, 0.5), "p90": _q(sig, 0.9), "max": max(sig) if sig else None},
           "end_sigma_m": {"median": _q(endsig, 0.5), "p90": _q(endsig, 0.9), "max": max(endsig) if endsig else None},
           "rooms_area_sigma_m2": {r.room_id: r.area_sigma_m2 for r in topology.rooms},
           "unknown_sigma_entities": sorted(e.entity_id for e in ents if e.beliefs["position"].sigma is None)}

    contra = len(topology.by_severity(Severity.CONTRADICTION))
    for e in association.tentative():
        unknowns.append(f"{e.entity_id}: single view, not established")
    return CompletenessReport(geo, surf, det, topo, sem, ev, unc, contra, sorted(set(unknowns)))
