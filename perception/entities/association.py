"""Multi-view association: one physical entity seen in several photographs becomes ONE entity.

Rules (each one is a failure mode of naive merging):
  * two detections in the SAME image are two objects -- never merged (a row of identical sockets is the case that matters);
  * only the same class may merge (a socket never merges with a switch because they sit close together);
  * a link needs 3D agreement within a gate that scales with both positional uncertainties;
  * when an observation has two plausible partners in another image the link is AMBIGUOUS and is not made -- two
    entities beat one wrongly merged entity;
  * appearance (when given) can only veto a merge, never create one: similar looks alone never merge objects.
Source observations and the association confidence are preserved on every cluster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class EntityObservation:
    obs_id: str
    evidence_id: str
    class_id: str
    position: Tuple[float, float, float]         # 3D estimate in the world frame
    sigma_m: float = 0.02                         # 1-sigma positional uncertainty of this estimate
    confidence: float = 0.5
    descriptor: Optional[Tuple[float, ...]] = None


@dataclass
class Cluster:
    members: List[EntityObservation]
    association_confidence: Optional[float]       # None for a single observation (nothing was associated)
    ambiguous_with: List[str] = field(default_factory=list)

    @property
    def position(self) -> Tuple[float, float, float]:
        w = np.asarray([1.0 / max(m.sigma_m, 1e-6) ** 2 for m in self.members])
        p = np.asarray([m.position for m in self.members], dtype=float)
        return tuple(float(v) for v in (p * w[:, None]).sum(0) / w.sum())

    @property
    def evidence_ids(self) -> List[str]:
        return sorted({m.evidence_id for m in self.members})


def _cos(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def associate(observations: Sequence[EntityObservation], *, gate_sigmas: float = 3.0, min_gate_m: float = 0.01,
              ambiguity_ratio: float = 1.5, appearance_veto: float = 0.3) -> List[Cluster]:
    obs = list(observations)
    parent = list(range(len(obs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges: List[Tuple[float, int, int]] = []
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            a, b = obs[i], obs[j]
            if a.evidence_id == b.evidence_id or a.class_id != b.class_id:
                continue
            if a.descriptor is not None and b.descriptor is not None and _cos(a.descriptor, b.descriptor) < appearance_veto:
                continue
            d = math.dist(a.position, b.position)
            if d <= max(min_gate_m, gate_sigmas * math.hypot(a.sigma_m, b.sigma_m)):
                edges.append((d, i, j))
    edges.sort()

    # ambiguity: an observation with several candidate partners in one other image (within ratio of the best) is not linked
    cand: Dict[Tuple[int, str], List[Tuple[float, int]]] = {}
    for d, i, j in edges:
        cand.setdefault((i, obs[j].evidence_id), []).append((d, j))
        cand.setdefault((j, obs[i].evidence_id), []).append((d, i))
    bad = set()
    ambiguous: Dict[int, List[str]] = {}
    for (x, _), lst in cand.items():
        lst.sort()
        if len(lst) > 1 and lst[1][0] <= lst[0][0] * ambiguity_ratio + 1e-9:
            for _, y in lst:
                bad.add((x, y))
                bad.add((y, x))
            ambiguous.setdefault(x, []).extend(obs[y].obs_id for _, y in lst)

    members: Dict[int, List[int]] = {i: [i] for i in range(len(obs))}
    links: Dict[int, List[float]] = {i: [] for i in range(len(obs))}
    for d, i, j in edges:
        if (i, j) in bad:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if {obs[k].evidence_id for k in members[ri]} & {obs[k].evidence_id for k in members[rj]}:
            continue                                  # one image cannot show the same entity twice
        parent[rj] = ri
        members[ri] += members.pop(rj)
        links[ri] += links.pop(rj) + [d]

    out: List[Cluster] = []
    for r, idx in members.items():
        ms = [obs[k] for k in idx]
        if len(ms) == 1:
            out.append(Cluster(ms, None, sorted(set(ambiguous.get(idx[0], [])))))
            continue
        spread = np.mean(links[r]) / max(min_gate_m, gate_sigmas * float(np.mean([m.sigma_m for m in ms])))
        out.append(Cluster(ms, float(max(0.0, min(1.0, 1.0 - spread * 0.5))), []))
    out.sort(key=lambda c: c.members[0].obs_id)
    return out
