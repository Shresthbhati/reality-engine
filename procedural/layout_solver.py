"""Deterministic apartment layout solver (World Factory, task C3).

DATA in, a list of candidate floor plans out. No ML, no wall-clock, no
unseeded randomness: the only random source is `random.Random(spec.seed)`
and the search budget is a fixed trial COUNT, so the same spec + seed
always yields the byte-identical result.

Method (guillotine / BSP partition with constraint scoring)
-----------------------------------------------------------
The footprint rectangle is recursively cut into two rectangles until
every leaf is one room. A candidate is (tree shape, cut orientations,
room-to-leaf assignment). For a candidate the cut POSITIONS are derived,
not searched: each cut splits its area proportionally to the rooms'
target areas, clamped so BOTH sides stay feasible for the rooms they
contain (sum of min_area <= area <= sum of max_area), then snapped to the
grid. Candidates come from seeded random sampling followed by a short
seeded hill-climb (swap two rooms / flip a cut / reshape the tree).

Hard constraints (a plan violating any is never returned):
  * every room area within [min_area, max_area]; aspect <= aspect_max;
    every side >= 1.2 m
  * every REQUIRED adjacency shares >= 0.9 m of wall (one door width)
  * the entrance room has >= 0.9 m of one footprint edge (an exterior door
    fits); a balcony is on the exterior too
Soft score: preferred adjacencies, area fit, aspect fit, daylight (living,
bedroom, kitchen, office on the exterior).

If no candidate meets every hard constraint the result is ok=False with
the unmet REQUIRED pairs named -- never a distorted plan.

Default aspect_max (used only when the caller supplies none; an explicit
value always wins): hallway = 4.0 (a corridor is legitimately long), every
other kind = 2.0.

Grid: all dimensions are integer multiples of 0.1 m (GRID); internally
the solver works in integer decimetres so adjacency tests are exact.
Footprint derivation (when omitted): area = min(1.15 * sum(min_area),
mid of [sum(min), sum(max)]), aspect w:d = 1.5.

Known limits: guillotine layouts only (no L-shaped rooms, no
non-rectangular footprints); one storey; room-level door graph only
(no door placement here); the search is a bounded heuristic, so "ok=False"
means "no plan found within the search budget", which is conclusive only
for the area-based violations that are checked analytically first.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

GRID = 0.1                 # metres; the grid every dimension is snapped to
DOOR_WIDTH_DM = 9          # 0.9 m
MIN_SIDE_DM = 12           # 1.2 m
MAX_ROOMS = 20
MAX_SIDE_M = 100.0
KINDS = ("living", "kitchen", "bedroom", "bathroom", "hallway",
         "balcony", "storage", "office")
LEVELS = ("required", "preferred")
_DAYLIGHT = {"living", "bedroom", "kitchen", "office"}
RANDOM_TRIALS = 20000
CLIMB_STARTS = 6
CLIMB_STEPS = 90
DEFAULT_ASPECT = {"hallway": 4.0}   # default aspect_max by kind when not supplied
DEFAULT_ASPECT_OTHER = 2.0
DERIVED_ASPECT = 1.5
DERIVED_SLACK = 1.15

Rect = Tuple[float, float, float, float]   # x, y, w, d in metres
_IRect = Tuple[int, int, int, int]         # same, integer decimetres


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass
class RoomReq:
    id: str
    kind: str
    min_area: float
    max_area: float
    aspect_max: Optional[float] = None   # None -> DEFAULT_ASPECT by kind

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip() or len(self.id) > 100:
            raise ValueError("room id must be a non-empty string of at most 100 chars")
        if self.kind not in KINDS:
            raise ValueError(f"room {self.id!r}: unknown kind {self.kind!r} (one of {KINDS})")
        if self.aspect_max is None:
            self.aspect_max = DEFAULT_ASPECT.get(self.kind, DEFAULT_ASPECT_OTHER)
        for label in ("min_area", "max_area", "aspect_max"):
            if not _finite(getattr(self, label)):
                raise ValueError(f"room {self.id!r}: {label} must be a finite number")
        if self.min_area <= 0:
            raise ValueError(f"room {self.id!r}: min_area must be > 0")
        if self.max_area < self.min_area:
            raise ValueError(f"room {self.id!r}: max_area must be >= min_area")
        if self.aspect_max < 1.0:
            raise ValueError(f"room {self.id!r}: aspect_max must be >= 1")


@dataclass
class ApartmentSpec:
    name: str
    rooms: list
    entrance: str
    footprint: Optional[Tuple[float, float]] = None
    adjacency: list = field(default_factory=list)
    seed: int = 0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 200:
            raise ValueError("apartment name must be 1-200 characters")
        if not isinstance(self.rooms, list) or not self.rooms:
            raise ValueError("an apartment needs at least one room")
        if len(self.rooms) > MAX_ROOMS:
            raise ValueError(f"an apartment is limited to {MAX_ROOMS} rooms")
        self.rooms = [r if isinstance(r, RoomReq) else RoomReq(**r) for r in self.rooms]
        ids = [r.id for r in self.rooms]
        if len(set(ids)) != len(ids):
            raise ValueError("room ids must be unique")
        if self.entrance not in ids:
            raise ValueError(f"entrance {self.entrance!r} is not a room id")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.footprint is not None:
            if len(self.footprint) != 2 or not all(_finite(v) for v in self.footprint):
                raise ValueError("footprint must be two finite numbers (w, d)")
            if not all(0 < v <= MAX_SIDE_M for v in self.footprint):
                raise ValueError(f"footprint sides must be in (0, {MAX_SIDE_M}] metres")
            self.footprint = (float(self.footprint[0]), float(self.footprint[1]))
        seen, adj = set(), []
        for item in self.adjacency:
            if len(item) != 3:
                raise ValueError("adjacency entries are (room_a, room_b, level)")
            a, b, level = item
            if a not in ids or b not in ids:
                raise ValueError(f"adjacency ({a!r}, {b!r}) names an unknown room id")
            if a == b:
                raise ValueError(f"adjacency ({a!r}, {b!r}) joins a room to itself")
            if level not in LEVELS:
                raise ValueError(f"adjacency level must be one of {LEVELS}, got {level!r}")
            key = frozenset((a, b))
            if key in seen:
                raise ValueError(f"duplicate adjacency between {a!r} and {b!r}")
            seen.add(key)
            adj.append((a, b, level))
        self.adjacency = adj

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "footprint": list(self.footprint) if self.footprint else None,
            "rooms": [dict(id=r.id, kind=r.kind, min_area=r.min_area,
                           max_area=r.max_area, aspect_max=r.aspect_max)
                      for r in self.rooms],
            "adjacency": [list(a) for a in self.adjacency],
            "entrance": self.entrance,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ApartmentSpec":
        return cls(name=d["name"], rooms=list(d["rooms"]), entrance=d["entrance"],
                   footprint=tuple(d["footprint"]) if d.get("footprint") else None,
                   adjacency=[tuple(a) for a in d.get("adjacency", [])],
                   seed=d.get("seed", 0))


@dataclass
class Plan:
    rects: Dict[str, Rect]
    score: float
    satisfied: list
    unsatisfied: list

    @property
    def unsatisfied_required(self) -> list:
        return [u for u in self.unsatisfied if u[2] == "required"]

    def to_dict(self) -> dict:
        return {"rects": {k: list(v) for k, v in self.rects.items()},
                "score": self.score,
                "satisfied": [list(s) for s in self.satisfied],
                "unsatisfied": [list(s) for s in self.unsatisfied]}


@dataclass
class LayoutResult:
    ok: bool
    plans: List[Plan]
    violations: List[str]
    seed: int
    footprint: Optional[Tuple[float, float]] = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "plans": [p.to_dict() for p in self.plans],
                "violations": list(self.violations), "seed": self.seed,
                "footprint": list(self.footprint) if self.footprint else None}


# ------------------------------------------------------------------ helpers

def _dm(v: float) -> int:
    return int(round(v * 10))


def _irect(r: Rect) -> _IRect:
    return (_dm(r[0]), _dm(r[1]), _dm(r[2]), _dm(r[3]))


def _adj_dm(a: _IRect, b: _IRect) -> int:
    """Length (dm) of wall two rectangles share; 0 when they only touch at a corner."""
    ax, ay, aw, ad = a
    bx, by, bw, bd = b
    if ax + aw == bx or bx + bw == ax:
        return max(0, min(ay + ad, by + bd) - max(ay, by))
    if ay + ad == by or by + bd == ay:
        return max(0, min(ax + aw, bx + bw) - max(ax, bx))
    return 0


def adjacent_length(a: Rect, b: Rect) -> float:
    """Metres of wall shared by two rects (0 if not edge-adjacent)."""
    return _adj_dm(_irect(a), _irect(b)) / 10.0


def _exterior_dm(r: _IRect, W: int, D: int) -> int:
    """Longest run of this room along ONE footprint edge (dm)."""
    x, y, w, d = r
    runs = [0]
    if x == 0:
        runs.append(d)
    if x + w == W:
        runs.append(d)
    if y == 0:
        runs.append(w)
    if y + d == D:
        runs.append(w)
    return max(runs)


def exterior_length(rect: Rect, footprint: Tuple[float, float]) -> float:
    return _exterior_dm(_irect(rect), _dm(footprint[0]), _dm(footprint[1])) / 10.0


def _fmt(v: float) -> str:
    return f"{v:g}"


def resolve_footprint(spec: ApartmentSpec) -> Tuple[Optional[Tuple[float, float]], List[str]]:
    """The footprint (snapped to the grid) or a derived one; (None, why) when impossible."""
    if spec.footprint is not None:
        return (_dm(spec.footprint[0]) / 10.0, _dm(spec.footprint[1]) / 10.0), []
    smin = sum(r.min_area for r in spec.rooms)
    smax = sum(r.max_area for r in spec.rooms)
    area = min(smin * DERIVED_SLACK, (smin + smax) / 2.0)
    W = max(_dm(math.sqrt(area * DERIVED_ASPECT)), 1)
    D = max(math.ceil(area * 100 / W - 1e-9), 1)
    if not (smin <= W * D / 100.0 <= smax + 1e-9):
        return None, [f"cannot derive a footprint: rooms need {smin:.1f}-{smax:.1f} m2 "
                      "but no grid rectangle at aspect 1.5 fits that range; pass a footprint"]
    return (W / 10.0, D / 10.0), []


# --------------------------------------------------------------- core search

class _Ctx:
    """Everything the inner loop needs, precomputed in integers."""

    def __init__(self, spec: ApartmentSpec, W: int, D: int):
        self.spec, self.W, self.D = spec, W, D
        rooms = spec.rooms
        self.n = len(rooms)
        self.ids = [r.id for r in rooms]
        idx = {r.id: i for i, r in enumerate(rooms)}
        self.mins = [math.ceil(r.min_area * 100 - 1e-9) for r in rooms]   # dm^2
        self.maxs = [math.floor(r.max_area * 100 + 1e-9) for r in rooms]
        self.rng_ = [m - n for m, n in zip(self.maxs, self.mins)]
        self.aspect = [r.aspect_max for r in rooms]
        self.adj = [(idx[a], idx[b], lvl) for a, b, lvl in spec.adjacency]
        self.entrance = idx[spec.entrance]
        self.must_exterior = {self.entrance} | {i for i, r in enumerate(rooms) if r.kind == "balcony"}
        self.daylight = [i for i, r in enumerate(rooms) if r.kind in _DAYLIGHT]
        self.ideal = [min(r.max_area, max(r.min_area, r.min_area * 1.25)) * 100 for r in rooms]


def _random_tree(slots: list, rng: random.Random):
    if len(slots) == 1:
        return slots[0]
    k = rng.randint(1, len(slots) - 1)
    return (rng.random() < 0.2, _random_tree(slots[:k], rng), _random_tree(slots[k:], rng))


def _leaves(node) -> list:
    return [node] if isinstance(node, int) else _leaves(node[1]) + _leaves(node[2])


def _layout(ctx: _Ctx, tree, perm: list) -> Optional[List[_IRect]]:
    out: List[Optional[_IRect]] = [None] * ctx.n
    mins, maxs, rng_ = ctx.mins, ctx.maxs, ctx.rng_

    def go(node, x, y, w, d) -> bool:
        if isinstance(node, int):
            r = perm[node]
            if not (mins[r] <= w * d <= maxs[r]):
                return False
            lo, hi = (w, d) if w <= d else (d, w)
            if lo < MIN_SIDE_DM or hi > ctx.aspect[r] * lo + 1e-9:
                return False
            out[r] = (x, y, w, d)
            return True
        flip, left, right = node
        L = [perm[s] for s in _leaves(left)]
        R = [perm[s] for s in _leaves(right)]
        A = w * d
        mn1, mn2 = sum(mins[i] for i in L), sum(mins[i] for i in R)
        mx1, mx2 = sum(maxs[i] for i in L), sum(maxs[i] for i in R)
        lo = max(mn1, A - mx2)
        hi = min(mx1, A - mn2)
        if lo > hi:
            return False
        r1, r2 = sum(rng_[i] for i in L), sum(rng_[i] for i in R)
        share = 0.5 if r1 + r2 == 0 else r1 / (r1 + r2)
        target = mn1 + (A - mn1 - mn2) * share
        # flip: bool (False = cut across the longer side) or a forced 'v'/'h'
        # (spine candidates); True: vertical cut splitting the width
        along_w = (flip == "v") if isinstance(flip, str) else ((w >= d) != flip)
        length, other = (w, d) if along_w else (d, w)
        lmin = max(math.ceil(lo / other), MIN_SIDE_DM)
        lmax = min(math.floor(hi / other), length - MIN_SIDE_DM)
        if lmin > lmax:
            return False
        l1 = min(max(int(round(target / other)), lmin), lmax)
        if along_w:
            return go(left, x, y, l1, d) and go(right, x + l1, y, w - l1, d)
        return go(left, x, y, w, l1) and go(right, x, y + l1, w, d - l1)

    return out if go(tree, 0, 0, ctx.W, ctx.D) else None   # type: ignore[return-value]


def _evaluate(ctx: _Ctx, rects: List[_IRect]):
    """(hard_ok, required_met, objective). hard_ok means exterior rules hold."""
    for i in ctx.must_exterior:
        if _exterior_dm(rects[i], ctx.W, ctx.D) < DOOR_WIDTH_DM:
            return False, -1, -1.0
    req_met = pref_met = 0
    for a, b, lvl in ctx.adj:
        ok = _adj_dm(rects[a], rects[b]) >= DOOR_WIDTH_DM
        if lvl == "required":
            req_met += ok
        else:
            pref_met += ok
    fit = 0.0
    asp = 0.0
    for i, (_x, _y, w, d) in enumerate(rects):
        fit += max(0.0, 1 - abs(w * d - ctx.ideal[i]) / ctx.ideal[i])
        lo, hi = (w, d) if w <= d else (d, w)
        cap = max(ctx.aspect[i] - 1, 1e-9)
        asp += max(0.0, 1 - (hi / lo - 1) / cap)
    day = (sum(1 for i in ctx.daylight if _exterior_dm(rects[i], ctx.W, ctx.D) >= DOOR_WIDTH_DM)
           / len(ctx.daylight)) if ctx.daylight else 1.0
    obj = (10 * req_met + 3 * pref_met + 2 * fit / ctx.n + 1 * asp / ctx.n + 0.5 * day)
    return True, req_met, obj


def _signature(rects: List[_IRect]) -> tuple:
    return tuple((round(x / 10), round(y / 10), round(w / 10), round(d / 10))
                 for x, y, w, d in rects)


def _search(ctx: _Ctx, seed: int):
    """Returns (full_plans, best_partial). full_plans: {signature: (obj, rects)}
    meeting every hard rule; best_partial: (req_met, obj, rects) otherwise."""
    rng = random.Random(seed)
    n = ctx.n
    n_req = sum(1 for _a, _b, lvl in ctx.adj if lvl == "required")
    full: Dict[tuple, tuple] = {}
    best_partial: List = [None]
    states: List = []        # (key, tree, perm) for the hill-climb

    def consider(tree, perm):
        rects = _layout(ctx, tree, perm)
        if rects is None:
            return None
        hard, req_met, obj = _evaluate(ctx, rects)
        if not hard:
            return None
        key = (req_met, obj)
        if req_met == n_req:
            sig = _signature(rects)
            if sig not in full or full[sig][0] < obj:
                full[sig] = (obj, rects)
        elif best_partial[0] is None or key > best_partial[0][:2]:
            best_partial[0] = (req_met, obj, rects)
        return key

    ident = list(range(n))
    for tree in _spine_trees(ctx, rng):
        key = consider(tree, ident)
        if key is not None:
            states.append((key, -1, tree, list(ident)))

    for t in range(RANDOM_TRIALS):
        tree = _random_tree(list(range(n)), rng)
        perm = list(range(n))
        if t:
            rng.shuffle(perm)
        key = consider(tree, perm)
        if key is not None:
            states.append((key, t, tree, perm))
    states.sort(key=lambda s: (-s[0][0], -s[0][1], s[1]))

    for _key, _t, tree, perm in states[:CLIMB_STARTS]:
        cur_key = consider(tree, perm)
        for _ in range(CLIMB_STEPS):
            m = rng.randrange(3)
            tree2, perm2 = tree, list(perm)
            if m == 0 and n > 1:
                i, j = rng.sample(range(n), 2)
                perm2[i], perm2[j] = perm2[j], perm2[i]
            elif m == 1:
                tree2 = _toggle(tree, rng.randrange(max(n - 1, 1)))
            else:
                tree2 = _random_tree(list(range(n)), rng)
            key2 = consider(tree2, perm2)
            if key2 is not None and (cur_key is None or key2 >= cur_key):
                tree, perm, cur_key = tree2, perm2, key2
    return full, best_partial[0]


def _stack(rooms: list, cut: str):
    """Rooms laid end to end with forced `cut` ('h' stacks, 'v' side by side)."""
    return rooms[0] if len(rooms) == 1 else (cut, rooms[0], _stack(rooms[1:], cut))


def _spine_trees(ctx: _Ctx, rng: random.Random) -> list:
    """Structured seeds for hub-style briefs: a hallway-kind room as a
    full-length strip (either direction; centred or on one side) with the
    other rooms stacked in the one or two columns beside it. Rooms that must
    touch each other stay consecutive in a column; groups go to a side by
    area balance. Only seeds: every one still passes the normal hard checks."""
    n = ctx.n
    hall = next((i for i, r in enumerate(ctx.spec.rooms) if r.kind == "hallway"), None)
    if hall is None or n < 3:
        return []
    others = [i for i in range(n) if i != hall]
    link = {i: set() for i in others}
    for a, b, lvl in ctx.adj:
        if lvl == "required" and hall not in (a, b):
            link[a].add(b)
            link[b].add(a)
    groups, seen = [], set()
    for i in others:
        if i in seen:
            continue
        comp, stack = [], [i]
        while stack:
            c = stack.pop()
            if c not in seen:
                seen.add(c)
                comp.append(c)
                stack.extend(sorted(link[c], reverse=True))
        # path order: start at an endpoint, walk along required links
        start = min(comp, key=lambda c: (len(link[c]), c))
        order, vis = [], set()
        stack = [start]
        while stack:
            c = stack.pop()
            if c not in vis:
                vis.add(c)
                order.append(c)
                stack.extend(sorted(link[c] - vis, reverse=True))
        groups.append(order)
    size = [sum((ctx.mins[i] + ctx.maxs[i]) / 2 for i in g) for g in groups]
    g = len(groups)
    masks = (range(1 << g) if g <= 10 else [(1 << g) - 1])
    ranked = sorted(masks, key=lambda m: (abs(sum(s if m >> k & 1 else -s
                                                  for k, s in enumerate(size))), m))[:4]
    trees = []
    for strip, cross in (("v", "h"), ("h", "v")):
        def column(gs: list, cross=cross):
            return _stack([r for grp in gs for r in grp], cross)

        for mask in ranked:
            left = [groups[k] for k in range(g) if mask >> k & 1]
            right = [groups[k] for k in range(g) if not mask >> k & 1]
            for _variant in range(3):
                if _variant:
                    rng.shuffle(left)
                    rng.shuffle(right)
                if left and right:
                    trees.append((strip, column(left), (strip, hall, column(right))))
                elif left or right:
                    col = column(left or right)
                    trees.append((strip, hall, col))
                    trees.append((strip, col, hall))
        # "head" hall: a compact hall at one end of the middle column, with
        # side columns on both flanks (for halls whose aspect_max forbids a
        # full-length strip). Groups are dealt into left / middle / right.
        if 1 <= g <= 7:
            weight = lambda gs: sum(size[k] for k in gs)
            deals = []
            for code in range(3 ** g):
                buckets, c = ([], [], []), code
                for k in range(g):
                    buckets[c % 3].append(k)
                    c //= 3
                if not buckets[1]:
                    continue
                w = [weight(b) for b in buckets]
                deals.append((max(w) - min(x for x, b in zip(w, buckets) if b), code, buckets))
            deals.sort(key=lambda d: (d[0], d[1]))
            for _imb, _code, (a, m, b) in deals[:6]:
                mid = [r for k in m for r in groups[k]]
                for hall_first in (True, False):
                    middle = (_stack([hall] + mid, cross) if hall_first
                              else _stack(mid + [hall], cross))
                    parts = [column([groups[k] for k in a])] if a else []
                    parts.append(middle)
                    if b:
                        parts.append(column([groups[k] for k in b]))
                    tree = parts[-1]
                    for p in reversed(parts[:-1]):
                        tree = (strip, p, tree)
                    trees.append(tree)
    return trees


def _toggle(node, which: int):
    """Flip the cut orientation of the `which`-th internal node (pre-order)."""
    counter = [which]

    def rec(nd):
        if isinstance(nd, int):
            return nd
        flip, l, r = nd
        if counter[0] == 0:
            counter[0] -= 1
            return (("h" if flip == "v" else "v") if isinstance(flip, str) else not flip, l, r)
        counter[0] -= 1
        l2 = rec(l)
        return (flip, l2, rec(r))

    return rec(node)


# ------------------------------------------------------------------- public

def _plan_from(ctx: _Ctx, obj: float, rects: List[_IRect]) -> Plan:
    sat, unsat = [], []
    for a, b, lvl in ctx.adj:
        entry = (ctx.ids[a], ctx.ids[b], lvl)
        (sat if _adj_dm(rects[a], rects[b]) >= DOOR_WIDTH_DM else unsat).append(entry)
    return Plan(
        rects={ctx.ids[i]: tuple(v / 10.0 for v in r) for i, r in enumerate(rects)},
        score=round(obj, 4), satisfied=sat, unsatisfied=unsat)


def _analytic_violations(spec: ApartmentSpec, W: int, D: int) -> List[str]:
    out = []
    A = W * D
    fp = f"{_fmt(W / 10)}x{_fmt(D / 10)}"
    for r in spec.rooms:
        if r.min_area * 100 > A + 1e-9:
            out.append(f"{r.id} min_area {_fmt(r.min_area)} m2 cannot fit footprint {fp}")
        elif min(W, D) < MIN_SIDE_DM:
            out.append(f"footprint {fp} is narrower than the {MIN_SIDE_DM / 10:g} m minimum room side")
    smin = sum(r.min_area for r in spec.rooms)
    smax = sum(r.max_area for r in spec.rooms)
    if smin * 100 > A + 1e-9:
        out.append(f"rooms need at least {smin:.1f} m2 but footprint {fp} offers {A / 100:.1f} m2")
    if smax * 100 < A - 1e-9:
        out.append(f"rooms can fill at most {smax:.1f} m2 of footprint {fp} "
                   f"({A / 100:.1f} m2); raise max_area or shrink the footprint")
    return out


def _aspect_hint(spec: ApartmentSpec, pair: tuple) -> str:
    """Hub rooms (hallway) with a tight aspect_max cannot become the long strip
    that touches many neighbours; say so instead of leaving the user guessing."""
    tight = [r.id for r in spec.rooms
             if r.id in pair and r.kind == "hallway" and r.aspect_max < 3.0]
    return (f"; hallway {tight[0]!r} has aspect_max {spec.rooms[[r.id for r in spec.rooms].index(tight[0])].aspect_max:g}"
            " which forbids a long corridor shape -- raise it (e.g. 4-6) if it must touch many rooms"
            if tight else "")


def solve_layout(spec: ApartmentSpec, max_alternatives: int = 3) -> LayoutResult:
    if not isinstance(max_alternatives, int) or isinstance(max_alternatives, bool) \
            or not 1 <= max_alternatives <= 10:
        raise ValueError("max_alternatives must be an integer in 1..10")
    fp, why = resolve_footprint(spec)
    if fp is None:
        return LayoutResult(False, [], why, spec.seed, None)
    W, D = _dm(fp[0]), _dm(fp[1])
    pre = _analytic_violations(spec, W, D)
    if pre:
        return LayoutResult(False, [], pre, spec.seed, fp)

    ctx = _Ctx(spec, W, D)
    full, partial = _search(ctx, spec.seed)
    if full:
        ranked = sorted(full.items(), key=lambda kv: (-kv[1][0], kv[0]))
        plans = [_plan_from(ctx, obj, rects) for _sig, (obj, rects) in ranked[:max_alternatives]]
        return LayoutResult(True, plans, [], spec.seed, fp)

    fpn = f"{_fmt(fp[0])}x{_fmt(fp[1])}"
    if partial is None:
        msg = (f"no layout inside footprint {fpn} satisfies every room's area, aspect "
               "and exterior-access rule (entrance/balcony on the outside); relax "
               "min/max areas or aspect_max, or change the footprint")
        return LayoutResult(False, [], [msg], spec.seed, fp)
    plan = _plan_from(ctx, partial[1], partial[2])
    n_req = sum(1 for _a, _b, lvl in spec.adjacency if lvl == "required")
    violations = [
        f"required adjacency '{a}'-'{b}' could not be satisfied (>= 0.9 m shared wall) "
        f"in any layout found (best attempt met {partial[0]} of {n_req} required)"
        + _aspect_hint(spec, (a, b))
        for a, b, _lvl in plan.unsatisfied_required]
    return LayoutResult(False, [], violations, spec.seed, fp)


def validate_plan(spec: ApartmentSpec, rects: dict) -> List[str]:
    """Server-side re-check of a plan against the spec. [] means valid."""
    fp, why = resolve_footprint(spec)
    if fp is None:
        return why
    W, D = _dm(fp[0]), _dm(fp[1])
    ids = [r.id for r in spec.rooms]
    problems: List[str] = []
    for rid in ids:
        if rid not in rects:
            problems.append(f"plan is missing room {rid!r}")
    for rid in rects:
        if rid not in ids:
            problems.append(f"plan has unknown room {rid!r}")
    if problems:
        return problems
    irects: Dict[str, _IRect] = {}
    for rid in ids:
        raw = rects[rid]
        if len(raw) != 4 or not all(_finite(v) for v in raw):
            return [f"room {rid!r}: rect must be four finite numbers (x, y, w, d)"]
        if any(abs(v * 10 - round(v * 10)) > 1e-6 for v in raw):
            problems.append(f"room {rid!r}: rect is not on the {GRID:g} m grid")
        irects[rid] = _irect(tuple(raw))
    if problems:
        return problems
    for rid, (x, y, w, d) in irects.items():
        if w <= 0 or d <= 0 or x < 0 or y < 0 or x + w > W or y + d > D:
            problems.append(f"room {rid!r}: rect lies outside the footprint")
    order = list(irects)
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            ax, ay, aw, ad = irects[a]
            bx, by, bw, bd = irects[b]
            if min(ax + aw, bx + bw) > max(ax, bx) and min(ay + ad, by + bd) > max(ay, by):
                problems.append(f"rooms {a!r} and {b!r} overlap")
    if sum(w * d for _x, _y, w, d in irects.values()) != W * D:
        problems.append("rooms do not exactly cover the footprint")
    if problems:
        return problems
    by_id = {r.id: r for r in spec.rooms}
    for rid, (x, y, w, d) in irects.items():
        room = by_id[rid]
        area = w * d / 100.0
        if not (room.min_area - 1e-9 <= area <= room.max_area + 1e-9):
            problems.append(f"room {rid!r}: area {area:.1f} m2 outside "
                            f"[{_fmt(room.min_area)}, {_fmt(room.max_area)}]")
        lo, hi = (w, d) if w <= d else (d, w)
        if lo < MIN_SIDE_DM or hi > room.aspect_max * lo + 1e-9:
            problems.append(f"room {rid!r}: {w / 10:g}x{d / 10:g} m violates aspect/min-side rules")
    for a, b, lvl in spec.adjacency:
        if lvl == "required" and _adj_dm(irects[a], irects[b]) < DOOR_WIDTH_DM:
            problems.append(f"required adjacency '{a}'-'{b}' is not satisfied")
    for rid in [spec.entrance] + [r.id for r in spec.rooms if r.kind == "balcony"]:
        if _exterior_dm(irects[rid], W, D) < DOOR_WIDTH_DM:
            problems.append(f"room {rid!r} needs >= 0.9 m of exterior wall")
    return problems
