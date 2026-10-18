"""Procedural building grammar (World Factory, task C4).

A BuildingSpec (data) becomes a multi-storey WorldIR: per storey a floor
slab, four exterior walls, interior rooms from ONE stored floor layout
(solved once, the same plan on every storey) or a single open plate, and
facade windows; STAIRS between consecutive storeys; a ROOF slab on top and
a ground-floor entrance door. Same contract as the other grammars:
deterministic (same spec -> byte-identical WorldIR), every statement
PROCEDURAL / GENERATED, authoring in local Z-up axes (x east, y north, z
up) emitted Y-up through room_grammar._entity. Compass sides are the
geographic ones here: "south" is the local y = 0 edge, "north" y = depth.

Vertical model. `storey_heights[k]` is floor-to-floor. Storey k's floor
slab occupies [z_k - 0.2, z_k] (z_k = sum of heights below), its walls and
rooms rise to z_k + h_k - 0.2, and the NEXT storey's floor slab closes
them: one shared slab entity per level (owned by the storey above), the
roof slab caps the top. Room floors/ceilings of the apartment grammar are
dropped in favour of those slabs.

Stairs core. When `core.stairs` and storeys > 1, a 1.5 m strip along the
full chosen side is reserved; the floor layout is solved on the remaining
rectangle. Stairs run along the strip from its low end: a box-run of
steps = ceil(h/0.18) risers (width 1.1 m, tread 0.28 m, run = steps*0.28)
plus a 1.0 m landing; the reserved core rectangle is (strip width) x (the
longest run among the stair-served storeys + landing), so the core -- and
therefore every storey's windows -- does not depend on the TOP storey's
height. The rest of the strip is open lobby (no walls, no stairs).

Headroom. The floor slab above a stair is open over the stair from the
first tread whose underside clearance would drop below 2.0 m, so the
clearance at the slab edge is (h - 0.2) - j*riser >= 2.0 for the
j treads under the slab; `_stair_dims` checks it and raises ValueError.

Facade. Each exterior wall is cut into n = max(1, round(length / bay))
equal bays; one window per bay, centred (i.e. starting half a bay in). A
window is skipped when it would come within 0.3 m of a wall corner, within
0.3 m (in plan) of the stairs core, or -- on the ground floor south wall --
of the entrance door. Windows are WINDOW boxes in the wall plane (the wall
is not carved), like the room grammar's openings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from procedural.apartment_grammar import generate_apartment, plan_from_params
from procedural.layout_solver import ApartmentSpec
from procedural.room_grammar import _entity
from procedural.urban_grammar import _link, _semantic
from world_ir.schema_v1 import EntityType, Geometry, RelationshipKind
from world_ir.world_v1 import WorldIR

SIDES = ("east", "west", "north", "south")
SLAB = 0.2
T = 0.2                  # exterior wall thickness (== room grammar)
STAIR_W, RISER, TREAD, LANDING = 1.1, 0.18, 0.28, 1.0
CORE_W = 1.5             # strip: 0.2 layout wall + 0.1 + 1.1 stairs + 0.1
HEADROOM = 2.0
MARGIN = 0.3             # windows keep this far from corners / the core / the door
DOOR_W, DOOR_TOP = 0.9, 2.1
MAX_STOREYS, H_MIN, H_MAX = 30, 2.4, 6.0
MIN_SIDE, MAX_SIDE = 4.0, 200.0
FACADE_DEFAULTS = {"bay": 3.0, "window_w": 1.5, "window_h": 1.2, "sill": 0.9, "entrance": True}
FACADE_RANGES = {"bay": (1.5, 8.0), "window_w": (0.3, 4.0), "window_h": (0.3, 3.0),
                 "sill": (0.0, 2.0)}
CORE_DEFAULTS = {"stairs": True, "side": "east"}
_EPS = 1e-9


def _num(label: str, v, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{label} must be a finite number")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValueError(f"{label} must be in [{lo}, {hi}]")
    return float(v)


def _merged(label: str, given, defaults: dict) -> dict:
    if given is None:
        given = {}
    if not isinstance(given, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f"unknown {label} field(s): {sorted(unknown)}")
    return {**defaults, **given}


def _stair_dims(h: float) -> tuple:
    """(steps, riser, run, j): j = treads that may sit under the slab above
    (the slab is open from tread j on). Raises ValueError on < 2.0 m headroom."""
    steps = math.ceil(h / RISER - _EPS)
    riser = h / steps
    j = max(0, min(steps, math.floor((h - SLAB - HEADROOM) / riser + _EPS)))
    if (h - SLAB) - j * riser < HEADROOM - _EPS:
        raise ValueError(f"stairs for a {h:g} m storey leave less than {HEADROOM:g} m headroom")
    return steps, riser, steps * TREAD, j


@dataclass
class BuildingSpec:
    name: str
    footprint: tuple
    storeys: int = 1
    storey_heights: Optional[list] = None
    floor_layout: Optional[dict] = None   # ApartmentSpec fields (no name/footprint) [+ "rects"]
    plan_index: int = 0
    core: Optional[dict] = None
    facade: Optional[dict] = None
    roof: bool = True
    seed: int = 0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 200:
            raise ValueError("building name must be 1-200 characters")
        if not isinstance(self.footprint, (list, tuple)) or len(self.footprint) != 2:
            raise ValueError("footprint must be (width, depth)")
        self.footprint = tuple(_num(f"footprint[{i}]", v) for i, v in enumerate(self.footprint))
        if not all(MIN_SIDE < v <= MAX_SIDE for v in self.footprint):
            raise ValueError(f"footprint sides must be greater than {MIN_SIDE:g} m "
                             f"and at most {MAX_SIDE:g} m")
        if isinstance(self.storeys, bool) or not isinstance(self.storeys, int) \
                or not 1 <= self.storeys <= MAX_STOREYS:
            raise ValueError(f"storeys must be an integer in 1..{MAX_STOREYS}")
        if self.storey_heights is not None:
            if not isinstance(self.storey_heights, (list, tuple)) \
                    or len(self.storey_heights) != self.storeys:
                raise ValueError("storey_heights needs exactly one height per storey")
            self.storey_heights = [_num(f"storey_heights[{i}]", v, H_MIN, H_MAX)
                                   for i, v in enumerate(self.storey_heights)]
        self.core = _merged("core", self.core, CORE_DEFAULTS)
        if not isinstance(self.core["stairs"], bool):
            raise ValueError("core.stairs must be true or false")
        if self.core["side"] not in SIDES:
            raise ValueError(f"core.side must be one of {SIDES}")
        self.facade = _merged("facade", self.facade, FACADE_DEFAULTS)
        for key, (lo, hi) in FACADE_RANGES.items():
            self.facade[key] = _num(f"facade.{key}", self.facade[key], lo, hi)
        if not isinstance(self.facade["entrance"], bool):
            raise ValueError("facade.entrance must be true or false")
        if not isinstance(self.roof, bool):
            raise ValueError("roof must be true or false")
        for label in ("seed", "plan_index"):
            v = getattr(self, label)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{label} must be an integer")
        if not 0 <= self.plan_index < 10:
            raise ValueError("plan_index must be in 0..9")
        if self.floor_layout is not None and (
                not isinstance(self.floor_layout, dict) or not self.floor_layout.get("rooms")
                or not self.floor_layout.get("entrance")):
            raise ValueError("floor_layout needs rooms and an entrance")
        f = self.facade
        if f["sill"] + f["window_h"] > min(self.heights) - SLAB - 0.1 + _EPS:
            raise ValueError("facade sill + window_h must leave 0.1 m below the slab "
                             f"(clear height is {min(self.heights) - SLAB:g} m)")
        if f["window_w"] > f["bay"]:
            raise ValueError("facade.window_w must not exceed facade.bay")
        self._check_geometry()

    # ---------------------------------------------------------------- derived
    @property
    def heights(self) -> list:
        return list(self.storey_heights) if self.storey_heights else [3.0] * self.storeys

    @property
    def has_core(self) -> bool:
        return bool(self.core["stairs"]) and self.storeys > 1

    def elevations(self) -> list:
        out, z = [], 0.0
        for h in self.heights:
            out.append(round(z, 6))
            z += h
        return out

    def to_params(self) -> dict:
        """JSON-safe stored params (what GeneratedSpec.params holds)."""
        return {"footprint": list(self.footprint), "storeys": self.storeys,
                "storey_heights": self.storey_heights, "floor_layout": self.floor_layout,
                "plan_index": self.plan_index, "core": dict(self.core),
                "facade": dict(self.facade), "roof": self.roof, "seed": self.seed}

    @classmethod
    def from_params(cls, name: str, p: dict) -> "BuildingSpec":
        keys = ("footprint", "storeys", "storey_heights", "floor_layout", "plan_index",
                "core", "facade", "roof", "seed")
        return cls(name=name, **{k: p[k] for k in keys if k in p})

    def _check_geometry(self) -> None:
        w, d = self.footprint
        g = _core_geom(self)
        if g is not None:
            length = d if g["axis"] == "y" else w
            if g["length"] > length + _EPS:
                raise ValueError(
                    f"stairs do not fit: the run plus landing needs {g['length']:.2f} m "
                    f"but the {self.core['side']} side is {length:g} m long")
        if self.facade["entrance"]:
            lo, hi = _entrance_span(self)
            if hi - lo < DOOR_W + 2 * MARGIN - _EPS:
                raise ValueError("no room for the entrance door beside the stairs core")


def _core_geom(spec: BuildingSpec) -> Optional[dict]:
    """Strip geometry (local axes), or None without a stairs core."""
    if not spec.has_core:
        return None
    w, d = spec.footprint
    side = spec.core["side"]
    length = max(_stair_dims(h)[2] for h in spec.heights[:-1]) + LANDING
    if side in ("east", "west"):
        x0, x1 = (w - CORE_W, w) if side == "east" else (0.0, CORE_W)
        rect = (x0, 0.0, x1, length)
        across = (x0, x1)
    else:
        y0, y1 = (d - CORE_W, d) if side == "north" else (0.0, CORE_W)
        rect = (0.0, y0, length, y1)
        across = (y0, y1)
    far = side in ("east", "north")
    return {
        "axis": "y" if side in ("east", "west") else "x", "length": length, "rect": rect,
        # stairs sit on the outer side of the strip; the opening in the slab above
        # excludes the layout wall's 0.2 m band on the inner side
        "stairs": (across[1] - 0.1 - STAIR_W, across[1] - 0.1) if far
        else (across[0] + 0.1, across[0] + 0.1 + STAIR_W),
        "open": (across[0] + T, across[1]) if far else (across[0], across[1] - T),
    }


def core_rect(spec: BuildingSpec) -> Optional[Tuple[float, float, float, float]]:
    """The reserved core rectangle (x0, y0, x1, y1), local axes, or None."""
    g = _core_geom(spec)
    return g["rect"] if g else None


def _entrance_span(spec: BuildingSpec) -> Tuple[float, float]:
    """Free stretch of the south wall (x range) not taken by the stairs core."""
    w, _d = spec.footprint
    lo, hi = 0.0, w
    r = core_rect(spec)
    if r and r[1] <= _EPS:                  # the core touches the south wall
        if r[0] <= _EPS:
            lo = r[2]
        else:
            hi = r[0]
    return lo, hi


def layout_frame(spec: BuildingSpec) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """(footprint, origin) of the floor layout inside the building footprint."""
    w, d = spec.footprint
    if not spec.has_core:
        return (w, d), (0.0, 0.0)
    side = spec.core["side"]
    if side in ("east", "west"):
        return (w - CORE_W, d), ((CORE_W if side == "west" else 0.0), 0.0)
    return (w, d - CORE_W), (0.0, (CORE_W if side == "south" else 0.0))


def _layout_spec(spec: BuildingSpec, name: str, footprint) -> ApartmentSpec:
    return ApartmentSpec.from_dict({**spec.floor_layout, "name": name,
                                    "footprint": list(footprint)})


def floor_plan(spec: BuildingSpec):
    """The ONE plan every storey reuses (stored rects when still valid, else
    the solver's `plan_index` alternative). None without a floor layout."""
    if spec.floor_layout is None:
        return None
    fp, _org = layout_frame(spec)
    a = _layout_spec(spec, spec.name, fp)
    return plan_from_params(a, {"rects": spec.floor_layout.get("rects"),
                                "plan_index": spec.plan_index})


def _slab_pieces(w: float, d: float, hole) -> List[tuple]:
    if hole is None:
        return [(0.0, 0.0, w, d)]
    hx0, hy0, hx1, hy1 = hole
    cand = [(0.0, 0.0, hx0, d), (hx1, 0.0, w, d), (hx0, 0.0, hx1, hy0), (hx0, hy1, hx1, d)]
    return [p for p in cand if p[2] - p[0] > _EPS and p[3] - p[1] > _EPS]


def _overlap(a, b) -> bool:
    return a[0] < b[2] - _EPS and a[2] > b[0] + _EPS and a[1] < b[3] - _EPS and a[3] > b[1] + _EPS


def generate_building(spec: BuildingSpec, world: Optional[WorldIR] = None,
                      origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> WorldIR:
    """Build the whole building into `world` (a fresh WorldIR when omitted)."""
    name = spec.name
    if world is None:
        world = WorldIR(id=f"world-{name}", main_branch_id=f"branch-main-{name}")
    ox, oy, oz = origin
    w, d = spec.footprint
    heights, elev = spec.heights, spec.elevations()
    n = spec.storeys
    core = _core_geom(spec)
    plan = floor_plan(spec)                    # solved once, shared by every storey
    (lw, ld), (lx, ly) = layout_frame(spec)
    f = spec.facade

    def box(eid, etype, label, x0, y0, z0, x1, y1, z1):
        geoms: Dict[str, Geometry] = {}
        r = lambda v: round(v, 6)            # noqa: E731 -- keeps bytes stable
        e = _entity(eid, etype, label, (r(ox + x0), r(oy + y0), r(z0)),
                    (r(ox + x1), r(oy + y1), r(z1)), geoms)
        world.entities[eid] = e
        world.geometries.update(geoms)
        return e

    def drop(eid):
        e = world.entities.pop(eid)
        for gid in e.geometry_ids:
            world.geometries.pop(gid, None)

    def outside(bmin, bmax) -> bool:         # apartment wall outside the building footprint
        cx, cy = (bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2
        return not (ox <= cx <= ox + w and oy <= cy <= oy + d)

    # entrance door span (south wall, ground floor)
    door = None
    if f["entrance"]:
        lo, hi = _entrance_span(spec)
        c = (lo + hi) / 2
        door = (c - DOOR_W / 2, c + DOOR_W / 2)

    storey_ids, stair_ids = [], []
    for k in range(n):
        h, zk = heights[k], round(oz + elev[k], 6)
        clear = h - SLAB
        sid = f"{name}-s{k}"

        # floor slab (open over the stair coming up from the storey below)
        hole = None
        if core is not None and k >= 1:
            a1 = _stair_dims(heights[k - 1])[2]
            a0 = _stair_dims(heights[k - 1])[3] * TREAD
            o0, o1 = core["open"]
            hole = (o0, a0, o1, a1) if core["axis"] == "y" else (a0, o0, a1, o1)
        for i, p in enumerate(_slab_pieces(w, d, hole)):
            box(f"{sid}-floor-{i}", EntityType.FLOOR, f"{name} storey {k} floor {i}",
                p[0], p[1], zk - SLAB, p[2], p[3], zk)

        # exterior walls (corner-closed on the south / north faces)
        walls = {"south": (-T, -T, w + T, 0.0), "north": (-T, d, w + T, d + T),
                 "west": (-T, 0.0, 0.0, d), "east": (w, 0.0, w + T, d)}
        for side, (x0, y0, x1, y1) in walls.items():
            box(f"{sid}-ext-{side}", EntityType.WALL, f"{name} storey {k} {side} wall",
                x0, y0, zk, x1, y1, zk + clear)

        # interior: the shared floor layout, or one open plate
        storey = _semantic(sid, EntityType.STOREY, f"{name} storey {k}")
        storey.custom_properties.update(level=k, elevation=elev[k], height=h)
        if plan is not None:
            pfx = f"{name}-s{k}"
            a_k = _layout_spec(spec, pfx, (lw, ld))
            generate_apartment(a_k, plan, world=world, origin=(ox + lx, oy + ly, zk),
                               height=clear, entrance_door=False, skip_wall=outside)
            for r in a_k.rooms:
                drop(f"{pfx}-{r.id}-floor")
                drop(f"{pfx}-{r.id}-ceiling")
            storey.relationships = world.entities.pop(f"{pfx}-apartment").relationships
        else:
            world.entities[f"{sid}-room"] = _semantic(f"{sid}-room", EntityType.ROOM,
                                                      f"{name} storey {k} open plan")
            storey.relationships = [_link(RelationshipKind.CONTAINS, f"{sid}-room")]
        world.entities[sid] = storey
        storey_ids.append(sid)

        # facade windows (plan boxes in the wall plane)
        sill, wh, ww = f["sill"], f["window_h"], f["window_w"]
        for side in ("south", "north", "east", "west"):
            length = w if side in ("south", "north") else d
            bays = max(1, round(length / f["bay"]))
            eff = length / bays
            for i in range(bays):
                a = (i + 0.5) * eff - ww / 2
                b = a + ww
                if a < MARGIN - _EPS or b > length - MARGIN + _EPS:
                    continue
                if side in ("south", "north"):
                    y0 = -T if side == "south" else d
                    plan_box = (a, y0, b, y0 + T)
                else:
                    x0 = -T if side == "west" else w
                    plan_box = (x0, a, x0 + T, b)
                if core is not None:
                    cr = core["rect"]
                    if _overlap(plan_box, (cr[0] - MARGIN, cr[1] - MARGIN,
                                           cr[2] + MARGIN, cr[3] + MARGIN)):
                        continue
                if k == 0 and door and side == "south" \
                        and a < door[1] + MARGIN - _EPS and b > door[0] - MARGIN + _EPS:
                    continue
                box(f"{sid}-win-{side}-{i}", EntityType.WINDOW,
                    f"{name} storey {k} {side} window {i}",
                    plan_box[0], plan_box[1], zk + sill, plan_box[2], plan_box[3], zk + sill + wh)

    # stairs: one box-run per consecutive storey pair, inside the core strip
    if core is not None:
        s_lo, s_hi = core["stairs"]
        for k in range(n - 1):
            steps, riser, _run, _j = _stair_dims(heights[k])
            z0 = round(oz + elev[k], 6)
            eid = f"{name}-stairs-{k}"
            for i in range(steps):
                a0, a1 = i * TREAD, (i + 1) * TREAD
                rect = (s_lo, a0, s_hi, a1) if core["axis"] == "y" else (a0, s_lo, a1, s_hi)
                if i == 0:
                    e = box(eid, EntityType.STAIRS, f"{name} stairs {k}-{k + 1}",
                            rect[0], rect[1], z0, rect[2], rect[3], z0 + (i + 1) * riser)
                else:
                    geoms: Dict[str, Geometry] = {}
                    r = lambda v: round(v, 6)   # noqa: E731
                    _entity(f"{eid}-step{i}", EntityType.STAIRS, "step",
                            (r(ox + rect[0]), r(oy + rect[1]), r(z0)),
                            (r(ox + rect[2]), r(oy + rect[3]), r(z0 + (i + 1) * riser)), geoms)
                    world.geometries.update(geoms)
                    e.geometry_ids.append(f"geom-{eid}-step{i}")
            e.relationships = [_link(RelationshipKind.ADJACENT_TO, f"{name}-s{k}"),
                               _link(RelationshipKind.ADJACENT_TO, f"{name}-s{k + 1}")]
            e.custom_properties.update(rise=heights[k], steps=steps)
            stair_ids.append(eid)

    # ground-floor entrance
    if door:
        e = box(f"{name}-entrance", EntityType.DOOR, f"{name} entrance",
                door[0], -T, round(oz, 6), door[1], 0.0, round(oz + DOOR_TOP, 6))
        e.relationships = [_link(RelationshipKind.CONNECTS, f"{name}-s0")]

    # roof slab
    roof_ids = []
    if spec.roof:
        top = round(oz + sum(heights), 6)
        box(f"{name}-roof", EntityType.ROOF, f"{name} roof",
            -T, -T, top - SLAB, w + T, d + T, top)
        roof_ids = [f"{name}-roof"]

    building = _semantic(f"{name}-building", EntityType.BUILDING, name)
    building.custom_properties.update(building=True, storeys=n)
    building.relationships = [_link(RelationshipKind.CONTAINS, t)
                              for t in storey_ids + stair_ids + roof_ids]
    world.entities[building.id] = building
    return world
