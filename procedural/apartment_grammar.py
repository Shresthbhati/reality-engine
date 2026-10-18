"""Procedural apartment grammar (World Factory, task C3).

Turns an ApartmentSpec + a solved Plan (procedural/layout_solver.py) into
a WorldIR: one room per plan rectangle, walls, doors and, only where a
room would otherwise have no door path to the entrance, a short CORRIDOR
link. Same contract as the other grammars: deterministic, every
statement PROCEDURAL / GENERATED, authoring in local Z-up axes (x east,
y north, z up) emitted Y-up through room_grammar._entity.

How it reuses the room grammar
  * Each room is built by `generate_room` with `exterior_walls=set()`:
    that yields its floor, ceiling and ROOM entity ("{apt}-{room}-..."),
    and NO walls.
  * Walls are then derived once from the whole plan: every footprint
    boundary line is cut at every room corner on it, and each maximal
    piece (same room on each side) becomes exactly ONE wall entity. That
    is `generate_floor_layout`'s "one wall entity per boundary segment"
    rule generalised to rooms of different sizes, where a single edge of
    one room can border several neighbours (the grid-based
    `exterior_walls` switch cannot express that). A shared wall sits on
    the +x / +y side of the boundary (the same band `generate_room` gives
    the left/lower room's east/south wall); a wall with a single room
    behind it sits on that room's own outer side.
  * A door is a DOOR entity, 0.9 m wide, bottom 0, top 2.1, centred on the
    shared piece -- like the room grammar's openings it is a box in the
    wall plane (the wall is not carved).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from procedural.layout_solver import (
    DOOR_WIDTH_DM,
    ApartmentSpec,
    Plan,
    _adj_dm,
    _dm,
    solve_layout,
    validate_plan,
)
from procedural.room_grammar import (
    _PROCEDURAL_PROVENANCE,
    _PROCEDURAL_STATE,
    _WALL_THICKNESS,
    RoomGrammarSpec,
    _entity,
    generate_room,
)
from procedural.urban_grammar import CorridorGrammarSpec, _link, _semantic, generate_corridor
from world_ir.schema_v1 import EntityType, Geometry, RelationshipKind
from world_ir.world_v1 import WorldIR

DOOR_TOP = 2.1
DEFAULT_HEIGHT = 2.8
_T_DM = _dm(_WALL_THICKNESS)


def plan_from_params(spec: ApartmentSpec, params: dict) -> Plan:
    """The plan a stored apartment spec should be built from.

    `params["rects"]` is used when it still validates against the spec;
    otherwise (footprint or room list was edited) the spec is re-solved
    deterministically and `params["plan_index"]` picks the alternative.
    Raises ValueError with the solver's violations when nothing fits.
    """
    rects = params.get("rects")
    if rects:
        rects = {k: tuple(v) for k, v in rects.items()}
        if not validate_plan(spec, rects):
            return Plan(rects=rects, score=0.0, satisfied=[], unsatisfied=[])
    index = int(params.get("plan_index", 0) or 0)
    result = solve_layout(spec, max_alternatives=max(index + 1, 1))
    if not result.ok:
        raise ValueError("no valid apartment layout: " + "; ".join(result.violations))
    return result.plans[min(index, len(result.plans) - 1)]


def _wall_pieces(irects: Dict[str, tuple]) -> List[tuple]:
    """Maximal boundary pieces: (orient, line, lo, hi, first, second).

    orient "v": a vertical line x=line spanning y in [lo, hi]; first is the
    room on its -x side, second on its +x side. orient "h": horizontal line
    y=line spanning x in [lo, hi]; first is the room below (smaller y).
    A missing neighbour (footprint edge) is None.
    """
    pieces: List[tuple] = []
    for orient in ("v", "h"):
        lines = sorted({(r[0] if orient == "v" else r[1]) for r in irects.values()}
                       | {(r[0] + r[2] if orient == "v" else r[1] + r[3]) for r in irects.values()})
        for line in lines:
            def first_of(r):   # room ends at the line
                return (r[0] + r[2] if orient == "v" else r[1] + r[3]) == line

            def second_of(r):  # room starts at the line
                return (r[0] if orient == "v" else r[1]) == line

            def span(r):
                return (r[1], r[1] + r[3]) if orient == "v" else (r[0], r[0] + r[2])

            on_line = {rid: r for rid, r in irects.items() if first_of(r) or second_of(r)}
            cuts = sorted({c for r in on_line.values() for c in span(r)})
            run: Optional[list] = None
            for lo, hi in zip(cuts, cuts[1:]):
                f = next((rid for rid, r in on_line.items()
                          if first_of(r) and span(r)[0] <= lo and span(r)[1] >= hi), None)
                s = next((rid for rid, r in on_line.items()
                          if second_of(r) and span(r)[0] <= lo and span(r)[1] >= hi), None)
                if f is None and s is None:
                    run = None
                    continue
                if run and run[0] == (f, s) and run[2] == lo:
                    run[2] = hi
                    pieces[-1] = (orient, line, run[1], hi, f, s)
                else:
                    run = [(f, s), lo, hi]
                    pieces.append((orient, line, lo, hi, f, s))
    return pieces


def generate_apartment(spec: ApartmentSpec, plan, world: Optional[WorldIR] = None,
                       origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                       height: float = DEFAULT_HEIGHT) -> WorldIR:
    """Build the apartment into `world` (a fresh WorldIR when omitted).

    `plan` is a layout_solver.Plan or a {room_id: (x, y, w, d)} dict; it is
    re-validated against the spec (ValueError when it does not hold).
    """
    rects = plan.rects if isinstance(plan, Plan) else {k: tuple(v) for k, v in plan.items()}
    problems = validate_plan(spec, rects)
    if problems:
        raise ValueError("plan does not satisfy the apartment spec: " + "; ".join(problems))
    if not (isinstance(height, (int, float)) and 0 < height < 20):
        raise ValueError("apartment height must be in (0, 20) metres")
    apt = spec.name
    if world is None:
        world = WorldIR(id=f"world-{apt}", main_branch_id=f"branch-main-{apt}")
    ox, oy, oz = origin
    irects = {rid: tuple(_dm(v) for v in r) for rid, r in rects.items()}
    order = [r.id for r in spec.rooms]
    kinds = {r.id: r.kind for r in spec.rooms}
    t = _WALL_THICKNESS

    # 1. rooms: floor + ceiling + ROOM entity each, no walls
    for rid in order:
        x, y, w, d = rects[rid]
        generate_room(RoomGrammarSpec(f"{apt}-{rid}", w, d, height), world=world,
                      origin=(ox + x, oy + y, oz), exterior_walls=set())
        world.entities[f"{apt}-{rid}-room"].custom_properties["room_kind"] = kinds[rid]

    def add_box(eid: str, etype: EntityType, name: str, bmin, bmax):
        geoms: Dict[str, Geometry] = {}
        entity = _entity(eid, etype, name, bmin, bmax, geoms)
        world.entities[eid] = entity
        world.geometries.update(geoms)
        return entity

    def band(piece, lo_dm: int, hi_dm: int, z0: float, z1: float):
        """Local-axis box of `piece` between lo..hi (dm) along it."""
        orient, line, _l, _h, first, _second = piece
        if orient == "v":
            x0 = ox + (line / 10.0 if first is not None else line / 10.0 - t)
            return ((x0, oy + lo_dm / 10.0, oz + z0), (x0 + t, oy + hi_dm / 10.0, oz + z1))
        y0 = oy + (line / 10.0 if first is not None else line / 10.0 - t)
        return ((ox + lo_dm / 10.0, y0, oz + z0), (ox + hi_dm / 10.0, y0 + t, oz + z1))

    # 2. walls: one entity per maximal boundary piece
    pieces = _wall_pieces(irects)
    for piece in pieces:
        orient, line, lo, hi, _f, _s = piece
        bmin, bmax = band(piece, lo, hi, 0.0, height)
        add_box(f"{apt}-wall-{orient}{line}-{lo}-{hi}", EntityType.WALL,
                f"{apt} wall {orient}{line} [{lo / 10:g}..{hi / 10:g}]", bmin, bmax)

    # 3. doors
    door_pairs = set()

    def cut_door(eid: str, piece, rooms: tuple, name: str) -> None:
        _o, _line, lo, hi, _f, _s = piece
        start = lo + ((hi - lo) - DOOR_WIDTH_DM) // 2
        bmin, bmax = band(piece, start, start + DOOR_WIDTH_DM, 0.0, DOOR_TOP)
        door = add_box(eid, EntityType.DOOR, name, bmin, bmax)
        door.relationships = [_link(RelationshipKind.CONNECTS, f"{apt}-{r}-room")
                              for r in rooms]

    def shared_piece(a: str, b: str):
        for p in pieces:
            if {p[4], p[5]} == {a, b}:
                return p
        return None

    def connect(a: str, b: str) -> None:
        lo_id, hi_id = sorted((a, b))
        door_pairs.add((lo_id, hi_id))
        cut_door(f"{apt}-door-{lo_id}-{hi_id}", shared_piece(a, b), (lo_id, hi_id),
                 f"{apt} door {lo_id}/{hi_id}")

    for a, b, _lvl in spec.adjacency:
        if _adj_dm(irects[a], irects[b]) >= DOOR_WIDTH_DM and \
                tuple(sorted((a, b))) not in door_pairs:
            connect(a, b)

    exterior = [p for p in pieces
                if spec.entrance in (p[4], p[5]) and (p[4] is None or p[5] is None)
                and p[3] - p[2] >= DOOR_WIDTH_DM]
    if not exterior:       # validate_plan guarantees one; defend the invariant
        raise ValueError(f"entrance room {spec.entrance!r} has no exterior wall for a door")
    entrance_piece = max(exterior, key=lambda p: (p[3] - p[2], p[0] == "v", -p[1], -p[2]))
    cut_door(f"{apt}-door-entrance", entrance_piece, (spec.entrance,),
             f"{apt} entrance door")

    # 4. reachability: link any room without a door path with a CORRIDOR
    links: List[str] = []

    def reached() -> set:
        seen, stack = {spec.entrance}, [spec.entrance]
        while stack:
            cur = stack.pop()
            for a, b in door_pairs:
                for u, v in ((a, b), (b, a)):
                    if u == cur and v not in seen:
                        seen.add(v)
                        stack.append(v)
        return seen

    while True:
        seen = reached()
        missing = [r for r in order if r not in seen]
        if not missing:
            break
        cands = [(_adj_dm(irects[u], irects[v]), u, v)
                 for u in missing for v in order if v in seen
                 if _adj_dm(irects[u], irects[v]) >= DOOR_WIDTH_DM]
        if not cands:
            raise ValueError(
                f"room(s) {missing} cannot be connected to the entrance: none shares "
                "a wall of at least 0.9 m with a reachable room")
        _len, u, v = min(cands, key=lambda c: (-c[0], order.index(c[1]), order.index(c[2])))
        piece = shared_piece(u, v)
        connect(u, v)
        links.append(_add_link_corridor(world, apt, u, v, piece, band, oz, ox, oy, t))

    # 5. the apartment entity relates everything it owns
    apartment = _semantic(f"{apt}-apartment", EntityType.BUILDING, apt)
    apartment.custom_properties["apartment"] = True
    apartment.relationships = (
        [_link(RelationshipKind.CONTAINS, f"{apt}-{rid}-room") for rid in order]
        + [_link(RelationshipKind.CONTAINS, f"{apt}-link-{u}-corridor") for u in links])
    world.entities[apartment.id] = apartment
    return world


def _add_link_corridor(world, apt, u, v, piece, band, oz, ox, oy, t) -> str:
    """A passage CORRIDOR through the wall between u and v (generate_corridor);
    its own end walls are dropped -- the apartment's wall pieces already
    bound the space."""
    orient, line, lo, hi, first, _second = piece
    start = lo + ((hi - lo) - DOOR_WIDTH_DM) // 2
    (bx0, by0, _), _ = band(piece, start, start + DOOR_WIDTH_DM, 0.0, DOOR_TOP)
    name = f"{apt}-link-{u}"
    spec = CorridorGrammarSpec(
        name, length=t, width=DOOR_WIDTH_DM / 10.0, height=DOOR_TOP,
        axis="x" if orient == "v" else "y",
        connects=[f"{apt}-{u}-room", f"{apt}-{v}-room"])
    generate_corridor(spec, world=world, origin=(bx0, by0, oz))
    for wall in ("north", "south") if orient == "v" else ("west", "east"):
        eid = f"{name}-wall-{wall}"
        for gid in world.entities[eid].geometry_ids:
            world.geometries.pop(gid, None)
        del world.entities[eid]
    return u
