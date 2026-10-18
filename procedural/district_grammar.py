"""Procedural district grammar (World Factory, task C5).

A DistrictSpec (data) becomes a WorldIR: a named street network, one parcel
(PLOT) per lot, one building per parcel drawn from a few seeded building
FAMILIES, under a per-parcel detail policy so large districts stay cheap:

  full            -> the real building_grammar building (storeys, facade
                     windows, stairs, roof) at the parcel.
  footprint_only  -> ONE BUILDING entity with ONE box (the massing: parcel
                     footprint minus setback x summed storey heights).

Layouts: "grid" (blocks_x x blocks_y parcels, a named street on every
parcel line, like generate_city) or "spine" (one Main Street along x with a
row of parcels on each side and a cross street every `cross_every` parcels).

Determinism and locality. Everything derives from the spec; the only
randomness is a per-parcel `random.Random(f"{seed}:{i}:{j}")` and a
`random.Random(seed)` that resolves default families, so a parcel's family
never depends on any other parcel. `parcels` overrides ({"i,j": {"family":
k, "detail": "full"|"footprint_only"}}) therefore change exactly that
parcel; regeneration then reports every other entity byte-identical.

Detail policy (`detail`): parcels are ranked by distance (m) from `focus`
(default the district centre). A parcel is "full" when it is within
`full_within` metres (if set) AND among the `max_full` nearest (if set);
the defaults are max_full = 25 and no radius. Per-parcel overrides win.

Authoring is grammar-local Z-up (x east, y north, z up) emitted Y-up via
room_grammar._entity. All dimensions are metres; every statement is
PROCEDURAL / GENERATED. Nothing here is surveyed or captured.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from procedural.building_grammar import BuildingSpec, generate_building
from procedural.room_grammar import _entity
from procedural.urban_grammar import _fresh_world, _link, _semantic
from world_ir.schema_v1 import EntityType, Geometry, RelationshipKind
from world_ir.world_v1 import WorldIR

LAYOUTS = ("grid", "spine")
DETAILS = ("full", "footprint_only")
MAX_PARCELS = 1000
MAX_FAMILIES = 12
DEFAULT_MAX_FULL = 25
SLAB = 0.2
FAMILY_KEYS = {"name", "storeys", "storey_height", "facade", "stairs"}


def _num(label, v, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) \
            or not lo <= v <= hi:
        raise ValueError(f"{label} must be a finite number in [{lo:g}, {hi:g}]")
    return float(v)


def _default_families(seed: int) -> list:
    """Three seeded variants (low / mid / tall). The numbers are drawn from
    the seed and STORED in the spec params -- never re-derived implicitly."""
    rng = random.Random(seed)
    ranges = [(1, 2), (3, 4), (5, 8)]
    return [{"name": f"family-{k}",
             "storeys": rng.randint(lo, hi),
             "storey_height": round(rng.uniform(2.8, 3.6), 1),
             "facade": {"bay": round(rng.uniform(2.5, 4.0), 1)},
             "stairs": True}
            for k, (lo, hi) in enumerate(ranges)]


@dataclass
class DistrictSpec:
    name: str
    layout: str = "grid"
    blocks_x: int = 4
    blocks_y: int = 4
    lot_w: float = 24.0
    lot_d: float = 24.0
    street_width: float = 8.0
    setback: float = 3.0
    cross_every: int = 4                 # spine: parcels between cross streets
    families: Optional[list] = None
    detail: Optional[dict] = None        # {"max_full": int|None, "full_within": m|None, "focus": [x, y]|None}
    parcels: Optional[dict] = None       # {"i,j": {"family": int, "detail": "full"|"footprint_only"}}
    seed: int = 0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 200:
            raise ValueError("district name must be 1-200 characters")
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        for label in ("blocks_x", "blocks_y", "cross_every", "seed"):
            v = getattr(self, label)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{label} must be an integer")
        if self.layout == "spine":
            self.blocks_y = 2            # one row each side of the spine
        if self.blocks_x < 1 or self.blocks_y < 1 or self.cross_every < 1:
            raise ValueError("a district needs at least one parcel in each direction")
        if self.blocks_x * self.blocks_y > MAX_PARCELS:
            raise ValueError(f"a district is limited to {MAX_PARCELS} parcels")
        self.lot_w = _num("lot_w", self.lot_w, 8, 200)
        self.lot_d = _num("lot_d", self.lot_d, 8, 200)
        self.street_width = _num("street_width", self.street_width, 1, 60)
        self.setback = _num("setback", self.setback, 0, 40)

        fams = self.families if self.families is not None else _default_families(self.seed)
        if not isinstance(fams, list) or not 1 <= len(fams) <= MAX_FAMILIES:
            raise ValueError(f"families must be a list of 1..{MAX_FAMILIES} entries")
        self.families = [self._family(k, f) for k, f in enumerate(fams)]

        d = self.detail or {}
        if not isinstance(d, dict) or set(d) - {"max_full", "full_within", "focus"}:
            raise ValueError("detail accepts only max_full, full_within, focus")
        mf = d.get("max_full", DEFAULT_MAX_FULL)
        if mf is not None and (isinstance(mf, bool) or not isinstance(mf, int) or mf < 0):
            raise ValueError("detail.max_full must be a non-negative integer or null")
        fw = d.get("full_within")
        if fw is not None:
            fw = _num("detail.full_within", fw, 0, 1e6)
        focus = d.get("focus")
        if focus is not None:
            if not isinstance(focus, (list, tuple)) or len(focus) != 2:
                raise ValueError("detail.focus must be [x, y] metres")
            focus = [_num("detail.focus", v, -1e6, 1e6) for v in focus]
        self.detail = {"max_full": mf, "full_within": fw, "focus": focus}

        cleaned = {}
        for key, ov in (self.parcels or {}).items():
            i, j = self._parse_key(key)
            if not isinstance(ov, dict) or set(ov) - {"family", "detail"} or not ov:
                raise ValueError(f"parcel {key!r}: override takes family and/or detail")
            if "family" in ov and (isinstance(ov["family"], bool)
                                   or not isinstance(ov["family"], int)
                                   or not 0 <= ov["family"] < len(self.families)):
                raise ValueError(f"parcel {key!r}: family must index 0..{len(self.families) - 1}")
            if "detail" in ov and ov["detail"] not in DETAILS:
                raise ValueError(f"parcel {key!r}: detail must be one of {DETAILS}")
            cleaned[f"{i},{j}"] = dict(ov)
        self.parcels = cleaned

    # -------------------------------------------------------------- helpers
    def _parse_key(self, key) -> Tuple[int, int]:
        try:
            i, j = (int(p) for p in str(key).split(","))
        except ValueError:
            raise ValueError(f"parcel key {key!r} must look like 'i,j'") from None
        if not (0 <= i < self.blocks_x and 0 <= j < self.blocks_y):
            raise ValueError(f"parcel {key!r} is outside the {self.blocks_x}x{self.blocks_y} district")
        return i, j

    @property
    def building_footprint(self) -> Tuple[float, float]:
        return (self.lot_w - 2 * self.setback, self.lot_d - 2 * self.setback)

    def _family(self, k: int, f: dict) -> dict:
        if not isinstance(f, dict) or set(f) - FAMILY_KEYS:
            raise ValueError(f"family {k}: unknown field(s) {sorted(set(f) - FAMILY_KEYS) if isinstance(f, dict) else f!r}")
        fam = {"name": str(f.get("name") or f"family-{k}"),
               "storeys": f.get("storeys", 3),
               "storey_height": _num(f"family {k} storey_height", f.get("storey_height", 3.0), 2.4, 6.0),
               "facade": dict(f.get("facade") or {}),
               "stairs": f.get("stairs", True)}
        if isinstance(fam["storeys"], bool) or not isinstance(fam["storeys"], int):
            raise ValueError(f"family {k}: storeys must be an integer")
        self.building_spec(fam, "probe")        # full validation (ValueError)
        return fam

    def building_spec(self, fam: dict, name: str) -> BuildingSpec:
        return BuildingSpec(
            name=name, footprint=self.building_footprint, storeys=fam["storeys"],
            storey_heights=[fam["storey_height"]] * fam["storeys"],
            core={"stairs": fam["stairs"], "side": "east"},
            facade=dict(fam["facade"]), roof=True, seed=self.seed)

    def to_params(self) -> dict:
        """JSON-safe stored params (what GeneratedSpec.params holds)."""
        return {"layout": self.layout, "blocks_x": self.blocks_x, "blocks_y": self.blocks_y,
                "lot_w": self.lot_w, "lot_d": self.lot_d, "street_width": self.street_width,
                "setback": self.setback, "cross_every": self.cross_every,
                "families": self.families, "detail": self.detail,
                "parcels": self.parcels, "seed": self.seed}

    @classmethod
    def from_params(cls, name: str, p: dict) -> "DistrictSpec":
        keys = ("layout", "blocks_x", "blocks_y", "lot_w", "lot_d", "street_width",
                "setback", "cross_every", "families", "detail", "parcels", "seed")
        return cls(name=name, **{k: p[k] for k in keys if k in p})

    # ----------------------------------------------------------- road/parcel plan
    def plan(self):
        """(streets, parcels, (width, depth)) in local metres from the origin.
        streets: [(id_suffix, label, x0, y0, x1, y1)]; parcels: {(i, j): (x, y)}
        (min corner of each lot)."""
        w, sw = self.lot_w, self.street_width
        pw, pd = w + sw, self.lot_d + sw
        n = self.name
        streets, parcels = [], {}
        if self.layout == "grid":
            tw, td = self.blocks_x * pw + sw, self.blocks_y * pd + sw
            for k in range(self.blocks_y + 1):
                streets.append((f"street-x{k}", f"{n} Street {k + 1}", 0.0, k * pd, tw, k * pd + sw))
            for k in range(self.blocks_x + 1):
                streets.append((f"street-y{k}", f"{n} Avenue {k + 1}", k * pw, 0.0, k * pw + sw, td))
            for j in range(self.blocks_y):
                for i in range(self.blocks_x):
                    parcels[(i, j)] = (sw + i * pw, sw + j * pd)
            return streets, parcels, (tw, td)
        # spine: row 0 south of the main street, row 1 north of it
        groups = (self.blocks_x - 1) // self.cross_every + 1
        tw = self.blocks_x * w + (groups + 1) * sw
        row = self.lot_d
        td = 2 * row + sw
        for i in range(self.blocks_x):
            x = sw + i * w + (i // self.cross_every) * sw
            parcels[(i, 0)] = (x, 0.0)
            parcels[(i, 1)] = (x, row + sw)
        streets.append(("street-main", f"{n} Main Street", 0.0, row, tw, row + sw))
        for k in range(groups + 1):
            x0 = k * (self.cross_every * w + sw) if k < groups else tw - sw
            streets.append((f"street-c{k}", f"{n} Cross Street {k + 1}", x0, 0.0, x0 + sw, td))
        return streets, parcels, (tw, td)


def _ranked_full(spec: DistrictSpec, parcels: dict, extent) -> set:
    d = spec.detail
    fx, fy = d["focus"] or (extent[0] / 2, extent[1] / 2)
    cx, cy = spec.building_footprint[0] / 2, spec.building_footprint[1] / 2

    def dist(key):
        x, y = parcels[key]
        return math.hypot(x + spec.setback + cx - fx, y + spec.setback + cy - fy)

    ranked = sorted(parcels, key=lambda k: (round(dist(k), 6), k[1], k[0]))
    if d["full_within"] is not None:
        ranked = [k for k in ranked if dist(k) <= d["full_within"] + 1e-9]
    if d["max_full"] is not None:
        ranked = ranked[:d["max_full"]]
    return set(ranked)


def parcel_assignment(spec: DistrictSpec) -> Dict[Tuple[int, int], Tuple[int, str]]:
    """{(i, j): (family index, 'full'|'footprint_only')} -- the whole policy."""
    _, parcels, extent = spec.plan()
    full = _ranked_full(spec, parcels, extent)
    out = {}
    for key in parcels:
        i, j = key
        fam = random.Random(f"{spec.seed}:{i}:{j}").randrange(len(spec.families))
        det = "full" if key in full else "footprint_only"
        ov = spec.parcels.get(f"{i},{j}", {})
        out[key] = (ov.get("family", fam), ov.get("detail", det))
    return out


def generate_district(spec: DistrictSpec, world: Optional[WorldIR] = None,
                      origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> WorldIR:
    if world is None:
        world = _fresh_world(spec.name)
    ox, oy, oz = origin
    r = lambda v: round(v, 6)                                   # noqa: E731
    district = _semantic(f"{spec.name}-district", EntityType.DISTRICT, spec.name)
    district.custom_properties.update(layout=spec.layout, units="m")
    streets, parcels, _extent = spec.plan()
    assignment = parcel_assignment(spec)

    for suffix, label, x0, y0, x1, y1 in streets:
        geoms: Dict[str, Geometry] = {}
        eid = f"{spec.name}-{suffix}"
        ent = _entity(eid, EntityType.STREET, label,
                      (r(ox + x0), r(oy + y0), r(oz - SLAB)), (r(ox + x1), r(oy + y1), r(oz)), geoms)
        world.entities[eid] = ent
        world.geometries.update(geoms)
        district.relationships.append(_link(RelationshipKind.CONTAINS, eid))

    for (i, j), (px, py) in sorted(parcels.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        fam_idx, detail = assignment[(i, j)]
        fam = spec.families[fam_idx]
        bname = f"{spec.name}-p{i}-{j}"
        bx, by = px + spec.setback, py + spec.setback
        bspec = spec.building_spec(fam, bname)
        if detail == "full":
            generate_building(bspec, world=world, origin=(ox + bx, oy + by, oz))
            building = world.entities[f"{bname}-building"]
        else:
            height = sum(bspec.heights)
            geoms = {}
            building = _entity(f"{bname}-building", EntityType.BUILDING, bname,
                               (r(ox + bx), r(oy + by), r(oz)),
                               (r(ox + bx + bspec.footprint[0]), r(oy + by + bspec.footprint[1]),
                                r(oz + height)), geoms)
            building.custom_properties.update(building=True, storeys=bspec.storeys)
            world.entities[building.id] = building
            world.geometries.update(geoms)
        building.custom_properties.update(parcel=[i, j], family=fam_idx,
                                          family_name=fam["name"], detail=detail)
        plot = _semantic(f"{bname}-plot", EntityType.PLOT, f"{spec.name} parcel {i},{j}")
        plot.custom_properties.update(
            parcel=[i, j], lot_m=[spec.lot_w, spec.lot_d], setback_m=spec.setback)
        plot.relationships.append(_link(RelationshipKind.CONTAINS, building.id))
        world.entities[plot.id] = plot
        district.relationships.append(_link(RelationshipKind.CONTAINS, plot.id))

    world.entities[district.id] = district
    return world

