"""The Apartment builder's UI templates (frontend apartment-model.ts) must be solvable
with the adjacency the UI prefills (hallway touches every bedroom/bathroom/living;
living touches kitchen), on the default auto footprint, with every required pair met."""

from __future__ import annotations

import re
from pathlib import Path

from procedural import layout_solver as L

SRC = Path(__file__).resolve().parents[1] / "frontend/src/lib/procedural/apartment-model.ts"
ROW = re.compile(r'row\("([\w-]+)",\s*"(\w+)",\s*([\d.]+),\s*([\d.]+)\)')


def _templates() -> dict[str, list[tuple[str, str, float, float]]]:
    text = SRC.read_text(encoding="utf-8")
    block = text[text.index("APARTMENT_TEMPLATES"):text.index("export function templateRows")]
    out = {}
    for part in block.split("key:")[1:]:
        key = re.match(r'\s*"([\w-]+)"', part).group(1)
        out[key] = [(i, k, float(a), float(b)) for i, k, a, b in ROW.findall(part)]
    return out


def test_ui_templates_solve_with_prefilled_adjacency():
    templates = _templates()
    assert set(templates) == {"studio", "2-bedroom", "3-bedroom"}
    for key, rows in templates.items():
        adj = [("hallway", i, "required") for i, k, _, _ in rows if k in ("bedroom", "bathroom", "living")]
        adj.append(("living", "kitchen", "required"))
        spec = L.ApartmentSpec(
            name=key, entrance="hallway", adjacency=adj,
            rooms=[dict(id=i, kind=k, min_area=a, max_area=b) for i, k, a, b in rows])
        res = L.solve_layout(spec, 3)
        assert res.ok, (key, res.violations)
        assert res.plans
        for p in res.plans:
            assert not p.unsatisfied_required, (key, p.unsatisfied)
            assert len(p.satisfied) >= len(adj)
