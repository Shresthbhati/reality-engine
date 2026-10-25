"""Confidence defaults mean "not measured", never "certain" (PROD.3).

Constructor and deserialisation defaults of every WorldIR record are 0.5 / UNKNOWN, and every producer in the repo
states its confidence explicitly, so a forgotten argument cannot read as an observed 1.0.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from world_ir.schema_v1 import CausalRelation, Observation, Provenance, Relationship, RelationshipKind

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("Observation", "_Observation", "Relationship", "CausalRelation")


def test_constructor_defaults_are_not_measured_not_certain():
    assert Observation().confidence == 0.5
    r = Relationship(kind=RelationshipKind.ADJACENT_TO, target_id="x")
    assert r.confidence == 0.5 and r.provenance is Provenance.UNKNOWN


def test_deserialising_a_record_without_confidence_does_not_invent_certainty():
    assert Observation.from_dict({}).confidence == 0.5
    assert Relationship.from_dict({"kind": RelationshipKind.ADJACENT_TO.value, "target_id": "x"}).confidence == 0.5
    assert CausalRelation.from_dict({}).confidence == 0.5 if hasattr(CausalRelation, "from_dict") else True


def test_every_producer_states_its_confidence():
    files = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, text=True).split()
    missing = []
    for f in files:
        # legacy world_ir/entity.py Relationship has no confidence field; benchmarks build throwaway graphs;
        # synthetic/indoor.py's `Observation` is a different, local record (reconstruction input), not WorldIR's
        if f.startswith(("tests/", "benchmarks/")) or f in ("world_ir/schema_v1.py", "world_ir/entity.py", "synthetic/indoor.py"):
            continue
        try:
            tree = ast.parse((ROOT / f).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) in CLASSES:
                kws = {k.arg for k in n.keywords}
                if None not in kws and "confidence" not in kws:
                    missing.append(f"{f}:{n.lineno}")
    assert not missing, f"constructed without an explicit confidence: {missing}"
