"""B15: 'where is X' resolves aliases, walks part_of/inside/on chains, and never matches unknown attributes."""
from types import SimpleNamespace as N

import pytest

pytest.importorskip("fastapi")
from apps.api.routes_query import find_entities  # noqa: E402


def _e(id, type, rels=(), attrs=None, labels=(), extra=None, matrix=None):
    ex = dict(extra or {})
    if attrs:
        ex["attributes"] = attrs
    t = N(matrix=matrix) if matrix else None
    return N(id=id, type=type, semantic_labels=list(labels), relationships=[N(kind=k, target_id=t_) for k, t_ in rels],
             extra_fields=ex, transform=t)


M = [[1, 0, 0, 2.0], [0, 1, 0, 3.0], [0, 0, 1, 0.5], [0, 0, 0, 1]]


def _world():
    es = [_e("room", "architecture/room"),
          _e("cab", "furniture/storage/wardrobe", [("inside", "room")], extra={"interior": "not_observed"}),
          _e("drawer2", "furniture/storage/drawer", [("part_of", "cab")],
             attrs={"contents": {"value": "snacks", "p": 1.0, "basis": "user_stated", "obs": []}}, matrix=M),
          _e("lock", "furniture/storage/lock", [("part_of", "cab")],
             attrs={"lock_type": {"value": None, "p": 0.3, "basis": "visual", "obs": []}})]
    return N(entities={e.id: e for e in es})


def test_user_stated_contents_are_found_with_chain_and_basis():
    (r,) = find_entities(_world(), "snacks")
    assert r["id"] == "drawer2" and r["basis"] == "user_stated" and r["position"] == [2.0, 3.0, 0.5]
    assert [c["id"] for c in r["chain"]] == ["cab", "room"]


def test_closed_interior_is_reported_not_observed():
    (r,) = find_entities(_world(), "wardrobe")
    assert r["id"] == "cab" and r["interior"] == "not_observed"


def test_unknown_attribute_never_matches():
    assert find_entities(_world(), "keypad") == []


def test_api_route_find_registered():
    from apps.api.main import app

    paths = [route.path for route in app.routes]
    assert "/api/worlds/{world_id}/find" in paths
    assert "/api/worlds/{world_id}/query/find" in paths
