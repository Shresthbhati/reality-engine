"""/api/registration/*: stateless pose-graph + session-alignment refinement, mounted on the real application."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from apps.api.main import app
from tests.test_cross_session_pose_graph import GT, anchor

client = TestClient(app)          # no `with`: startup (DB) is not needed for stateless routes
IDENT = [0.0, 0.0, 0.0, 1.0]


def chain_graph(drift=0.0):
    poses = {f"n{k}": {"t": [k * 1.0 + drift * k, 0.0, 0.0], "q": IDENT} for k in range(4)}
    edges = [{"id": f"e{k}", "i": f"n{k}", "j": f"n{k + 1}", "t": [1.0, 0.0, 0.0], "q": IDENT,
              "sigma_t": 0.01, "sigma_r": 0.01} for k in range(3)]
    return poses, edges


def test_route_is_mounted_in_openapi():
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/registration/pose-graph" in paths and "/api/registration/session-alignments" in paths


def test_pose_graph_loop_closure_reduces_drift_and_keeps_initial():
    poses, edges = chain_graph(drift=0.05)             # initial guess has drifted
    edges.append({"id": "loop", "i": "n3", "j": "n0", "t": [-3.0, 0.0, 0.0], "q": IDENT,
                  "sigma_t": 0.01, "sigma_r": 0.01})
    r = client.post("/api/registration/pose-graph", json={"poses": poses, "edges": edges})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["converged"] and body["backend"] == "scipy-least-squares"
    assert body["initial"]["n3"]["t"][0] == poses["n3"]["t"][0]                 # raw input echoed unchanged
    assert abs(body["optimized"]["n3"]["t"][0] - 3.0) < abs(poses["n3"]["t"][0] - 3.0)


def test_pose_graph_errors_are_422_with_reason():
    poses, edges = chain_graph()
    r = client.post("/api/registration/pose-graph", json={"poses": poses, "edges": edges[:1]})
    assert r.status_code == 422 and "disconnected" in r.json()["detail"]
    bad = dict(edges[0], sigma_t=0.0)
    r = client.post("/api/registration/pose-graph", json={"poses": poses, "edges": [bad] + edges[1:]})
    assert r.status_code == 422 and "sigma" in r.json()["detail"]
    r = client.post("/api/registration/pose-graph", json={"poses": {"a": {"t": [0, 0], "q": IDENT}}})
    assert r.status_code == 422                                                  # schema: t needs 3 numbers


def _payload(alignment):
    q = alignment.transform.rotation
    t = alignment.transform.translation
    return {"from_session": alignment.from_session, "to_session": alignment.to_session, "rmse": alignment.rmse,
            "transform": {"rotation": [q.w, q.x, q.y, q.z], "translation": [t.x, t.y, t.z]}}


def test_session_alignments_refine_and_report():
    als = [anchor("B", "A", 0.01), anchor("C", "B", 0.01), anchor("C", "A", 0.4, (0.4, 0, 0))]
    r = client.post("/api/registration/session-alignments",
                    json={"sessions": ["A", "B", "C"], "alignments": [_payload(a) for a in als]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["refinement"]["status"] == "refined" and body["refinement"]["loop_closures"] == 1
    t = body["transforms"]["C"]["translation"]
    assert np.linalg.norm(np.array([t["x"], t["y"], t["z"]]) - GT["C"][1]) < 0.2        # BFS was 0.4 off


def test_session_alignments_reject_bad_transform_and_unknown_shape():
    a = _payload(anchor("B", "A", 0.01))
    a["transform"]["rotation"] = [2.0, 0.0, 0.0, 0.0]                                # not a unit quaternion
    r = client.post("/api/registration/session-alignments", json={"sessions": ["A", "B"], "alignments": [a]})
    assert r.status_code == 422
    r = client.post("/api/registration/session-alignments", json={"sessions": ["A"], "alignments": []})
    assert r.status_code == 422
