"""Routes: pose-graph optimisation and cross-session alignment refinement.

Stateless: the caller supplies measurements, the server validates and optimises, nothing is persisted and no world
is modified (writing a registration into a world version is a separate, validated step). Invalid or unsolvable
graphs are a 422 with the optimiser's own reason -- never a silently "best-effort" answer.

Units: metres and radians; quaternions are [x, y, z, w] for /pose-graph (RigidTransform payloads below use w, x, y, z
to match the rest of the engine).
"""

from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine.math import Quat, Vec3
from registration.registration import RegistrationError, RigidTransform
from registration.cross_session import SessionAlignment, refine_session_chain
from registration.pose_graph import Edge, GnssPrior, Pose, PoseGraphError, optimize_pose_graph

registration = APIRouter(prefix="/api/registration", tags=["registration"])

MAX_NODES = 2000
MAX_EDGES = 20000


class PoseIn(BaseModel):
    t: List[float] = Field(min_length=3, max_length=3)
    q: List[float] = Field(min_length=4, max_length=4)


class EdgeIn(BaseModel):
    id: str = Field(min_length=1)
    i: str
    j: str
    t: List[float] = Field(min_length=3, max_length=3)
    q: List[float] = Field(min_length=4, max_length=4)
    sigma_t: float = 0.05
    sigma_r: float = 0.02


class GnssIn(BaseModel):
    node: str
    xyz: List[float] = Field(min_length=3, max_length=3)
    sigma: float = 1.0


class PoseGraphIn(BaseModel):
    poses: Dict[str, PoseIn]
    edges: List[EdgeIn] = Field(default_factory=list, max_length=MAX_EDGES)
    gnss: List[GnssIn] = Field(default_factory=list, max_length=MAX_NODES)


@registration.post("/pose-graph")
def solve_pose_graph(body: PoseGraphIn) -> dict:
    """Optimise an SE(3) pose graph. Returns optimised AND initial poses, costs, rejected/ambiguous constraints."""
    if len(body.poses) > MAX_NODES:
        raise HTTPException(413, f"too many poses ({len(body.poses)} > {MAX_NODES})")
    try:
        result = optimize_pose_graph(
            {k: Pose(tuple(v.t), tuple(v.q)) for k, v in body.poses.items()},
            [Edge(e.id, e.i, e.j, tuple(e.t), tuple(e.q), e.sigma_t, e.sigma_r) for e in body.edges],
            [GnssPrior(g.node, tuple(g.xyz), g.sigma) for g in body.gnss],
        )
    except PoseGraphError as exc:
        raise HTTPException(422, str(exc)) from exc
    return result.to_dict()


class TransformIn(BaseModel):
    """p_to = R(rotation) p_from + translation. Rotation is a unit quaternion (w, x, y, z)."""
    rotation: List[float] = Field(min_length=4, max_length=4)
    translation: List[float] = Field(min_length=3, max_length=3)


class AlignmentIn(BaseModel):
    from_session: str
    to_session: str
    transform: TransformIn
    rmse: float = Field(ge=0.0)
    method: str = "icp"


class SessionAlignmentsIn(BaseModel):
    #: first entry is the reference frame
    sessions: List[str] = Field(min_length=2, max_length=MAX_NODES)
    alignments: List[AlignmentIn] = Field(max_length=MAX_EDGES)
    characteristic_length_m: float = Field(default=10.0, gt=0.0)


@registration.post("/session-alignments")
def refine_alignments(body: SessionAlignmentsIn) -> dict:
    """Jointly refine pairwise session alignments (CrossSessionReport JSON, incl. the `refinement` record with the
    initial transforms, per-edge sigmas/chi2 and any rejected or ambiguous alignments)."""
    anchors = {}
    try:
        for a in body.alignments:
            w, x, y, z = a.transform.rotation
            tr = RigidTransform(from_frame=a.from_session, to_frame=a.to_session,
                                rotation=Quat(w=w, x=x, y=y, z=z), translation=Vec3(*a.transform.translation))
            anchors[(a.from_session, a.to_session)] = SessionAlignment(
                a.from_session, a.to_session, "accepted", a.method, tr, a.rmse, 1.0,
                "supplied by caller", 0, 0)
        report = refine_session_chain(anchors, body.sessions, body.characteristic_length_m)
    except (RegistrationError, PoseGraphError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return report.to_dict()
