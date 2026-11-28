"""Surface reconstruction backend (P0.11): oriented points -> triangle mesh.

Mature backend, not a research rewrite: the configured COLMAP binary's
CPU `poisson_mesher` (the screened Poisson implementation bundled with
COLMAP) over a binary PLY of oriented points. Availability is a
capability probe (the command actually listed under `colmap help`),
never the mere existence of the binary -- a COLMAP build without the
mesher must be reported, not discovered at mesh time.

Failure states are explicit: MeshingUnavailableError (dependency) vs
MeshingError (the backend ran and refused/failed). A fake "box mesh"
fallback is deliberately NOT provided -- a fallback that fabricates
geometry is worse than an honest absence.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .mesh import MeshData
from .preprocess import PreprocessError

Vert = Tuple[float, float, float]


class MeshingUnavailableError(RuntimeError):
    """The configured binary cannot perform this stage (missing binary or
    no poisson_mesher command) -- the caller should skip with a note."""


class MeshingError(RuntimeError):
    """The backend ran but failed on this input (non-zero exit or
    unreadable/empty output)."""


def _find_colmap(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        raise MeshingUnavailableError(
            f"COLMAP binary {binary!r} not found on PATH -- surface "
            "reconstruction unavailable (set colmap_binary or install COLMAP)"
        )
    return resolved


def _probe_poisson(colmap_path: str) -> bool:
    """True only if `colmap help` actually lists poisson_mesher. Presence
    of the binary alone proves nothing about this build's capabilities."""
    try:
        proc = subprocess.run(
            [colmap_path, "help"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "poisson_mesher" in (proc.stdout or "")


def poisson_mesher_available(binary: str = "colmap") -> bool:
    try:
        return _probe_poisson(_find_colmap(binary))
    except MeshingUnavailableError:
        return False


def reconstruct_surface(
    points: Sequence[Vert],
    normals: Sequence[Vert],
    colmap_binary: str = "colmap",
    depth: int = 10,
    trim: float = 0.0,
    num_threads: int = -1,
    work_dir: Optional[Path] = None,
) -> MeshData:
    """Oriented point cloud -> COLMAP screened-Poisson triangle mesh.

    Determinism note: Poisson's octree scheduling is multi-threaded, so
    byte-identical output is NOT guaranteed across runs; the mesh is
    content-hashed at persistence time, and the deterministic
    preprocessing upstream guarantees the same *input*. `depth` controls
    octree resolution.

    Observed COLMAP 4.2 behavior (this backend adapts to both):
      - Poisson's solver depth is bounded by point density; a sparse
        cloud reports "Solver depth should not exceed maximum depth:
        D <= M" and meshes nothing at D > M. When the first attempt
        comes back empty and that warning is present, the run retries
        once at the observed maximum M.
      - trim > 0 culls vertices whose (density-scaled) sample weight is
        below it; COLMAP's dense-pipeline default (10) empties sparse
        clouds entirely. Default here is 0 -- Poisson already produces
        closed surfaces; trimming is an optional post-cull.

    Raises MeshingUnavailableError (probe failed), PreprocessError-shape
    ValueErrors from MeshData validation on degenerate output, and
    MeshingError when the backend fails on valid input.
    """
    if not points:
        raise MeshingError("no points -- nothing to mesh")
    if len(points) != len(normals):
        raise MeshingError(
            f"points ({len(points)}) and normals ({len(normals)}) must be 1:1 -- "
            "an unoriented cloud cannot be Poisson-meshed honestly"
        )

    colmap_path = _find_colmap(colmap_binary)
    if not _probe_poisson(colmap_path):
        raise MeshingUnavailableError(
            f"{colmap_path} has no poisson_mesher command -- this COLMAP "
            "build cannot do surface reconstruction"
        )

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "element face 0\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    body = bytearray()
    for p, n in zip(points, normals):
        body += struct.pack("<6f", p[0], p[1], p[2], n[0], n[1], n[2])

    with tempfile.TemporaryDirectory(prefix="re_mesh_", dir=work_dir) as td:
        tdp = Path(td)
        in_ply = tdp / "oriented_points.ply"
        out_ply = tdp / "mesh.ply"
        in_ply.write_bytes(header + bytes(body))

        def _attempt(depth_value: int):
            cmd = [
                colmap_path, "poisson_mesher",
                "--input_path", str(in_ply),
                "--output_path", str(out_ply),
                "--PoissonMeshing.depth", str(depth_value),
                "--PoissonMeshing.trim", str(trim),
                "--PoissonMeshing.num_threads", str(num_threads),
                "--log_target", "stderr",
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            except subprocess.TimeoutExpired as exc:
                raise MeshingError(f"poisson_mesher timed out after 1800s: {exc}") from exc
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-800:]
                raise MeshingError(f"poisson_mesher exited {proc.returncode}: ...{tail}")
            if not out_ply.exists():
                raise MeshingError(
                    "poisson_mesher reported success but wrote no output mesh; "
                    f"log tail: {(proc.stderr or proc.stdout or '')[-400:]}"
                )
            return MeshData.from_ply_bytes(out_ply.read_bytes()), (proc.stderr or proc.stdout or "")

        mesh, log = _attempt(depth)
        if not mesh.faces and depth > 0:
            # Density-bounded solver: retry once at the observed maximum.
            import re

            match = re.search(
                r"Solver depth should not exceed maximum depth:\s*\d+\s*<=\s*(\d+)",
                log,
            )
            if match:
                max_depth = int(match.group(1))
                if 0 < max_depth < depth:
                    mesh, log = _attempt(max_depth)
        if not mesh.faces:
            raise MeshingError(
                f"poisson_mesher meshed nothing at depth {depth} "
                f"({len(mesh.vertices)} vertices) -- input cloud likely too "
                "sparse/noisy; log tail: " + log[-400:]
            )
        return mesh
