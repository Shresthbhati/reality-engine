"""Photorealistic Gaussian Splatting (3DGS) pipeline for Reality Engine.

Provides ingestion, conversion, decimation, and binary serialization for
3D Gaussian Splat primitives (.splat and .ply formats).

Standard 32-byte .splat specification:
- 0..12:   float32 x, y, z (little-endian)
- 12..24:  float32 scale_x, scale_y, scale_z (little-endian)
- 24..28:  uint8 r, g, b, a (where a is opacity 0..255)
- 28..32:  uint8 qx, qy, qz, qw (normalized quaternion encoded as uint8(q * 128 + 128))

3DGS PLY format support:
- Handles Inria 3DGS PLYs with spherical harmonics DC components, log-scales,
  logit opacities, and quaternion orientations.
- Handles standard colored/scaled PLY clouds.
- Supports both binary little-endian and ASCII PLY encodings.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Any, Sequence

import numpy as np

# Spherical Harmonics degree 0 constant: 0.5 * sqrt(1 / pi)
SH_C0 = 0.28209479177387814


@dataclass(slots=True)
class GaussianPrimitive:
    """A single 3D Gaussian Splat primitive."""

    position: tuple[float, float, float]
    scale: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # (qx, qy, qz, qw)
    color: tuple[int, int, int]  # (r, g, b) in 0..255
    opacity: float  # in 0..1


class GaussianSplatCloud:
    """In-memory representation of a 3D Gaussian Splat model."""

    def __init__(
        self,
        positions: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
    ) -> None:
        """Initialize splat cloud from vectorized numpy arrays.

        Args:
            positions: (N, 3) float32 array of [x, y, z] in world coordinates.
            scales: (N, 3) float32 array of [sx, sy, sz] in linear scale units.
            rotations: (N, 4) float32 array of normalized quaternions [qx, qy, qz, qw].
            colors: (N, 3) uint8 array of [r, g, b] in 0..255.
            opacities: (N,) float32 array of opacity values in 0..1.
        """
        n = len(positions)
        self.positions = np.asarray(positions, dtype=np.float32).reshape((n, 3))
        self.scales = np.asarray(scales, dtype=np.float32).reshape((n, 3))
        self.rotations = np.asarray(rotations, dtype=np.float32).reshape((n, 4))
        self.colors = np.asarray(colors, dtype=np.uint8).reshape((n, 3))
        self.opacities = np.clip(np.asarray(opacities, dtype=np.float32).reshape((n,)), 0.0, 1.0)

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def count(self) -> int:
        return len(self.positions)

    @property
    def byte_size(self) -> int:
        """Size in standard 32-byte .splat format."""
        return len(self.positions) * 32

    @property
    def bounds(self) -> dict[str, list[float]]:
        """Axis-aligned bounding box of splat centroids."""
        if len(self.positions) == 0:
            return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
        min_pt = self.positions.min(axis=0).tolist()
        max_pt = self.positions.max(axis=0).tolist()
        return {
            "min": [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])],
            "max": [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])],
        }

    def to_primitives(self) -> list[GaussianPrimitive]:
        """Convert to a list of dataclass primitives."""
        primitives: list[GaussianPrimitive] = []
        for i in range(len(self.positions)):
            pos = (float(self.positions[i, 0]), float(self.positions[i, 1]), float(self.positions[i, 2]))
            s = (float(self.scales[i, 0]), float(self.scales[i, 1]), float(self.scales[i, 2]))
            r = (
                float(self.rotations[i, 0]),
                float(self.rotations[i, 1]),
                float(self.rotations[i, 2]),
                float(self.rotations[i, 3]),
            )
            c = (int(self.colors[i, 0]), int(self.colors[i, 1]), int(self.colors[i, 2]))
            o = float(self.opacities[i])
            primitives.append(GaussianPrimitive(position=pos, scale=s, rotation=r, color=c, opacity=o))
        return primitives

    @classmethod
    def from_primitives(cls, primitives: Sequence[GaussianPrimitive]) -> GaussianSplatCloud:
        """Create cloud from a sequence of GaussianPrimitive objects."""
        n = len(primitives)
        if n == 0:
            return cls(
                positions=np.zeros((0, 3), dtype=np.float32),
                scales=np.zeros((0, 3), dtype=np.float32),
                rotations=np.zeros((0, 4), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
                opacities=np.zeros((0,), dtype=np.float32),
            )

        positions = np.empty((n, 3), dtype=np.float32)
        scales = np.empty((n, 3), dtype=np.float32)
        rotations = np.empty((n, 4), dtype=np.float32)
        colors = np.empty((n, 3), dtype=np.uint8)
        opacities = np.empty((n,), dtype=np.float32)

        for i, p in enumerate(primitives):
            positions[i] = p.position
            scales[i] = p.scale
            rotations[i] = p.rotation
            colors[i] = p.color
            opacities[i] = p.opacity

        return cls(positions, scales, rotations, colors, opacities)

    def to_splat_binary(self) -> bytes:
        """Serialize cloud into standard 32-byte .splat binary format."""
        return export_splat_binary(self)

    def to_ply(self, binary: bool = True) -> bytes:
        """Serialize cloud into 3DGS PLY format."""
        return export_ply_splats(self, binary=binary)

    def decimate(self, target_count: int) -> GaussianSplatCloud:
        """Decimate splat cloud to target_count."""
        return decimate_splats(self, target_count=target_count)


def parse_splat_binary(data: bytes) -> GaussianSplatCloud:
    """Parse standard 32-byte chunk .splat binary buffer.

    Specification:
    - 0..12:  float32 x, y, z
    - 12..24: float32 sx, sy, sz
    - 24..28: uint8 r, g, b, a
    - 28..32: uint8 qx, qy, qz, qw (where q = (byte - 128) / 128.0)
    """
    total_bytes = len(data)
    if total_bytes == 0:
        return GaussianSplatCloud(
            positions=np.zeros((0, 3), dtype=np.float32),
            scales=np.zeros((0, 3), dtype=np.float32),
            rotations=np.zeros((0, 4), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
            opacities=np.zeros((0,), dtype=np.float32),
        )

    if total_bytes % 32 != 0:
        # Truncate to nearest valid 32-byte boundary if needed
        valid_bytes = (total_bytes // 32) * 32
        data = data[:valid_bytes]

    n = len(data) // 32
    if n == 0:
        return GaussianSplatCloud(
            positions=np.zeros((0, 3), dtype=np.float32),
            scales=np.zeros((0, 3), dtype=np.float32),
            rotations=np.zeros((0, 4), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
            opacities=np.zeros((0,), dtype=np.float32),
        )

    # Use structured numpy array for zero-copy parsing
    dtype = np.dtype(
        [
            ("position", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("color", "u1", (4,)),  # r, g, b, a
            ("rotation", "u1", (4,)),  # qx, qy, qz, qw encoded
        ]
    )
    arr = np.frombuffer(data, dtype=dtype, count=n)

    positions = arr["position"].copy()
    scales = arr["scale"].copy()

    # Extract RGB and alpha
    rgba = arr["color"]
    colors = rgba[:, :3].copy()
    opacities = (rgba[:, 3].astype(np.float32) / 255.0).copy()

    # Decode quaternions: q = (encoded - 128) / 128.0
    rot_raw = arr["rotation"].astype(np.float32)
    rotations = (rot_raw - 128.0) / 128.0
    # Normalize quaternions
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    rotations /= norms

    return GaussianSplatCloud(
        positions=positions,
        scales=scales,
        rotations=rotations,
        colors=colors,
        opacities=opacities,
    )


def export_splat_binary(cloud: GaussianSplatCloud) -> bytes:
    """Serialize GaussianSplatCloud into standard 32-byte chunk .splat format."""
    n = len(cloud)
    if n == 0:
        return b""

    dtype = np.dtype(
        [
            ("position", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("color", "u1", (4,)),
            ("rotation", "u1", (4,)),
        ]
    )
    arr = np.empty(n, dtype=dtype)
    arr["position"] = cloud.positions
    arr["scale"] = cloud.scales

    # Encode colors & opacity
    rgba = np.empty((n, 4), dtype=np.uint8)
    rgba[:, :3] = cloud.colors
    rgba[:, 3] = np.clip(cloud.opacities * 255.0, 0, 255).astype(np.uint8)
    arr["color"] = rgba

    # Encode quaternions: uint8(clip(q * 128.0 + 128.0, 0, 255))
    rot_norm = cloud.rotations.copy()
    norms = np.linalg.norm(rot_norm, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    rot_norm /= norms
    arr["rotation"] = np.clip(rot_norm * 128.0 + 128.0, 0, 255).astype(np.uint8)

    return arr.tobytes()


def _ply_type_to_numpy(t: str) -> str:
    """Map PLY property type string to numpy type code."""
    mapping = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    return mapping.get(t.lower(), "<f4")


def parse_ply_splats(data: bytes) -> GaussianSplatCloud:
    """Parse 3D Gaussian Splats from a PLY file (binary or ASCII)."""
    if not data.startswith(b"ply"):
        raise ValueError("Invalid PLY data: missing 'ply' signature")

    head_end = data.find(b"end_header")
    if head_end == -1:
        raise ValueError("Invalid PLY data: missing 'end_header'")

    header_bytes = data[:head_end]
    body_bytes = data[head_end + len(b"end_header") :].lstrip(b"\r\n")
    header_lines = header_bytes.decode("ascii", errors="replace").splitlines()

    is_binary = False
    is_ascii = False
    vertex_count = 0
    properties: list[tuple[str, str]] = []  # (name, type)

    in_vertex = False
    for line in header_lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        if tokens[0] == "format":
            if "binary_little_endian" in tokens[1]:
                is_binary = True
            elif "ascii" in tokens[1]:
                is_ascii = True
            else:
                raise ValueError(f"Unsupported PLY format: {tokens[1]}")
        elif tokens[0] == "element":
            if tokens[1] == "vertex":
                in_vertex = True
                vertex_count = int(tokens[2])
            else:
                in_vertex = False
        elif tokens[0] == "property" and in_vertex:
            # e.g. property float x
            prop_type = tokens[1]
            prop_name = tokens[2]
            properties.append((prop_name, prop_type))

    if vertex_count == 0 or not properties:
        return GaussianSplatCloud(
            positions=np.zeros((0, 3), dtype=np.float32),
            scales=np.zeros((0, 3), dtype=np.float32),
            rotations=np.zeros((0, 4), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
            opacities=np.zeros((0,), dtype=np.float32),
        )

    prop_names = [p[0] for p in properties]
    prop_dict = {p[0]: idx for idx, p in enumerate(properties)}

    if is_binary:
        # Parse binary little endian
        dtype_fields = [(p[0], _ply_type_to_numpy(p[1])) for p in properties]
        arr_dtype = np.dtype(dtype_fields)
        record_size = arr_dtype.itemsize
        expected_bytes = record_size * vertex_count

        actual_body = body_bytes[:expected_bytes]
        parsed_records = len(actual_body) // record_size
        arr = np.frombuffer(actual_body[: parsed_records * record_size], dtype=arr_dtype)
        n = len(arr)

        # Positions (x, y, z)
        positions = np.zeros((n, 3), dtype=np.float32)
        for i, coord in enumerate(["x", "y", "z"]):
            if coord in prop_dict:
                positions[:, i] = arr[coord].astype(np.float32)

        # Scales (scale_0, scale_1, scale_2 or sx, sy, sz)
        scales = np.zeros((n, 3), dtype=np.float32)
        has_log_scale = False
        if all(k in prop_dict for k in ["scale_0", "scale_1", "scale_2"]):
            for i, k in enumerate(["scale_0", "scale_1", "scale_2"]):
                scales[:, i] = arr[k].astype(np.float32)
            has_log_scale = True
        elif all(k in prop_dict for k in ["sx", "sy", "sz"]):
            for i, k in enumerate(["sx", "sy", "sz"]):
                scales[:, i] = arr[k].astype(np.float32)
        else:
            # Default isotropic radius if scales are absent
            scales.fill(0.03)

        if has_log_scale:
            # Inria 3DGS stores log-scales: exp(scale)
            # Clip exponent to prevent overflow
            scales = np.exp(np.clip(scales, -20.0, 10.0))

        # Rotations: rot_0, rot_1, rot_2, rot_3
        # Inria PLY layout: rot_0 = qw, rot_1 = qx, rot_2 = qy, rot_3 = qz
        rotations = np.zeros((n, 4), dtype=np.float32)
        if all(k in prop_dict for k in ["rot_0", "rot_1", "rot_2", "rot_3"]):
            rotations[:, 0] = arr["rot_1"].astype(np.float32)  # qx
            rotations[:, 1] = arr["rot_2"].astype(np.float32)  # qy
            rotations[:, 2] = arr["rot_3"].astype(np.float32)  # qz
            rotations[:, 3] = arr["rot_0"].astype(np.float32)  # qw
        elif all(k in prop_dict for k in ["qx", "qy", "qz", "qw"]):
            rotations[:, 0] = arr["qx"].astype(np.float32)
            rotations[:, 1] = arr["qy"].astype(np.float32)
            rotations[:, 2] = arr["qz"].astype(np.float32)
            rotations[:, 3] = arr["qw"].astype(np.float32)
        else:
            rotations[:, 3] = 1.0  # Identity quaternion (0, 0, 0, 1)

        # Normalize rotations
        norms = np.linalg.norm(rotations, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        rotations /= norms

        # Colors (RGB)
        colors = np.zeros((n, 3), dtype=np.uint8)
        if all(k in prop_dict for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
            # Spherical harmonics DC components: RGB = clip((0.5 + SH_C0 * f_dc) * 255)
            f_dc = np.column_stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]]).astype(np.float32)
            rgb_f = (0.5 + SH_C0 * f_dc) * 255.0
            colors = np.clip(rgb_f, 0, 255).astype(np.uint8)
        elif all(k in prop_dict for k in ["red", "green", "blue"]):
            for i, k in enumerate(["red", "green", "blue"]):
                c_val = arr[k]
                if np.issubdtype(c_val.dtype, np.floating) and c_val.max() <= 1.0:
                    colors[:, i] = np.clip(c_val * 255.0, 0, 255).astype(np.uint8)
                else:
                    colors[:, i] = np.clip(c_val, 0, 255).astype(np.uint8)
        else:
            colors.fill(200)

        # Opacities
        opacities = np.ones((n,), dtype=np.float32)
        if "opacity" in prop_dict:
            raw_op = arr["opacity"].astype(np.float32)
            # Inria 3DGS stores logit: 1 / (1 + exp(-opacity))
            if raw_op.min() < 0.0 or raw_op.max() > 1.0:
                opacities = 1.0 / (1.0 + np.exp(-np.clip(raw_op, -20.0, 20.0)))
            else:
                opacities = np.clip(raw_op, 0.0, 1.0)
        elif "alpha" in prop_dict:
            a_val = arr["alpha"]
            if np.issubdtype(a_val.dtype, np.floating) and a_val.max() <= 1.0:
                opacities = np.clip(a_val, 0.0, 1.0)
            else:
                opacities = np.clip(a_val.astype(np.float32) / 255.0, 0.0, 1.0)

        return GaussianSplatCloud(
            positions=positions,
            scales=scales,
            rotations=rotations,
            colors=colors,
            opacities=opacities,
        )

    else:
        # ASCII parsing
        lines = body_bytes.decode("ascii", errors="replace").splitlines()
        valid_lines = [ln.strip() for ln in lines if ln.strip()][:vertex_count]
        n = len(valid_lines)

        positions = np.zeros((n, 3), dtype=np.float32)
        scales = np.full((n, 3), 0.03, dtype=np.float32)
        rotations = np.zeros((n, 4), dtype=np.float32)
        rotations[:, 3] = 1.0
        colors = np.full((n, 3), 200, dtype=np.uint8)
        opacities = np.ones((n,), dtype=np.float32)

        has_sh = all(k in prop_dict for k in ["f_dc_0", "f_dc_1", "f_dc_2"])
        has_rgb = all(k in prop_dict for k in ["red", "green", "blue"])
        has_rot = all(k in prop_dict for k in ["rot_0", "rot_1", "rot_2", "rot_3"])
        has_scales = all(k in prop_dict for k in ["scale_0", "scale_1", "scale_2"])

        for row_idx, line in enumerate(valid_lines):
            vals = [float(v) for v in line.split()]
            if len(vals) < len(properties):
                continue
            if "x" in prop_dict and "y" in prop_dict and "z" in prop_dict:
                positions[row_idx, 0] = vals[prop_dict["x"]]
                positions[row_idx, 1] = vals[prop_dict["y"]]
                positions[row_idx, 2] = vals[prop_dict["z"]]

            if has_scales:
                s0 = vals[prop_dict["scale_0"]]
                s1 = vals[prop_dict["scale_1"]]
                s2 = vals[prop_dict["scale_2"]]
                scales[row_idx, 0] = math.exp(max(-20.0, min(10.0, s0)))
                scales[row_idx, 1] = math.exp(max(-20.0, min(10.0, s1)))
                scales[row_idx, 2] = math.exp(max(-20.0, min(10.0, s2)))

            if has_rot:
                qw = vals[prop_dict["rot_0"]]
                qx = vals[prop_dict["rot_1"]]
                qy = vals[prop_dict["rot_2"]]
                qz = vals[prop_dict["rot_3"]]
                norm = math.hypot(qx, qy, qz, qw) or 1.0
                rotations[row_idx, 0] = qx / norm
                rotations[row_idx, 1] = qy / norm
                rotations[row_idx, 2] = qz / norm
                rotations[row_idx, 3] = qw / norm

            if has_sh:
                r = int(max(0, min(255, (0.5 + SH_C0 * vals[prop_dict["f_dc_0"]]) * 255.0)))
                g = int(max(0, min(255, (0.5 + SH_C0 * vals[prop_dict["f_dc_1"]]) * 255.0)))
                b = int(max(0, min(255, (0.5 + SH_C0 * vals[prop_dict["f_dc_2"]]) * 255.0)))
                colors[row_idx] = [r, g, b]
            elif has_rgb:
                colors[row_idx] = [
                    int(vals[prop_dict["red"]]),
                    int(vals[prop_dict["green"]]),
                    int(vals[prop_dict["blue"]]),
                ]

            if "opacity" in prop_dict:
                op_val = vals[prop_dict["opacity"]]
                if op_val < 0.0 or op_val > 1.0:
                    opacities[row_idx] = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, op_val))))
                else:
                    opacities[row_idx] = op_val

        return GaussianSplatCloud(
            positions=positions,
            scales=scales,
            rotations=rotations,
            colors=colors,
            opacities=opacities,
        )


def export_ply_splats(cloud: GaussianSplatCloud, binary: bool = True) -> bytes:
    """Serialize GaussianSplatCloud to 3DGS-compatible PLY format.

    Exports standard Inria 3DGS properties:
    - x, y, z
    - nx, ny, nz
    - f_dc_0, f_dc_1, f_dc_2 (spherical harmonics DC color)
    - opacity (logit)
    - scale_0, scale_1, scale_2 (log-scale)
    - rot_0, rot_1, rot_2, rot_3 (qw, qx, qy, qz)
    """
    n = len(cloud)

    # Convert RGB 0..255 back to SH DC: f_dc = (RGB / 255 - 0.5) / SH_C0
    rgb_norm = cloud.colors.astype(np.float32) / 255.0
    f_dc = (rgb_norm - 0.5) / SH_C0

    # Convert opacity to logit: log(p / (1 - p))
    p = np.clip(cloud.opacities, 1e-4, 1.0 - 1e-4)
    logit_opacity = np.log(p / (1.0 - p))

    # Convert scale to log-scale
    scales = np.clip(cloud.scales, 1e-6, 1e3)
    log_scales = np.log(scales)

    # Rotations: Inria layout (rot_0=qw, rot_1=qx, rot_2=qy, rot_3=qz)
    rot_qw = cloud.rotations[:, 3]
    rot_qx = cloud.rotations[:, 0]
    rot_qy = cloud.rotations[:, 1]
    rot_qz = cloud.rotations[:, 2]

    header = (
        "ply\n"
        f"format {'binary_little_endian 1.0' if binary else 'ascii 1.0'}\n"
        "comment reality-engine 3dgs export\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    ).encode("ascii")

    if binary:
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("nx", "<f4"),
                ("ny", "<f4"),
                ("nz", "<f4"),
                ("f_dc_0", "<f4"),
                ("f_dc_1", "<f4"),
                ("f_dc_2", "<f4"),
                ("opacity", "<f4"),
                ("scale_0", "<f4"),
                ("scale_1", "<f4"),
                ("scale_2", "<f4"),
                ("rot_0", "<f4"),
                ("rot_1", "<f4"),
                ("rot_2", "<f4"),
                ("rot_3", "<f4"),
            ]
        )
        arr = np.empty(n, dtype=dtype)
        arr["x"] = cloud.positions[:, 0]
        arr["y"] = cloud.positions[:, 1]
        arr["z"] = cloud.positions[:, 2]
        arr["nx"] = 0.0
        arr["ny"] = 0.0
        arr["nz"] = 0.0
        arr["f_dc_0"] = f_dc[:, 0]
        arr["f_dc_1"] = f_dc[:, 1]
        arr["f_dc_2"] = f_dc[:, 2]
        arr["opacity"] = logit_opacity
        arr["scale_0"] = log_scales[:, 0]
        arr["scale_1"] = log_scales[:, 1]
        arr["scale_2"] = log_scales[:, 2]
        arr["rot_0"] = rot_qw
        arr["rot_1"] = rot_qx
        arr["rot_2"] = rot_qy
        arr["rot_3"] = rot_qz
        return header + arr.tobytes()
    else:
        body_lines: list[str] = []
        for i in range(n):
            body_lines.append(
                f"{cloud.positions[i, 0]:.6f} {cloud.positions[i, 1]:.6f} {cloud.positions[i, 2]:.6f} "
                f"0.0 0.0 0.0 "
                f"{f_dc[i, 0]:.6f} {f_dc[i, 1]:.6f} {f_dc[i, 2]:.6f} "
                f"{logit_opacity[i]:.6f} "
                f"{log_scales[i, 0]:.6f} {log_scales[i, 1]:.6f} {log_scales[i, 2]:.6f} "
                f"{rot_qw[i]:.6f} {rot_qx[i]:.6f} {rot_qy[i]:.6f} {rot_qz[i]:.6f}\n"
            )
        return header + "".join(body_lines).encode("ascii")


def generate_splats_from_points(
    positions: np.ndarray | Sequence[Sequence[float]],
    colors: np.ndarray | Sequence[Sequence[int]] | None = None,
    normals: np.ndarray | Sequence[Sequence[float]] | None = None,
    default_radius: float = 0.03,
) -> GaussianSplatCloud:
    """Initialize a Gaussian Splat cloud from 3D point positions.

    Allows converting existing reconstruction point clouds or mesh vertices into
    renderable 3DGS primitives with proper scales, orientations, and colors.
    """
    pos_arr = np.asarray(positions, dtype=np.float32)
    if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos_arr.shape}")

    n = len(pos_arr)
    if n == 0:
        return GaussianSplatCloud(
            positions=np.zeros((0, 3), dtype=np.float32),
            scales=np.zeros((0, 3), dtype=np.float32),
            rotations=np.zeros((0, 4), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
            opacities=np.zeros((0,), dtype=np.float32),
        )

    # Colors
    if colors is not None:
        c_arr = np.asarray(colors)
        if c_arr.shape[0] != n:
            raise ValueError(f"colors count ({len(c_arr)}) != positions count ({n})")
        if np.issubdtype(c_arr.dtype, np.floating) and c_arr.max() <= 1.0:
            c_arr = (c_arr * 255.0).astype(np.uint8)
        else:
            c_arr = c_arr.astype(np.uint8)
    else:
        # Default warm studio architectural tone
        c_arr = np.full((n, 3), [195, 205, 215], dtype=np.uint8)

    # Scales & Rotations
    scales = np.full((n, 3), default_radius, dtype=np.float32)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 3] = 1.0  # Identity (qx=0, qy=0, qz=0, qw=1)

    if normals is not None:
        norm_arr = np.asarray(normals, dtype=np.float32)
        if norm_arr.shape == (n, 3):
            # Disc-like splat: scale along normal is smaller (tangent disc)
            scales[:, 2] = default_radius * 0.25
            scales[:, 0] = default_radius * 1.2
            scales[:, 1] = default_radius * 1.2

            # Compute rotation from +Z (0, 0, 1) to normal vector n
            # Using half-way quaternion formula
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            for i in range(n):
                n_vec = norm_arr[i]
                n_len = float(np.linalg.norm(n_vec))
                if n_len > 1e-4:
                    n_vec = n_vec / n_len
                    # Cross product: axis = z_axis x n_vec
                    axis = np.cross(z_axis, n_vec)
                    dot = float(np.dot(z_axis, n_vec))
                    if dot < -0.9999:
                        rotations[i] = [1.0, 0.0, 0.0, 0.0]
                    else:
                        w = 1.0 + dot
                        q = np.array([axis[0], axis[1], axis[2], w], dtype=np.float32)
                        q /= np.linalg.norm(q)
                        rotations[i] = q

    opacities = np.full((n,), 0.95, dtype=np.float32)

    return GaussianSplatCloud(
        positions=pos_arr,
        scales=scales,
        rotations=rotations,
        colors=c_arr,
        opacities=opacities,
    )


def decimate_splats(cloud: GaussianSplatCloud, target_count: int) -> GaussianSplatCloud:
    """Downsample a GaussianSplatCloud to target_count using spatial uniform sampling.

    Preserves bounds and distribution of splats for level-of-detail (LOD) streaming.
    """
    n = len(cloud)
    if target_count <= 0 or n == 0:
        return GaussianSplatCloud(
            positions=np.zeros((0, 3), dtype=np.float32),
            scales=np.zeros((0, 3), dtype=np.float32),
            rotations=np.zeros((0, 4), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
            opacities=np.zeros((0,), dtype=np.float32),
        )

    if n <= target_count:
        return cloud

    # Strided uniform downsample preserving spatial ordering
    step = n / float(target_count)
    indices = np.clip((np.arange(target_count) * step).astype(np.int64), 0, n - 1)

    return GaussianSplatCloud(
        positions=cloud.positions[indices],
        scales=cloud.scales[indices] * math.sqrt(n / float(target_count)),  # slightly increase splat radius to compensate for density loss
        rotations=cloud.rotations[indices],
        colors=cloud.colors[indices],
        opacities=cloud.opacities[indices],
    )
