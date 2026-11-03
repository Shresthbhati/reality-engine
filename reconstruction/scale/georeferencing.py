"""Georeferencing and Global Coordinate Fusion Engine.

Implements rigorous geodetic transformation and 7-DOF Sim(3) alignment
between local SfM/reconstruction coordinates and Earth-fixed geodetic frames:
1. WGS84 Geodetic coordinates (lat, lon, height above ellipsoid) <-> ECEF (Earth-Centered Earth-Fixed).
2. ECEF <-> Local ENU (East, North, Up) tangent plane coordinates relative to an explicit reference point.
3. Closed-form Umeyama 7-DOF Sim(3) alignment solver (scale s, rotation R in SO(3), translation t in R^3)
   with optional RANSAC outlier filtering to reject GNSS multipath spikes.
4. Metric precision assessment (horizontal RMSE, vertical RMSE, 3D RMSE).
5. Grounding into canonical WorldIR coordinate frames (Frame.ENU / Frame.WORLD) and Transform edges.

Never hallucinates coordinate conversions: all transforms store explicit reference origins and RMSE bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from provenance import Provenance, Uncertainty
from world_ir.coordinates import Frame, Transform
from world_ir.world_v1 import WorldIR


# WGS-84 Ellipsoid Geodetic Constants
WGS84_A = 6378137.0  # semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis
WGS84_E_SQ = 2.0 * WGS84_F - WGS84_F**2  # first eccentricity squared
WGS84_E_PRIME_SQ = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)  # second eccentricity squared


@dataclass(frozen=True)
class GeodeticPoint:
    """WGS84 geodetic coordinates (latitude, longitude, altitude/ellipsoidal height)."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.latitude_deg, self.longitude_deg, self.altitude_m)

    def to_dict(self) -> Dict[str, float]:
        return {
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "altitude_m": self.altitude_m,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GeodeticPoint:
        return cls(
            latitude_deg=float(data["latitude_deg"]),
            longitude_deg=float(data["longitude_deg"]),
            altitude_m=float(data.get("altitude_m", 0.0)),
        )


def lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert WGS84 (lat, lon, alt) to Earth-Centered, Earth-Fixed (ECEF) XYZ in meters."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)

    # Radius of curvature in prime vertical
    n = WGS84_A / math.sqrt(1.0 - WGS84_E_SQ * sin_phi**2)

    x = (n + alt_m) * cos_phi * cos_lam
    y = (n + alt_m) * cos_phi * sin_lam
    z = (n * (1.0 - WGS84_E_SQ) + alt_m) * sin_phi

    return np.array([x, y, z], dtype=np.float64)


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ECEF XYZ (meters) back to WGS84 (lat_deg, lon_deg, alt_m) using Bowring's method."""
    p = math.sqrt(x**2 + y**2)
    if p < 1e-6:
        # Near poles
        lat = 90.0 if z > 0 else -90.0
        lon = 0.0
        alt = abs(z) - WGS84_B
        return lat, lon, alt

    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    phi = math.atan2(
        z + WGS84_E_PRIME_SQ * WGS84_B * (math.sin(theta) ** 3),
        p - WGS84_E_SQ * WGS84_A * (math.cos(theta) ** 3),
    )
    lam = math.atan2(y, x)

    sin_phi = math.sin(phi)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E_SQ * sin_phi**2)
    alt = p / math.cos(phi) - n

    return math.degrees(phi), math.degrees(lam), alt


def lla_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> np.ndarray:
    """Convert geodetic (lat, lon, alt) to local East-North-Up (ENU) tangent coordinates relative to a reference origin."""
    pos_ecef = lla_to_ecef(lat_deg, lon_deg, alt_m)
    ref_ecef = lla_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    diff = pos_ecef - ref_ecef

    phi = math.radians(ref_lat_deg)
    lam = math.radians(ref_lon_deg)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)

    # Rotation matrix from ECEF to local ENU
    # [ -sin_lam,            cos_lam,           0       ]
    # [ -sin_phi*cos_lam,   -sin_phi*sin_lam,   cos_phi ]
    # [  cos_phi*cos_lam,    cos_phi*sin_lam,   sin_phi ]
    r_ecef_to_enu = np.array(
        [
            [-sin_lam, cos_lam, 0.0],
            [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
            [cos_phi * cos_lam, cos_phi * sin_lam, sin_phi],
        ],
        dtype=np.float64,
    )

    return r_ecef_to_enu @ diff


def enu_to_lla(
    e: float,
    n: float,
    u: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> Tuple[float, float, float]:
    """Convert local ENU coordinates back to geodetic WGS84 (lat, lon, alt) relative to the reference origin."""
    phi = math.radians(ref_lat_deg)
    lam = math.radians(ref_lon_deg)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)

    r_enu_to_ecef = np.array(
        [
            [-sin_lam, -sin_phi * cos_lam, cos_phi * cos_lam],
            [cos_lam, -sin_phi * sin_lam, cos_phi * sin_lam],
            [0.0, cos_phi, sin_phi],
        ],
        dtype=np.float64,
    )

    enu_vec = np.array([e, n, u], dtype=np.float64)
    ref_ecef = lla_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    pos_ecef = ref_ecef + (r_enu_to_ecef @ enu_vec)

    return ecef_to_lla(pos_ecef[0], pos_ecef[1], pos_ecef[2])


def solve_umeyama_sim3(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    with_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Computes similarity transform (s, R, t) that maps src_points into dst_points:

        dst ~ s * R @ src + t

    Args:
        src_points: (N, 3) coordinates in source space.
        dst_points: (N, 3) coordinates in target space.
        with_scale: If True, computes optimal scale s; if False, forces s = 1.0 (rigid).

    Returns:
        (scale, rotation_3x3, translation_3)
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        raise ValueError(f"Umeyama requires at least 3 point correspondences; got {len(src_points)}")

    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)

    n = src.shape[0]
    dim = src.shape[1]
    if dim != 3:
        raise ValueError("Points must have dimension 3")

    # Center centroids
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    # Variance of src
    var_src = np.sum(src_centered**2) / n
    if var_src < 1e-12:
        raise ValueError("Degenerate source point distribution: near-zero variance")

    # Cross-covariance matrix
    cov = (dst_centered.T @ src_centered) / n

    # SVD of covariance
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(dim)

    # Orientation reflection correction
    det_u_vt = np.linalg.det(u) * np.linalg.det(vt)
    if det_u_vt < 0.0:
        s_mat[dim - 1, dim - 1] = -1.0

    r = u @ s_mat @ vt

    if with_scale:
        scale = float(np.trace(np.diag(d) @ s_mat) / var_src)
    else:
        scale = 1.0

    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


@dataclass
class GeoreferenceDiagnostics:
    """Quantitative quality metrics for georeferencing registration."""

    rmse_horizontal_m: float
    rmse_vertical_m: float
    rmse_total_3d_m: float
    max_residual_m: float
    inlier_count: int
    total_correspondences: int
    scale_factor: float
    residual_spread_p95_m: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rmse_horizontal_m": round(self.rmse_horizontal_m, 4),
            "rmse_vertical_m": round(self.rmse_vertical_m, 4),
            "rmse_total_3d_m": round(self.rmse_total_3d_m, 4),
            "max_residual_m": round(self.max_residual_m, 4),
            "inlier_count": self.inlier_count,
            "total_correspondences": self.total_correspondences,
            "scale_factor": round(self.scale_factor, 6),
            "residual_spread_p95_m": round(self.residual_spread_p95_m, 4),
        }


@dataclass
class GeoreferenceResult:
    """Complete georeferencing solution."""

    success: bool
    reference_point: GeodeticPoint
    transform_4x4: List[List[float]]
    diagnostics: GeoreferenceDiagnostics
    inlier_indices: List[int]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "reference_point": self.reference_point.to_dict(),
            "transform_4x4": self.transform_4x4,
            "diagnostics": self.diagnostics.to_dict(),
            "inlier_count": len(self.inlier_indices),
            "notes": list(self.notes),
        }


def compute_residuals(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Computes Euclidean residual vectors (dst - (s*R*src + t)) for all points."""
    transformed = scale * (src_points @ rotation.T) + translation
    return dst_points - transformed


def align_georeference_ransac(
    src_local_pts: np.ndarray,
    dst_enu_pts: np.ndarray,
    ref_origin: GeodeticPoint,
    max_inlier_residual_m: float = 2.5,
    max_iterations: int = 100,
    with_scale: bool = True,
) -> GeoreferenceResult:
    """Robust RANSAC-based 7-DOF Sim(3) georeferencing solver.

    Handles realistic GNSS multipath outliers while aligning local camera centers to ENU.
    """
    n = len(src_local_pts)
    if n < 3:
        raise ValueError(f"Georeferencing requires at least 3 correspondences; got {n}")

    best_inliers: List[int] = []
    best_scale = 1.0
    best_rot = np.eye(3)
    best_trans = np.zeros(3)

    if n == 3:
        # Exact solution without RANSAC
        s, r, t = solve_umeyama_sim3(src_local_pts, dst_enu_pts, with_scale=with_scale)
        best_inliers = list(range(n))
        best_scale, best_rot, best_trans = s, r, t
    else:
        rng = np.random.RandomState(42)  # Deterministic seed
        for _ in range(max_iterations):
            sample_idx = rng.choice(n, size=3, replace=False)
            try:
                s, r, t = solve_umeyama_sim3(
                    src_local_pts[sample_idx], dst_enu_pts[sample_idx], with_scale=with_scale
                )
            except ValueError:
                continue

            residuals = compute_residuals(src_local_pts, dst_enu_pts, s, r, t)
            errors = np.linalg.norm(residuals, axis=1)
            inliers = [int(i) for i in np.where(errors <= max_inlier_residual_m)[0]]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_scale, best_rot, best_trans = s, r, t
                if len(best_inliers) == n:
                    break

        if len(best_inliers) >= 3:
            # Re-estimate on all inliers
            inlier_src = src_local_pts[best_inliers]
            inlier_dst = dst_enu_pts[best_inliers]
            best_scale, best_rot, best_trans = solve_umeyama_sim3(
                inlier_src, inlier_dst, with_scale=with_scale
            )

    # Compute diagnostics over inliers
    inlier_src = src_local_pts[best_inliers]
    inlier_dst = dst_enu_pts[best_inliers]
    res = compute_residuals(inlier_src, inlier_dst, best_scale, best_rot, best_trans)

    horiz_errors = np.linalg.norm(res[:, :2], axis=1)
    vert_errors = np.abs(res[:, 2])
    total_errors = np.linalg.norm(res, axis=1)

    rmse_h = float(np.sqrt(np.mean(horiz_errors**2))) if len(horiz_errors) > 0 else 0.0
    rmse_v = float(np.sqrt(np.mean(vert_errors**2))) if len(vert_errors) > 0 else 0.0
    rmse_3d = float(np.sqrt(np.mean(total_errors**2))) if len(total_errors) > 0 else 0.0
    max_res = float(np.max(total_errors)) if len(total_errors) > 0 else 0.0
    p95_res = float(np.percentile(total_errors, 95)) if len(total_errors) > 0 else 0.0

    diag = GeoreferenceDiagnostics(
        rmse_horizontal_m=rmse_h,
        rmse_vertical_m=rmse_v,
        rmse_total_3d_m=rmse_3d,
        max_residual_m=max_res,
        inlier_count=len(best_inliers),
        total_correspondences=n,
        scale_factor=best_scale,
        residual_spread_p95_m=p95_res,
    )

    # Build 4x4 matrix representation
    # [ s*R_00, s*R_01, s*R_02, t_0 ]
    # [ s*R_10, s*R_11, s*R_12, t_1 ]
    # [ s*R_20, s*R_21, s*R_22, t_2 ]
    # [   0,      0,      0,     1  ]
    sr = best_scale * best_rot
    t4x4 = [
        [float(sr[0, 0]), float(sr[0, 1]), float(sr[0, 2]), float(best_trans[0])],
        [float(sr[1, 0]), float(sr[1, 1]), float(sr[1, 2]), float(best_trans[1])],
        [float(sr[2, 0]), float(sr[2, 1]), float(sr[2, 2]), float(best_trans[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]

    notes = [
        f"Sim(3) Umeyama alignment with {len(best_inliers)}/{n} inliers",
        f"Horizontal RMSE: {rmse_h:.3f} m, Vertical RMSE: {rmse_v:.3f} m",
        f"Derived scale factor: {best_scale:.6f}",
    ]

    return GeoreferenceResult(
        success=(len(best_inliers) >= 3 and rmse_3d < max_inlier_residual_m * 2.0),
        reference_point=ref_origin,
        transform_4x4=t4x4,
        diagnostics=diag,
        inlier_indices=best_inliers,
        notes=notes,
    )


def georeference_world_from_gnss(
    world: WorldIR,
    camera_poses: Sequence[Tuple[str, np.ndarray]],
    gnss_records: Sequence[Tuple[str, float, float, float]],
    reference_origin: Optional[GeodeticPoint] = None,
    max_inlier_residual_m: float = 3.0,
) -> GeoreferenceResult:
    """Georeferences a WorldIR instance using paired camera centers and GNSS (lat, lon, alt) observations.

    Args:
        world: Canonical WorldIR instance to georeference.
        camera_poses: Sequence of (evidence_id, local_xyz_array).
        gnss_records: Sequence of (evidence_id, latitude_deg, longitude_deg, altitude_m).
        reference_origin: Optional explicitly provided origin; defaults to centroid of GNSS readings.
        max_inlier_residual_m: Residual error threshold in meters for RANSAC inlier consensus.

    Returns:
        GeoreferenceResult containing Sim(3) transform, diagnostics, and reference origin.
    """
    pose_map = {ev_id: np.asarray(xyz, dtype=np.float64) for ev_id, xyz in camera_poses}
    gnss_map = {ev_id: (lat, lon, alt) for ev_id, lat, lon, alt in gnss_records}

    common_keys = sorted(set(pose_map.keys()) & set(gnss_map.keys()))
    if len(common_keys) < 3:
        raise ValueError(
            f"Cannot georeference world: found only {len(common_keys)} matching camera/GNSS pairs (need >= 3)"
        )

    # Establish reference origin if not provided
    if reference_origin is None:
        lats = [gnss_map[k][0] for k in common_keys]
        lons = [gnss_map[k][1] for k in common_keys]
        alts = [gnss_map[k][2] for k in common_keys]
        reference_origin = GeodeticPoint(
            latitude_deg=float(np.mean(lats)),
            longitude_deg=float(np.mean(lons)),
            altitude_m=float(np.mean(alts)),
        )

    # Convert GNSS observations into local ENU tangent coordinates
    src_pts = np.array([pose_map[k] for k in common_keys], dtype=np.float64)
    dst_enu = np.array(
        [
            lla_to_enu(
                gnss_map[k][0],
                gnss_map[k][1],
                gnss_map[k][2],
                reference_origin.latitude_deg,
                reference_origin.longitude_deg,
                reference_origin.altitude_m,
            )
            for k in common_keys
        ],
        dtype=np.float64,
    )

    result = align_georeference_ransac(
        src_local_pts=src_pts,
        dst_enu_pts=dst_enu,
        ref_origin=reference_origin,
        max_inlier_residual_m=max_inlier_residual_m,
    )

    if result.success:
        # Promote world coordinate frame to Frame.ENU
        world.coordinate_frame = Frame.ENU

        # Register georeference transform edge
        tf = Transform(
            source_frame=Frame.WORLD,
            target_frame=Frame.ENU,
            matrix=result.transform_4x4,
            uncertainty=Uncertainty(
                confidence=max(0.0, min(1.0, 1.0 - result.diagnostics.rmse_total_3d_m / 10.0)),
                note="umeyama_sim3_gnss_fusion",
            ),
        )
        world.transforms["world_to_enu"] = tf

        # Add georeferencing metadata to world metadata
        world.metadata["georeference"] = {
            "status": "georeferenced",
            "frame": Frame.ENU.value,
            "origin": reference_origin.to_dict(),
            "diagnostics": result.diagnostics.to_dict(),
            "notes": result.notes,
        }

    return result
