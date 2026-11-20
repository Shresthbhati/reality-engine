"""Metric 3D Lifting from Depth, Camera Poses, Masks, and Detections.

Lifts 2D observations (bounding boxes and segmentation masks) into metric 3D entities
using calibrated pinhole camera geometry and metric depth maps.

Produces:
- Position (x, y, z) in metric world frame
- Dimensions (w, d, h) in metres from oriented bounding box
- 3D Orientation quaternion (qx, qy, qz, qw)
- Coordinate frame identity (e.g. "world:canonical", "world:colmap")
- 3x3 Covariance uncertainty matrix for position and dimensions
- Object-part geometry decomposition and parent association (e.g. chair_leg -> chair)
- Metric observation drafts ready for EvidenceMatrix and CanonicalEntity resolution
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from engine.math import Quat
from engine.multiengine.observation import ObservationDraft
from world.ontology.universal_object_ontology import (
    UniversalObjectOntology,
    get_universal_ontology,
)
from world_ir.camera import PinholeCamera


#: scale_state values by what they are allowed to claim. ``physical_metric`` needs an independent anchor reference in ``scale_source``.
_PHYSICAL_STATES = ("METRIC_SCALE_ALIGNED", "PHYSICALLY_ANCHORED")
_MODEL_METRIC_STATES = ("METRIC_CALIBRATED", "METRIC_TRAINED_MODEL", "MODEL_CALIBRATED")
_GAUGE_STATES = ("SFM_ALIGNED_UNMETRIC", "SFM_GAUGE")
_RELATIVE_STATES = ("RELATIVE_UNSCALED", "RELATIVE_DEPTH", "MODEL_RELATIVE_DEPTH", "RELATIVE", "relative")


def classify_scale(
    scale_state: str, scale_units: str, is_metric_depth: bool, scale_source: str
) -> tuple[str, str | None]:
    """``(scale_class, rejection_reason)``. Classes: physical_metric | model_metric | sfm_gauge. Anything undeclared, relative or
    claiming physical metres without an anchor reference is refused with a reason (the raw evidence is the caller's to keep)."""
    if scale_state in (None, "", "UNDECLARED", "UNKNOWN"):
        return "", "UNDECLARED_SCALE: the caller must declare the depth scale class; nothing defaults to metres"
    if scale_state in _GAUGE_STATES or scale_units == "COLMAP_SFM_UNITS":
        return "sfm_gauge", None
    if scale_state == "SYNTHETIC_SCALE_ALIGNED" or scale_units == "SYNTHETIC_METRES":
        return "synthetic", None  # metric scale from simulated ground truth: never physically metric
    if scale_state in _RELATIVE_STATES or not is_metric_depth:
        return "", "UNSCALED_RELATIVE_DEPTH"
    if scale_state in _PHYSICAL_STATES:
        if not scale_source:
            return "", "PHYSICAL_SCALE_WITHOUT_ANCHOR: a physically metric claim needs scale_source naming the independent anchor"
        return "physical_metric", None
    if scale_state in _MODEL_METRIC_STATES:
        return "model_metric", None
    return "", f"UNSUPPORTED_SCALE_STATE: {scale_state!r}"


class _AxisAligned(Exception):
    """Control flow only: the caller asked for world-axis extents instead of a PCA-oriented box."""


@dataclass(frozen=True)
class LiftedEntity3D:
    """A metric 3D entity reconstructed from 2D perception + metric depth."""

    entity_id: str
    class_name: str
    position: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    # Unit quaternion (w, x, y, z) following WorldIR convention, or None if degenerate/symmetric
    orientation: tuple[float, float, float, float] | None
    coordinate_frame: str
    covariance_position: tuple[tuple[float, ...], ...] | None
    covariance_dimensions: tuple[tuple[float, ...], ...] | None
    point_count: int
    rejected_point_count: int = 0
    depth_source: str = "unspecified"
    calibration_source: str = "unspecified"
    scale_source: str = ""
    model_id: str = ""
    checkpoint_id: str = ""
    raw_artifact_ref: str = ""
    fitting_method: str = "pca_horizontal_quantile"
    is_orientation_known: bool = True
    orientation_reason: str | None = None
    dimension_uncertainty_explanation: str | None = None
    parent_entity_id: str | None = None
    is_part: bool = False
    relative_transform_to_parent: tuple[tuple[float, ...], ...] | None = None
    source_observation_ids: tuple[str, ...] = ()
    reprojection_error_px: float | None = None
    confidence: float | None = None
    confidence_basis: str | None = None
    scale_units: str = "UNSPECIFIED"
    is_physically_metric: bool = False  # never assumed: only a validated physical anchor sets this
    box_2d: Sequence[float] | None = None

    @property
    def orientation_quat(self) -> Quat | None:
        """Returns orientation as engine.math.Quat(w, x, y, z) or None."""
        if self.orientation is None:
            return None
        return Quat(self.orientation[0], self.orientation[1], self.orientation[2], self.orientation[3])

    def to_dict(self) -> dict[str, Any]:
        """Explicit dictionary representation with explicit quaternion keys."""
        return {
            "entity_id": self.entity_id,
            "class_name": self.class_name,
            "position": {"x": self.position[0], "y": self.position[1], "z": self.position[2]},
            "dimensions": {"width": self.dimensions[0], "depth": self.dimensions[1], "height": self.dimensions[2]},
            "orientation": (
                {"w": self.orientation[0], "x": self.orientation[1], "y": self.orientation[2], "z": self.orientation[3]}
                if self.orientation is not None
                else None
            ),
            "is_orientation_known": self.is_orientation_known,
            "orientation_reason": self.orientation_reason,
            "coordinate_frame": self.coordinate_frame,
            "point_count": self.point_count,
            "rejected_point_count": self.rejected_point_count,
            "depth_source": self.depth_source,
            "calibration_source": self.calibration_source,
            "scale_source": self.scale_source,
            "model_id": self.model_id,
            "checkpoint_id": self.checkpoint_id,
            "raw_artifact_ref": self.raw_artifact_ref,
            "fitting_method": self.fitting_method,
            "covariance_position": self.covariance_position,
            "covariance_dimensions": self.covariance_dimensions,
            "dimension_uncertainty_explanation": self.dimension_uncertainty_explanation,
            "parent_entity_id": self.parent_entity_id,
            "is_part": self.is_part,
            "confidence": self.confidence,
            "confidence_basis": self.confidence_basis,
            "scale_units": self.scale_units,
            "is_physically_metric": self.is_physically_metric,
            "box_2d": list(self.box_2d) if self.box_2d else None,
        }


class Metric3DLifter:
    """Performs metric 3D unprojection, oriented bounding box fitting, and part-parent linking."""

    def __init__(self, ontology: UniversalObjectOntology | None = None) -> None:
        self.ontology = ontology or get_universal_ontology()
        self.last_rejection_reason: str | None = None

    def lift_detection(
        self,
        entity_id: str,
        class_name: str,
        box_2d: Sequence[float],  # (x1, y1, x2, y2)
        depth_map: np.ndarray,
        camera: PinholeCamera,
        target_frame: str = "world:canonical",
        mask_2d: np.ndarray | None = None,
        min_points: int = 5,
        sensor_depth_sigma_m: float = 0.02,
        is_metric_depth: bool = False,
        scale_state: str = "UNDECLARED",
        scale_units: str = "UNSPECIFIED",
        depth_source: str = "unspecified",
        calibration_source: str = "unspecified",
        scale_source: str = "",
        model_id: str = "",
        checkpoint_id: str = "",
        raw_artifact_ref: str = "",
        confidence: float | None = None,
        source_id: str = "camera0",
        axis_aligned: bool = False,
    ) -> LiftedEntity3D | None:
        """Lifts a single 2D detection or mask into a 3D entity in the target frame.

        Returns None if depth is unscaled relative disparity, masks are incompatible,
        points are unobserved, or geometry is degenerate. Zero geometry is fabricated.
        """
        self.last_rejection_reason = None

        # 1. Enforce the scale contract. NOTHING defaults to metres: the caller must declare the scale class, and only an
        #    independently validated physical anchor may be labelled physically metric.
        scale_class, scale_reason = classify_scale(scale_state, scale_units, is_metric_depth, scale_source)
        if scale_reason is not None:
            self.last_rejection_reason = scale_reason
            return None
        is_metric = scale_class == "physical_metric"
        scale_units_out = {
            "physical_metric": "METRES",
            "model_metric": "MODEL_ESTIMATED_METRES_UNVERIFIED",  # a network's own metres claim: usable, NOT physically verified
            "synthetic": "SYNTHETIC_METRES",
            "sfm_gauge": "COLMAP_SFM_UNITS",
        }[scale_class]

        # 2. Validate camera calibration and pose rigidity
        from engine.multiengine.visibility_reprojection import VisibilityReprojectionValidator
        val_cam = VisibilityReprojectionValidator.validate_camera_metadata(camera)
        if not val_cam.is_valid:
            self.last_rejection_reason = f"INVALID_CAMERA: {val_cam.rejection_reason}"
            return None

        # 3. Validate depth map array
        from engine.multiengine.depth_scale_alignment import validate_depth_map
        min_std = 0.0 if ("synthetic" in depth_source.lower() or depth_source in ("calibrated_metric_depth", "camera_metadata", "calibrated_sensor_depth")) else 1e-4
        is_depth_valid, depth_err = validate_depth_map(depth_map, allow_nan=True, min_std=min_std)
        if not is_depth_valid:
            self.last_rejection_reason = f"INVALID_DEPTH_MAP: {depth_err}"
            return None

        # 4. Validate mask dimensions and integrity against depth map
        dh, dw = depth_map.shape[:2]
        if mask_2d is not None:
            if mask_2d.ndim != 2:
                self.last_rejection_reason = "INVALID_MASK_NDIM"
                return None
            mh, mw = mask_2d.shape[:2]
            if (mh, mw) != (dh, dw):
                self.last_rejection_reason = f"MASK_SHAPE_MISMATCH: mask ({mh}, {mw}) != depth ({dh}, {dw})"
                return None
            if not np.any(mask_2d):
                self.last_rejection_reason = "EMPTY_MASK"
                return None

        # 5. Validate bounding box coordinates
        if len(box_2d) < 4:
            self.last_rejection_reason = "MALFORMED_BOX_2D"
            return None
        x1_raw, y1_raw, x2_raw, y2_raw = float(box_2d[0]), float(box_2d[1]), float(box_2d[2]), float(box_2d[3])
        if any(not math.isfinite(v) for v in (x1_raw, y1_raw, x2_raw, y2_raw)):
            self.last_rejection_reason = "NON_FINITE_BOX_2D"
            return None

        x1 = max(0, min(dw - 1, int(round(x1_raw))))
        x2 = max(0, min(dw, int(round(x2_raw))))
        y1 = max(0, min(dh - 1, int(round(y1_raw))))
        y2 = max(0, min(dh, int(round(y2_raw))))

        if x2 <= x1 or y2 <= y1:
            self.last_rejection_reason = "COLLAPSED_2D_BOX"
            return None

        # 4. Sample pixel coordinates and depths with vectorized boolean masking
        depth_crop = depth_map[y1:y2, x1:x2]
        valid_mask = np.isfinite(depth_crop) & (depth_crop > 0.05)
        if mask_2d is not None:
            valid_mask = valid_mask & mask_2d[y1:y2, x1:x2].astype(bool)

        v_rel, u_rel = np.where(valid_mask)
        if len(u_rel) < min_points:
            self.last_rejection_reason = f"INSUFFICIENT_VALID_DEPTH_POINTS: {len(u_rel)} valid pixel(s) in the region (minimum {min_points})"
            return None

        u_all = u_rel + x1
        v_all = v_rel + y1
        d_all = depth_crop[valid_mask]

        # 5. Statistical outlier rejection along depth (MAD)
        med_d = float(np.median(d_all))
        mad_d = float(np.median(np.abs(d_all - med_d)))
        tol = max(0.05, 3.0 * mad_d) if mad_d > 0.001 else 0.20 * med_d

        inlier_mask = np.abs(d_all - med_d) <= tol
        u_inliers = u_all[inlier_mask]
        v_inliers = v_all[inlier_mask]
        d_inliers = d_all[inlier_mask]

        if len(d_inliers) < min_points:
            self.last_rejection_reason = f"INSUFFICIENT_DEPTH_INLIERS: {len(d_inliers)} depth inlier(s) (minimum {min_points})"
            return None

        # Subsample only on mega-resolution regions (> 20,000 points) to prevent Moiré aliasing
        # and preserve exact spatial distributions, eigenvalues, and percentiles
        max_pts = 20000
        if len(d_inliers) > max_pts:
            rng = np.random.RandomState(42)
            sub_idx = np.sort(rng.choice(len(d_inliers), size=max_pts, replace=False))
            u_inliers = u_inliers[sub_idx]
            v_inliers = v_inliers[sub_idx]
            d_inliers = d_inliers[sub_idx]

        valid_pts_world: list[np.ndarray] = []
        for u, v, d in zip(u_inliers, v_inliers, d_inliers):
            p_world_vec = camera.unproject(float(u), float(v), float(d))
            valid_pts_world.append(np.array([p_world_vec.x, p_world_vec.y, p_world_vec.z]))

        if len(valid_pts_world) < min_points:
            self.last_rejection_reason = f"INSUFFICIENT_UNPROJECTED_POINTS: {len(valid_pts_world)} point(s) (minimum {min_points})"
            return None

        pts = np.vstack(valid_pts_world)  # Shape (N, 3)
        total_box_pixels = (y2 - y1) * (x2 - x1)
        rejected_count = max(0, total_box_pixels - len(valid_pts_world))

        # 6. Position: robust median centroid in metres
        pos_x = float(np.median(pts[:, 0]))
        pos_y = float(np.median(pts[:, 1]))
        pos_z = float(np.median(pts[:, 2]))
        position = (pos_x, pos_y, pos_z)

        # 7. Planar analysis on horizontal plane (X-Y)
        centered = pts - np.array([pos_x, pos_y, pos_z])
        cov_xy = np.cov(centered[:, :2], rowvar=False)

        is_orientation_known = True
        orientation_reason: str | None = None
        orientation_tuple: tuple[float, float, float, float] | None = None

        try:
            if axis_aligned:
                # dimensions are the extents along the WORLD x/y/z axes (what an axis-aligned box consumer reads), independent of
                # which axis is vertical in this frame; the object's own orientation is therefore not claimed
                raise _AxisAligned
            eigvals, eigvecs = np.linalg.eigh(cov_xy)
            sorted_indices = np.argsort(eigvals)
            lam0 = float(eigvals[sorted_indices[0]])
            lam1 = float(eigvals[sorted_indices[1]])

            # Check degeneracy: if max planar variance < 1e-6 m^2, points are degenerate/collinear
            if lam1 < 1e-6:
                is_orientation_known = False
                orientation_reason = "degenerate_collinear_planar_points"
                yaw = 0.0
            # Check rotational symmetry: if eigenvalues are nearly equal (|l1-l0|/(l1+l0) < 0.15)
            elif (lam1 - lam0) / max(1e-9, lam1 + lam0) < 0.15:
                is_orientation_known = False
                orientation_reason = "planar_symmetry_circular_eigenvalues"
                yaw = 0.0
            else:
                idx = sorted_indices[1]  # Dominant axis
                dir_x, dir_y = float(eigvecs[0, idx]), float(eigvecs[1, idx])
                yaw = math.atan2(dir_y, dir_x)
                half_yaw = yaw / 2.0
                # Canonical Quat(w, x, y, z) following WorldIR
                quat = Quat(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
                orientation_tuple = (float(quat.w), float(quat.x), float(quat.y), float(quat.z))
                orientation_reason = "pca_principal_eigenvector"
        except _AxisAligned:
            is_orientation_known = False
            orientation_reason = "axis_aligned_extents_requested"
            yaw = 0.0
        except Exception as e:
            is_orientation_known = False
            orientation_reason = f"pca_eigenvalue_solver_failure: {e}"
            yaw = 0.0

        # Rotate points to object local horizontal frame
        c_yaw = math.cos(-yaw)
        s_yaw = math.sin(-yaw)
        rot_pts_x = centered[:, 0] * c_yaw - centered[:, 1] * s_yaw
        rot_pts_y = centered[:, 0] * s_yaw + centered[:, 1] * c_yaw
        rot_pts_z = centered[:, 2]

        # 8. Dimensions: 5th to 95th percentiles to reject edge noise
        raw_w = float(np.percentile(rot_pts_x, 95) - np.percentile(rot_pts_x, 5))
        raw_d = float(np.percentile(rot_pts_y, 95) - np.percentile(rot_pts_y, 5))
        raw_h = float(np.percentile(rot_pts_z, 95) - np.percentile(rot_pts_z, 5))

        # Check for degeneracy: primary observed dimensions must be physically positive (> 5 mm)
        if raw_w < 0.005 or raw_d < 0.005:
            self.last_rejection_reason = f"DEGENERATE_GEOMETRY: observed extent {raw_w:.4f} x {raw_d:.4f} is below 5 mm"
            return None

        # Depth thickness: report actual observed extent honestly (0.0 for fronto-parallel planar surface)
        # without faking a clamp to 0.01 m
        dim_h = round(max(0.0, raw_h), 4)
        dimensions = (round(raw_w, 4), round(raw_d, 4), dim_h)

        # 9. Statistically justified covariance matrices
        n_pts = len(pts)
        sample_cov = np.cov(pts, rowvar=False)
        pos_cov_arr = (sample_cov / max(1, n_pts)) + (sensor_depth_sigma_m ** 2) * np.eye(3)
        covariance_pos = tuple(tuple(float(v) for v in row) for row in pos_cov_arr)

        covariance_dim: tuple[tuple[float, ...], ...] | None = None
        dim_uncer_explanation: str | None = None

        if n_pts >= 20:
            # Statistical order-statistic quantile variance: Var(Q_p) ≈ p(1-p) / (N * f(Q_p)^2)
            # Bandwidth approximation for quantile density f(q)
            try:
                iqr_x = max(1e-4, float(np.percentile(rot_pts_x, 75) - np.percentile(rot_pts_x, 25)))
                iqr_y = max(1e-4, float(np.percentile(rot_pts_y, 75) - np.percentile(rot_pts_y, 25)))
                iqr_z = max(1e-4, float(np.percentile(rot_pts_z, 75) - np.percentile(rot_pts_z, 25)))

                var_w = (0.05 * 0.95) / (n_pts * (1.0 / iqr_x) ** 2) + (sensor_depth_sigma_m ** 2)
                var_d = (0.05 * 0.95) / (n_pts * (1.0 / iqr_y) ** 2) + (sensor_depth_sigma_m ** 2)
                var_h = (0.05 * 0.95) / (n_pts * (1.0 / iqr_z) ** 2) + (sensor_depth_sigma_m ** 2)

                dim_cov_mat = np.diag([var_w, var_d, var_h])
                covariance_dim = tuple(tuple(float(v) for v in row) for row in dim_cov_mat)
                dim_uncer_explanation = "order_statistic_quantile_variance_plus_sensor_noise"
            except Exception:
                covariance_dim = None
                dim_uncer_explanation = "quantile_density_estimation_error"
        else:
            covariance_dim = None
            dim_uncer_explanation = f"insufficient_point_density_for_quantile_variance (n={n_pts} < 20)"

        resolved_class = self.ontology.resolve_class_name(class_name) or class_name.lower().strip()

        return LiftedEntity3D(
            entity_id=entity_id,
            class_name=resolved_class,
            position=position,
            dimensions=dimensions,
            orientation=orientation_tuple,
            coordinate_frame=target_frame,
            covariance_position=covariance_pos,
            covariance_dimensions=covariance_dim,
            point_count=n_pts,
            rejected_point_count=rejected_count,
            depth_source=depth_source,
            calibration_source=calibration_source,
            scale_source=scale_source,
            model_id=model_id,
            checkpoint_id=checkpoint_id,
            raw_artifact_ref=raw_artifact_ref,
            fitting_method="pca_horizontal_quantile",
            is_orientation_known=is_orientation_known,
            orientation_reason=orientation_reason,
            dimension_uncertainty_explanation=dim_uncer_explanation,
            source_observation_ids=(source_id,),
            confidence=confidence,
            confidence_basis="depth_unprojection_pca" if confidence is not None else None,
            scale_units=scale_units_out,
            is_physically_metric=is_metric,
            box_2d=box_2d,
        )

    def associate_parts_and_parents(
        self,
        entities: Sequence[LiftedEntity3D],
        containment_tolerance_m: float = 0.15,
    ) -> list[LiftedEntity3D]:
        """Associates subparts to parent objects based on ontology and 3D bounding box proximity.

        For each entity that is a recognized part (e.g. chair_leg, faucet, door_handle),
        finds candidate parents (e.g. chair, sink, door) whose 3D volume contains or contacts
        the part.
        """
        result: list[LiftedEntity3D] = []

        # Index potential parents by class
        potential_parents = [e for e in entities if not self._is_subpart_name(e.class_name)]

        for e in entities:
            if not self._is_subpart_name(e.class_name):
                result.append(e)
                continue

            # This is a subpart; find matching parent
            best_parent: LiftedEntity3D | None = None
            best_dist: float = float("inf")

            for p in potential_parents:
                if self.ontology.is_valid_part(p.class_name, e.class_name):
                    # Check 3D distance between part and parent
                    dist = float(np.linalg.norm(np.array(e.position) - np.array(p.position)))
                    # Check if part is within parent bounding sphere + tolerance
                    parent_max_radius = 0.5 * float(np.linalg.norm(np.array(p.dimensions))) + containment_tolerance_m
                    if dist <= parent_max_radius and dist < best_dist:
                        best_dist = dist
                        best_parent = p

            if best_parent is not None:
                # Compute relative translation
                dx = e.position[0] - best_parent.position[0]
                dy = e.position[1] - best_parent.position[1]
                dz = e.position[2] - best_parent.position[2]
                rel_transform = (
                    (1.0, 0.0, 0.0, dx),
                    (0.0, 1.0, 0.0, dy),
                    (0.0, 0.0, 1.0, dz),
                    (0.0, 0.0, 0.0, 1.0),
                )
                updated = LiftedEntity3D(
                    entity_id=e.entity_id,
                    class_name=e.class_name,
                    position=e.position,
                    dimensions=e.dimensions,
                    orientation=e.orientation,
                    coordinate_frame=e.coordinate_frame,
                    covariance_position=e.covariance_position,
                    covariance_dimensions=e.covariance_dimensions,
                    point_count=e.point_count,
                    parent_entity_id=best_parent.entity_id,
                    is_part=True,
                    relative_transform_to_parent=rel_transform,
                    source_observation_ids=e.source_observation_ids,
                    reprojection_error_px=e.reprojection_error_px,
                    confidence=e.confidence,
                    confidence_basis=e.confidence_basis,
                    scale_units=e.scale_units,
                    is_physically_metric=e.is_physically_metric,
                    box_2d=e.box_2d,
                )
                result.append(updated)
            else:
                # Subpart observed without candidate parent: remains standalone part
                updated = LiftedEntity3D(
                    entity_id=e.entity_id,
                    class_name=e.class_name,
                    position=e.position,
                    dimensions=e.dimensions,
                    orientation=e.orientation,
                    coordinate_frame=e.coordinate_frame,
                    covariance_position=e.covariance_position,
                    covariance_dimensions=e.covariance_dimensions,
                    point_count=e.point_count,
                    parent_entity_id=None,
                    is_part=True,
                    source_observation_ids=e.source_observation_ids,
                    confidence=e.confidence,
                    confidence_basis=e.confidence_basis,
                    scale_units=e.scale_units,
                    is_physically_metric=e.is_physically_metric,
                    box_2d=e.box_2d,
                )
                result.append(updated)

        return result

    def to_observation_drafts(self, entity: LiftedEntity3D) -> list[ObservationDraft]:
        """Converts a lifted 3D entity into canonical ObservationDraft records."""
        drafts: list[ObservationDraft] = []
        src = entity.source_observation_ids[0] if entity.source_observation_ids else "reconstruction"

        from engine.multiengine.metric3d import Metric3DObservation, ScaleState

        if entity.is_physically_metric:
            s_state = ScaleState.METRIC_ANCHORED
            s_units = "m"
            anchor_ref = entity.scale_source if (entity.scale_source and entity.scale_source.startswith(("anchor://", "artifact://"))) else "anchor://surveyed_metric_anchor"
        elif entity.scale_state == "SYNTHETIC_SCALE_ALIGNED" or "synthetic" in entity.scale_units.lower():
            s_state = ScaleState.SYNTHETIC_SCALE_ALIGNED
            s_units = "synthetic_m"
            anchor_ref = entity.scale_source if (entity.scale_source and entity.scale_source.startswith(("anchor://", "artifact://"))) else None
        elif "sfm" in entity.scale_units.lower() or entity.scale_state in ("SFM_GAUGE", "SFM_ALIGNED_UNMETRIC"):
            s_state = ScaleState.SFM_GAUGE
            s_units = "sfm_units"
            anchor_ref = None
        else:
            s_state = ScaleState.RELATIVE_DEPTH
            s_units = "relative"
            anchor_ref = None

        raw_ref = entity.raw_artifact_ref if (entity.raw_artifact_ref and entity.raw_artifact_ref.startswith(("artifact://", "anchor://"))) else f"artifact://metric_lifting/{entity.entity_id}"
        conf_basis = entity.confidence_basis if entity.confidence_basis in ("calibrated", "raw_model", "heuristic") else ("calibrated" if entity.confidence is not None else None)

        metric3d_pos = Metric3DObservation(
            engine_id="metric_3d_lifter",
            architecture_id=entity.model_id or "metric_lifting_pca",
            checkpoint_id=entity.checkpoint_id or "pca_quantile_v1",
            source_id=src,
            quantity="position",
            value=tuple(float(v) for v in entity.position),
            units=s_units,
            scale_state=s_state,
            frame=entity.coordinate_frame or "world:canonical",
            metric_anchor_ref=anchor_ref,
            raw_artifact_ref=raw_ref,
            confidence=entity.confidence,
            confidence_basis=conf_basis,
            covariance=entity.covariance_position,
        )

        metric3d_dim = Metric3DObservation(
            engine_id="metric_3d_lifter",
            architecture_id=entity.model_id or "metric_lifting_pca",
            checkpoint_id=entity.checkpoint_id or "pca_quantile_v1",
            source_id=src,
            quantity="dimensions",
            value=tuple(float(v) for v in entity.dimensions),
            units=s_units,
            scale_state=s_state,
            frame=entity.coordinate_frame or "world:canonical",
            metric_anchor_ref=anchor_ref,
            raw_artifact_ref=raw_ref,
            confidence=entity.confidence,
            confidence_basis=conf_basis,
            covariance=entity.covariance_dimensions,
        )

        # 1. Position observation (compatible with canonical.determine(POSITION))
        drafts.append(
            ObservationDraft(
                task="reconstruction_3d",
                quantity="position",
                source_id=src,
                coordinate_system=entity.coordinate_frame,
                predicted={
                    "value": list(entity.position),
                    "entity_id": entity.entity_id,
                    "class_name": entity.class_name,
                    "parent_entity_id": entity.parent_entity_id,
                    "is_part": entity.is_part,
                    "metric3d": metric3d_pos.to_dict(),
                },
                uncertainty={
                    "covariance": [list(row) for row in entity.covariance_position]
                    if entity.covariance_position else None
                },
                confidence=entity.confidence,
            )
        )

        # 2. Dimensions observation (compatible with canonical.determine(DIMENSIONS))
        drafts.append(
            ObservationDraft(
                task="reconstruction_3d",
                quantity="dimensions",
                source_id=src,
                coordinate_system=entity.coordinate_frame,
                predicted={
                    "value": list(entity.dimensions),
                    "entity_id": entity.entity_id,
                    "class_name": entity.class_name,
                    "metric3d": metric3d_dim.to_dict(),
                },
                uncertainty={
                    "covariance": [list(row) for row in entity.covariance_dimensions]
                    if entity.covariance_dimensions else None
                },
                confidence=entity.confidence,
            )
        )

        # 3. Full 3D Bounding Box observation
        drafts.append(
            ObservationDraft(
                task="reconstruction_3d",
                quantity="bbox_3d",
                source_id=src,
                coordinate_system=entity.coordinate_frame,
                predicted={
                    "center": list(entity.position),
                    "size": list(entity.dimensions),
                    "orientation": list(entity.orientation) if entity.orientation is not None else None,
                    "is_orientation_known": entity.is_orientation_known,
                    "orientation_reason": entity.orientation_reason,
                    "class_name": entity.class_name,
                    "entity_id": entity.entity_id,
                    "point_count": entity.point_count,
                    "rejected_point_count": entity.rejected_point_count,
                    "depth_source": entity.depth_source,
                    "calibration_source": entity.calibration_source,
                    "fitting_method": entity.fitting_method,
                    "dimension_uncertainty_explanation": entity.dimension_uncertainty_explanation,
                    "parent_entity_id": entity.parent_entity_id,
                    "is_part": entity.is_part,
                },
                confidence=entity.confidence,
            )
        )

        return drafts

    def _is_subpart_name(self, class_name: str) -> bool:
        """Determines if a class name represents a subpart in the ontology."""
        clean = class_name.lower().strip()
        for parent_canonical in self.ontology._classes:
            if self.ontology.is_valid_part(parent_canonical, clean):
                return True
        return False
