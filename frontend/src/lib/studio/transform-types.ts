/**
 * Reality Studio 3D Interactive Transform Types.
 * Defines gizmo modes, coordinate representations, snapping tolerances,
 * collision bounds, and history actions.
 */

export type TransformGizmoMode = "translate" | "rotate" | "scale";

export type TransformAxis = "X" | "Y" | "Z" | "XY" | "XZ" | "YZ" | "XYZ";

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export type Point3D = [number, number, number];
export type Point2D = [number, number];

export interface TransformState {
  position: Point3D;
  /** Euler angles in radians [x, y, z] */
  rotation: Point3D;
  scale: Point3D;
}

export interface SnapConfig {
  /** Master snapping switch */
  enabled: boolean;
  /** Snap distance to adjacent wall faces in meters (default: 0.15m = 15cm) */
  wallSnapThresholdM: number;
  /** Grid translation snap increment in meters (0 = continuous) */
  gridSnapIncrementM: number;
  /** Angle snap increment in degrees (default: 15°) */
  angleSnapIncrementDeg: number;
  /** Automatically align entity rotation parallel or perpendicular to snapped wall */
  alignRotationToWall: boolean;
  /** Collision prevention keeping entity strictly within room interior */
  preventWallPenetration: boolean;
  /** Floor / ceiling vertical constraint */
  constrainToVerticalBounds: boolean;
}

export const DEFAULT_SNAP_CONFIG: SnapConfig = {
  enabled: true,
  wallSnapThresholdM: 0.15,
  gridSnapIncrementM: 0.05,
  angleSnapIncrementDeg: 15,
  alignRotationToWall: true,
  preventWallPenetration: true,
  constrainToVerticalBounds: true,
};

export interface RoomBoundary {
  roomId: string;
  name?: string;
  /** Plan polygon vertices in metric world coordinates [x, y, z] or [x, z] */
  polygon3D: Point3D[];
  /** 2D plan polygon projected on the horizontal ground plane [x, z] (or [x, y] for Z-up) */
  polygon2D: Point2D[];
  /** Elevation of the floor slab in meters */
  floorHeightM: number;
  /** Elevation of the ceiling slab in meters */
  ceilingHeightM: number;
  /** Up axis: "y" (Three.js standard) or "z" */
  upAxis: "y" | "z";
}

export interface WallSnapResult {
  snapped: boolean;
  originalPosition: Point3D;
  snappedPosition: Point3D;
  originalRotation?: Point3D;
  snappedRotation?: Point3D;
  wallSegmentIndex: number | null;
  wallNormal: Point3D | null;
  distanceToWall: number;
  clampedDueToCollision: boolean;
}

export interface EntityBounds {
  width: number;  // X extent
  height: number; // Y extent (elevation in Y-up)
  depth: number;  // Z extent
}
