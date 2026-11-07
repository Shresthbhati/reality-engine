/**
 * Pure architectural walk-mode logic (T-4b.5): eye-height first-person navigation,
 * wall boundary collision detection, touch/keyboard input translation, and pitch clamping.
 *
 * Runs under Node and browser with zero DOM/WebGL dependencies.
 */

export const DEFAULT_EYE_HEIGHT = 1.65; // metres (average human eye height)
export const DEFAULT_BODY_RADIUS = 0.35; // metres (clearance radius against obstacles)
export const DEFAULT_WALK_SPEED = 2.8; // metres per second
export const MIN_PITCH_DEG = -75;
export const MAX_PITCH_DEG = 75;

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

export interface WallSegment {
  p1: Vec2; // [x, z]
  p2: Vec2; // [x, z]
}

export interface WallBox {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
}

export type Obstacle = WallSegment | WallBox;

export interface WalkState {
  position: Vec3; // [x, y, z]
  yawRad: number; // azimuth rotation around Y axis
  pitchRad: number; // elevation angle
}

/**
 * Clamp pitch in radians to prevent camera inversion.
 */
export function clampPitch(pitchRad: number, minDeg = MIN_PITCH_DEG, maxDeg = MAX_PITCH_DEG): number {
  const minRad = (minDeg * Math.PI) / 180;
  const maxRad = (maxDeg * Math.PI) / 180;
  return Math.max(minRad, Math.min(maxRad, pitchRad));
}

/**
 * Place the eye at human height (1.65 m) above floor plane.
 */
export function placeEyeAtHeight(floorY: number, eyeHeight = DEFAULT_EYE_HEIGHT, current?: Vec3): Vec3 {
  return [
    current ? current[0] : 0.0,
    floorY + eyeHeight,
    current ? current[2] : 0.0,
  ];
}

/**
 * Compute XZ horizontal movement delta from forward/strafe inputs and yaw angle.
 * Forward (+1: W/Up, -1: S/Down), Strafe (+1: D/Right, -1: A/Left).
 */
export function computeWalkDelta(
  forward: number,
  strafe: number,
  yawRad: number,
  speed: number = DEFAULT_WALK_SPEED,
  deltaSec: number = 0.016,
): Vec2 {
  // Normalized direction in camera local coordinates
  const len = Math.hypot(forward, strafe);
  if (len < 0.0001) return [0, 0];
  const f = forward / Math.max(1, len);
  const s = strafe / Math.max(1, len);

  // In standard Three.js coordinate system (Z points backward, -Z is forward):
  // Forward vector on XZ plane: [-sin(yaw), -cos(yaw)]
  // Right strafe vector on XZ plane: [cos(yaw), -sin(yaw)]
  const fwdX = -Math.sin(yawRad);
  const fwdZ = -Math.cos(yawRad);
  const rightX = Math.cos(yawRad);
  const rightZ = -Math.sin(yawRad);

  const step = speed * deltaSec;
  const dx = (fwdX * f + rightX * s) * step;
  const dz = (fwdZ * f + rightZ * s) * step;

  return [dx, dz];
}

/**
 * Calculate distance from point (px, pz) to a 2D line segment between p1 and p2.
 */
export function distancePointToSegment(px: number, pz: number, p1: Vec2, p2: Vec2): number {
  const dx = p2[0] - p1[0];
  const dz = p2[1] - p1[1];
  const l2 = dx * dx + dz * dz;
  if (l2 < 1e-6) {
    return Math.hypot(px - p1[0], pz - p1[1]);
  }
  // Projection fraction t clamped to [0, 1]
  const t = Math.max(0, Math.min(1, ((px - p1[0]) * dx + (pz - p1[1]) * dz) / l2));
  const projX = p1[0] + t * dx;
  const projZ = p1[1] + t * dz;
  return Math.hypot(px - projX, pz - projZ);
}

/**
 * Calculate distance from point (px, pz) to an axis-aligned 2D box.
 */
export function distancePointToBox(px: number, pz: number, box: WallBox): number {
  const cx = Math.max(box.minX, Math.min(px, box.maxX));
  const cz = Math.max(box.minZ, Math.min(pz, box.maxZ));
  return Math.hypot(px - cx, pz - cz);
}

/**
 * Check collision against obstacles and apply collision resolution / wall sliding.
 */
export function resolveWallCollision(
  currentX: number,
  currentZ: number,
  desiredX: number,
  desiredZ: number,
  obstacles: Obstacle[],
  radius = DEFAULT_BODY_RADIUS,
): { x: number; z: number; collided: boolean } {
  // Check full destination
  let blocked = false;
  for (const obs of obstacles) {
    const dist = "p1" in obs
      ? distancePointToSegment(desiredX, desiredZ, obs.p1, obs.p2)
      : distancePointToBox(desiredX, desiredZ, obs);
    if (dist < radius) {
      blocked = true;
      break;
    }
  }

  if (!blocked) {
    return { x: desiredX, z: desiredZ, collided: false };
  }

  // Attempt sliding along X axis only
  let xBlocked = false;
  for (const obs of obstacles) {
    const dist = "p1" in obs
      ? distancePointToSegment(desiredX, currentZ, obs.p1, obs.p2)
      : distancePointToBox(desiredX, currentZ, obs);
    if (dist < radius) {
      xBlocked = true;
      break;
    }
  }
  if (!xBlocked) {
    return { x: desiredX, z: currentZ, collided: true };
  }

  // Attempt sliding along Z axis only
  let zBlocked = false;
  for (const obs of obstacles) {
    const dist = "p1" in obs
      ? distancePointToSegment(currentX, desiredZ, obs.p1, obs.p2)
      : distancePointToBox(currentX, desiredZ, obs);
    if (dist < radius) {
      zBlocked = true;
      break;
    }
  }
  if (!zBlocked) {
    return { x: currentX, z: desiredZ, collided: true };
  }

  // Fully blocked: stay at current position
  return { x: currentX, z: currentZ, collided: true };
}

/**
 * Partition screen touch into movement (left 50%) or look (right 50%).
 */
export function classifyTouchRegion(touchX: number, viewportWidth: number): "move" | "look" {
  return touchX < viewportWidth * 0.5 ? "move" : "look";
}

/**
 * Compute normalized touch joystick movement inputs (-1 to 1).
 */
export function computeTouchJoystick(
  deltaX: number,
  deltaY: number,
  maxRadius = 60,
): { forward: number; strafe: number } {
  const dist = Math.hypot(deltaX, deltaY);
  if (dist < 4) return { forward: 0, strafe: 0 };
  const scale = Math.min(1, dist / maxRadius) / dist;
  const strafe = (Math.max(-1, Math.min(1, deltaX * scale))) || 0;
  const forward = (Math.max(-1, Math.min(1, -deltaY * scale))) || 0; // Drag up moves forward
  return { forward, strafe };
}
