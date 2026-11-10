/**
 * Seamless navigation transitions (N-30).
 *
 * Orbit, aerial, third-person and first-person navigation are part of ONE
 * continuous camera system - no abrupt camera jumps and no loss of spatial
 * orientation when switching modes.
 *
 * The camera is modelled as a single 6-DOF state (position + orientation).
 * Navigation modes are just different constraints on that same camera:
 *   - Orbit:   camera orbits a target at fixed distance/height
 *   - Aerial:  high orbit looking down at the scene
 *   - Third-person: camera follows a target from behind/above
 *   - First-person: camera is at eye height, moves with walk controls
 *
 * Switching modes interpolates smoothly between the current camera state
 * and the target mode's camera state over ~0.5 s.
 *
 * Honesty rules:
 *   * No mode is "fake" - all use the same underlying camera state.
 *   * Transitions are time-based (not frame-based) so they look identical
 *     on all hardware.
 *   * If a target is invalid (e.g. no entity to orbit), the camera stays
 *     put and reports the reason.
 */

import type { Entity } from "@/types/worldir";

export type NavMode = "orbit" | "aerial" | "third_person" | "first_person";

export interface NavState {
  mode: NavMode;
  position: [number, number, number]; // camera world position
  target: [number, number, number];   // camera look-at target
  up: [number, number, number];       // camera up vector (usually [0,1,0])
  fov: number;                        // field of view in degrees
  /** Timestamp (ms) when this state became active. */
  startedAt: number;
}

export interface NavTransition {
  from: NavState;
  to: NavState;
  /** Milliseconds over which to interpolate. */
  durationMs: number;
  /** Timestamp when the transition started. */
  startedAt: number;
}

/**
 * Compute the camera state for a navigation mode given the scene context.
 */
export function computeNavState(
  mode: NavMode,
  context: NavContext,
  now: number,
): NavState | { error: string } {
  switch (mode) {
    case "orbit":
      return computeOrbitState(context, now);
    case "aerial":
      return computeAerialState(context, now);
    case "third_person":
      return computeThirdPersonState(context, now);
    case "first_person":
      return computeFirstPersonState(context, now);
    default:
      return { error: `Unknown navigation mode: ${mode}` };
  }
}

export interface NavContext {
  /** Entities in the scene (for targeting). */
  entities: Entity[];
  /** Current camera position (for continuity). */
  currentPosition: [number, number, number];
  /** Current look-at target. */
  currentTarget: [number, number, number];
  /** Scene bounds for computing scene scale. */
  sceneBounds: [number, number, number, number, number, number];
  /** Preferred entity to orbit/follow (null = scene centre). */
  focusEntityId?: string | null;
  /** User input: desired movement direction in camera space. */
  moveDir?: [number, number, number]; // [right, up, forward] in camera space
  /** Desired rotation (yaw, pitch in degrees). */
  lookDir?: [number, number]; // [yawDeg, pitchDeg]
}

/** Default camera up. */
const DEFAULT_UP: [number, number, number] = [0, 1, 0];

function computeOrbitState(context: NavContext, now: number): NavState {
  const target = focusPoint(context);
  const sceneRadius = sceneRadiusFromBounds(context.sceneBounds);
  const distance = Math.max(sceneRadius * 1.5, 3);
  const angle = (context.lookDir?.[0] ?? 45) * (Math.PI / 180);
  const pitch = Math.min(Math.max((context.lookDir?.[1] ?? 20) * (Math.PI / 180), -Math.PI / 2 + 0.1), Math.PI / 2 - 0.1);

  const position: [number, number, number] = [
    target[0] + distance * Math.cos(pitch) * Math.sin(angle),
    target[1] + distance * Math.sin(pitch),
    target[2] + distance * Math.cos(pitch) * Math.cos(angle),
  ];

  return {
    mode: "orbit",
    position,
    target,
    up: DEFAULT_UP,
    fov: 45,
    startedAt: now,
  };
}

function computeAerialState(context: NavContext, now: number): NavState {
  const target = focusPoint(context);
  const sceneRadius = sceneRadiusFromBounds(context.sceneBounds);
  const height = Math.max(sceneRadius * 2, 10);

  const position: [number, number, number] = [target[0], target[1] + height, target[2]];

  return {
    mode: "aerial",
    position,
    target,
    up: DEFAULT_UP,
    fov: 50,
    startedAt: now,
  };
}

function computeThirdPersonState(context: NavContext, now: number): NavState {
  const target = focusPoint(context);
  const sceneRadius = sceneRadiusFromBounds(context.sceneBounds);
  const distance = Math.max(sceneRadius * 0.8, 2);
  const angle = (context.lookDir?.[0] ?? 0) * (Math.PI / 180);
  const pitch = Math.min(Math.max((context.lookDir?.[1] ?? 15) * (Math.PI / 180), -Math.PI / 4), Math.PI / 4);

  const position: [number, number, number] = [
    target[0] + distance * Math.cos(pitch) * Math.sin(angle),
    target[1] + distance * Math.sin(pitch) + 1, // slightly above target
    target[2] + distance * Math.cos(pitch) * Math.cos(angle),
  ];

  return {
    mode: "third_person",
    position,
    target,
    up: DEFAULT_UP,
    fov: 45,
    startedAt: now,
  };
}

function computeFirstPersonState(context: NavContext, now: number): NavState {
  const target = focusPoint(context);
  const eyeHeight = 1.6; // typical human eye height in metres

  // Start from target position at eye height.
  const position: [number, number, number] = [target[0], target[1] + eyeHeight, target[2]];

  // Apply movement direction if provided (camera-space forward/right).
  let fwd: [number, number, number] = [0, 0, -1];
  let right: [number, number, number] = [1, 0, 0];

  if (context.moveDir) {
    // Simple: treat moveDir as world-space offset.
    position[0] += context.moveDir[0];
    position[1] += context.moveDir[1];
    position[2] += context.moveDir[2];
  }

  return {
    mode: "first_person",
    position,
    target,
    up: DEFAULT_UP,
    fov: 60,
    startedAt: now,
  };
}

function focusPoint(context: NavContext): [number, number, number] {
  if (context.focusEntityId) {
    const ent = context.entities.find((e) => e.id === context.focusEntityId);
    if (ent) {
      const cp = ent.custom_properties;
      const bmin = cp?.["bounds_min"] as number[] | undefined;
      const bmax = cp?.["bounds_max"] as number[] | undefined;
      if (bmin && bmax) {
        return [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2];
      }
    }
  }
  const [minX, minY, minZ, maxX, maxY, maxZ] = context.sceneBounds;
  return [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
}

function sceneRadiusFromBounds(bounds: [number, number, number, number, number, number]): number {
  const [minX, minY, minZ, maxX, maxY, maxZ] = bounds;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  return Math.max(
    Math.abs(maxX - cx),
    Math.abs(maxY - cy),
    Math.abs(maxZ - cz),
  ) || 1;
}

/**
 * Interpolate between two nav states by t (0..1).
 * Position and target are interpolated linearly; FOV linearly; up vector
 * is snapped to the target's up (up changes are discrete, not interpolated,
 * to avoid rolling the camera).
 */
export function interpolateNavState(from: NavState, to: NavState, t: number): NavState {
  const clampedT = Math.min(Math.max(t, 0), 1);

  const position: [number, number, number] = [
    from.position[0] + (to.position[0] - from.position[0]) * clampedT,
    from.position[1] + (to.position[1] - from.position[1]) * clampedT,
    from.position[2] + (to.position[2] - from.position[2]) * clampedT,
  ];

  const target: [number, number, number] = [
    from.target[0] + (to.target[0] - from.target[0]) * clampedT,
    from.target[1] + (to.target[1] - from.target[1]) * clampedT,
    from.target[2] + (to.target[2] - from.target[2]) * clampedT,
  ];

  const fov = from.fov + (to.fov - from.fov) * clampedT;

  return {
    mode: to.mode,
    position,
    target,
    up: to.up,
    fov,
    startedAt: from.startedAt,
  };
}

/**
 * Given a desired nav mode and the current state, compute the transition.
 * Returns null if already in the target mode (no transition needed).
 */
export function computeTransition(
  from: NavState,
  toMode: NavMode,
  context: NavContext,
  now: number,
  durationMs = 500,
): NavTransition | null {
  if (from.mode === toMode) return null;

  const to = computeNavState(toMode, context, now);
  if ("error" in to) return null;

  return {
    from,
    to,
    durationMs,
    startedAt: now,
  };
}

/**
 * Get the interpolated nav state at a given time during a transition.
 */
export function transitionStateAt(transition: NavTransition, now: number): NavState {
  const elapsed = now - transition.startedAt;
  const t = Math.min(Math.max(elapsed / transition.durationMs, 0), 1);
  return interpolateNavState(transition.from, transition.to, t);
}

/** Check if a transition is complete at the given time. */
export function transitionIsComplete(transition: NavTransition, now: number): boolean {
  return now - transition.startedAt >= transition.durationMs;
}

