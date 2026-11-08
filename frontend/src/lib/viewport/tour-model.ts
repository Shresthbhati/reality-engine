/**
 * Pure architectural tour playback & multi-room navigation model (Track B).
 *
 * Runs in Node and browser with zero DOM/WebGL dependencies.
 */

import type {
  ArchitecturalTour,
  RoomWaypoint,
  SpatialBookmark,
} from "../api/navigation";

export interface InterpolatedPose {
  position: [number, number, number];
  yawRad: number;
  pitchRad: number;
  currentRoom: string;
  narration: string;
  progressPct: number;
  activeKeyframeIndex: number;
  isDwell: boolean;
}

/**
 * Shortest circular difference between two angles in radians in [-pi, pi].
 */
export function shortestAngleDiff(target: number, source: number): number {
  let diff = (target - source) % (2.0 * Math.PI);
  if (diff < -Math.PI) diff += 2.0 * Math.PI;
  if (diff > Math.PI) diff -= 2.0 * Math.PI;
  return diff;
}

/**
 * Interpolate angle in radians taking the shortest circular route.
 */
export function interpolateAngle(a0: number, a1: number, t: number): number {
  const diff = shortestAngleDiff(a1, a0);
  return a0 + diff * t;
}

/**
 * Format seconds into mm:ss display string.
 */
export function formatTourTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Identify which room a given 3D position falls into.
 */
export function detectActiveRoom(
  pos: [number, number, number],
  rooms: RoomWaypoint[],
  margin = 0.2,
): RoomWaypoint | null {
  const [x, y, z] = pos;
  for (const r of rooms) {
    const withinXz =
      x >= r.bounds_min[0] - margin &&
      x <= r.bounds_max[0] + margin &&
      z >= r.bounds_min[2] - margin &&
      z <= r.bounds_max[2] + margin;
    const withinY =
      y >= r.floor_elevation_m - 0.4 &&
      y <= r.floor_elevation_m + r.ceiling_height_m + 0.4;
    if (withinXz && withinY) {
      return r;
    }
  }

  // Fallback: closest room center if within reasonable distance (5.0 m)
  let closest: RoomWaypoint | null = null;
  let minDist = Infinity;
  for (const r of rooms) {
    const d = Math.hypot(x - r.center[0], z - r.center[2]);
    if (d < minDist) {
      minDist = d;
      closest = r;
    }
  }
  return minDist <= 5.0 ? closest : null;
}

/**
 * Evaluates the tour camera pose, yaw, pitch, room, and narration at any elapsed timestamp.
 */
export function interpolateTourPose(
  tour: ArchitecturalTour,
  elapsedSeconds: number,
): InterpolatedPose {
  if (!tour.keyframes || tour.keyframes.length === 0) {
    return {
      position: [0, 1.65, 0],
      yawRad: 0,
      pitchRad: 0,
      currentRoom: "Space",
      narration: "",
      progressPct: 0,
      activeKeyframeIndex: 0,
      isDwell: true,
    };
  }

  const duration = Math.max(0.1, tour.total_duration_seconds);
  const clampedT = Math.max(0, Math.min(duration, elapsedSeconds));
  const progressPct = (clampedT / duration) * 100.0;

  let accumulated = 0;
  for (let i = 0; i < tour.keyframes.length; i++) {
    const kf = tour.keyframes[i];

    // Transition phase
    if (kf.transition_seconds > 0) {
      if (accumulated + kf.transition_seconds > clampedT) {
        const prevKf = tour.keyframes[Math.max(0, i - 1)];
        const alpha = (clampedT - accumulated) / kf.transition_seconds;
        // Smooth Hermite smoothstep
        const s = alpha * alpha * (3.0 - 2.0 * alpha);

        const pos: [number, number, number] = [
          prevKf.position[0] + (kf.position[0] - prevKf.position[0]) * s,
          prevKf.position[1] + (kf.position[1] - prevKf.position[1]) * s,
          prevKf.position[2] + (kf.position[2] - prevKf.position[2]) * s,
        ];
        const yaw = interpolateAngle(prevKf.yaw_rad, kf.yaw_rad, s);
        const pitch = prevKf.pitch_rad + (kf.pitch_rad - prevKf.pitch_rad) * s;

        return {
          position: pos,
          yawRad: yaw,
          pitchRad: pitch,
          currentRoom: kf.room_name,
          narration: `Navigating to ${kf.room_name}...`,
          progressPct,
          activeKeyframeIndex: i,
          isDwell: false,
        };
      }
      accumulated += kf.transition_seconds;
    }

    // Dwell phase
    if (accumulated + kf.dwell_seconds > clampedT) {
      return {
        position: kf.position,
        yawRad: kf.yaw_rad,
        pitchRad: kf.pitch_rad,
        currentRoom: kf.room_name,
        narration: kf.narration,
        progressPct,
        activeKeyframeIndex: i,
        isDwell: true,
      };
    }
    accumulated += kf.dwell_seconds;
  }

  // End of tour
  const last = tour.keyframes[tour.keyframes.length - 1];
  return {
    position: last.position,
    yawRad: last.yaw_rad,
    pitchRad: last.pitch_rad,
    currentRoom: last.room_name,
    narration: "Walkthrough tour complete.",
    progressPct: 100.0,
    activeKeyframeIndex: tour.keyframes.length - 1,
    isDwell: true,
  };
}

/**
 * Find the closest bookmark within 2 meters of the current camera position.
 */
export function findClosestBookmark(
  pos: [number, number, number],
  bookmarks: SpatialBookmark[],
  thresholdM = 2.0,
): SpatialBookmark | null {
  let closest: SpatialBookmark | null = null;
  let minDist = Infinity;
  for (const b of bookmarks) {
    const d = Math.hypot(pos[0] - b.position[0], pos[2] - b.position[2]);
    if (d < minDist && d <= thresholdM) {
      minDist = d;
      closest = b;
    }
  }
  return closest;
}
