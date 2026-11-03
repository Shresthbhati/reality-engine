/**
 * Studio Next-Best-View Capture Plan Client Model (Epic E08-1).
 *
 * Provides models and helper algorithms to interact with intelligent capture recommendations:
 * - Next-best-view poses and orientation (position, yaw, pitch, FOV)
 * - Information Gain (IG) evaluation and uncertainty reduction calculations
 * - Compass heading and intuitive human guidance instructions
 * - Interactive camera frustum visualization state
 */

export interface RecommendedPose {
  viewpoint_id: string;
  position: [number, number, number];
  look_direction: [number, number, number];
  yaw_deg: number;
  pitch_deg: number;
  fov_deg: number;
  information_gain: number;
  priority_rank: number;
  instruction: string;
}

export interface CapturePlanSummary {
  world_id: string;
  initial_uncertainty: number;
  unobserved_patches_count: number;
  uncertainty_reduction_ratio: number;
  recommended_viewpoints: RecommendedPose[];
}

export interface CapturePlanUIState {
  isOpen: boolean;
  selectedViewpointId: string | null;
  showCameraCones: boolean;
  filterMinGain: number;
}

/**
 * Maps yaw degree (-180 to +180 or 0 to 360) to 8-point compass heading string.
 */
export function formatCompassHeading(yawDeg: number): string {
  // Normalize to 0 .. 360
  let norm = ((yawDeg % 360) + 360) % 360;
  const directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  const index = Math.round(norm / 45) % 8;
  return directions[index];
}

/**
 * Formats pitch angle into natural language tilt direction.
 */
export function formatPitchAngle(pitchDeg: number): string {
  const rounded = Math.round(pitchDeg);
  if (Math.abs(rounded) <= 2) {
    return "Level (horizontal)";
  }
  if (rounded > 0) {
    return `Tilt up ${rounded}°`;
  }
  return `Tilt down ${Math.abs(rounded)}°`;
}

/**
 * Computes percentage of total coverage gained from an initial uncertainty and reduction ratio.
 */
export function calculateCoverageImprovementPct(initialUncertainty: number, reductionRatio: number): number {
  if (initialUncertainty <= 0) return 0;
  const clampedRatio = Math.max(0, Math.min(1.0, reductionRatio));
  return Math.round(clampedRatio * 1000) / 10;
}

/**
 * Filters and sorts viewpoints by information gain priority.
 */
export function filterAndSortViewpoints(
  viewpoints: RecommendedPose[],
  minGain: number = 0.0,
  maxCount: number = 10,
): RecommendedPose[] {
  return viewpoints
    .filter((v) => v.information_gain >= minGain)
    .sort((a, b) => b.information_gain - a.information_gain)
    .slice(0, maxCount);
}

/**
 * Generates natural camera pose description.
 */
export function describeCameraPose(pose: RecommendedPose): string {
  const compass = formatCompassHeading(pose.yaw_deg);
  const pitch = formatPitchAngle(pose.pitch_deg);
  return `Position (${pose.position[0].toFixed(1)}m, ${pose.position[1].toFixed(1)}m, ${pose.position[2].toFixed(1)}m), Facing ${compass}, ${pitch}`;
}
