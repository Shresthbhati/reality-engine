import type { CaptureRequestItem, SuggestedPose } from "@/lib/api/capture-requests";

/**
 * Format expected gain as an honest percentage string (+XX%).
 */
export function formatExpectedGainPct(gain: number): string {
  const clamped = Math.max(0, Math.min(1.0, Number.isFinite(gain) ? gain : 0));
  const pct = Math.round(clamped * 100);
  return `+${pct}% coverage`;
}

/**
 * Filter requests strictly by status.
 */
export function filterRequestsByStatus(
  requests: CaptureRequestItem[],
  status: "open" | "satisfied" | "dismissed"
): CaptureRequestItem[] {
  return requests.filter((r) => r.status === status);
}

/**
 * Sort requests by descending expected information gain.
 */
export function sortRequestsByGain(requests: CaptureRequestItem[]): CaptureRequestItem[] {
  return [...requests].sort((a, b) => (b.expected_gain || 0) - (a.expected_gain || 0));
}

/**
 * Format a human-readable guidance string for a suggested camera pose.
 */
export function describeSuggestedPose(pose: SuggestedPose): string {
  if (pose.description && pose.description.trim().length > 0) {
    return pose.description;
  }
  const parts: string[] = [];
  if (pose.distance_m !== undefined) {
    parts.push(`at ${pose.distance_m.toFixed(1)}m distance`);
  }
  if (pose.pitch_deg !== undefined) {
    const pitch = pose.pitch_deg;
    if (Math.abs(pitch) < 5) parts.push("level");
    else if (pitch > 0) parts.push(`${Math.round(pitch)}° pitch up`);
    else parts.push(`${Math.round(-pitch)}° pitch down`);
  }
  if (pose.yaw_deg !== undefined) {
    parts.push(`heading ${Math.round((pose.yaw_deg + 360) % 360)}°`);
  }
  if (parts.length > 0) {
    return `Suggested viewpoint: ${parts.join(", ")}`;
  }
  return "Point camera towards unobserved region";
}

/**
 * Group capture requests by region type (e.g. wall, floor, furniture, general).
 */
export function groupRequestsByRegionType(
  requests: CaptureRequestItem[]
): Record<string, CaptureRequestItem[]> {
  const groups: Record<string, CaptureRequestItem[]> = {};
  for (const r of requests) {
    const rType = r.target_region?.region_type || "general";
    if (!groups[rType]) {
      groups[rType] = [];
    }
    groups[rType].push(r);
  }
  return groups;
}
