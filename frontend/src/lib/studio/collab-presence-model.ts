/**
 * Data models and studio token helpers for Multi-User Real-Time Collaborative
 * Telepresence & Live Spatial Markup.
 */

export type MarkupType = "pin" | "stroke" | "measurement";

export type MarkupTrade =
  | "architecture"
  | "structure"
  | "mep"
  | "safety"
  | "general";

export type MarkupStatus = "open" | "in_review" | "resolved";

export interface PeerPresence {
  client_id: string;
  user_name: string;
  role: string;
  color: string;
  camera_position: [number, number, number];
  look_at: [number, number, number];
  field_of_view: number;
  last_heartbeat: number;
  is_active: boolean;
}

export interface CameraViewpoint {
  position?: [number, number, number];
  direction?: [number, number, number];
  fov?: number;
}

export interface SpatialMarkup {
  markup_id: string;
  world_id: string;
  markup_type: MarkupType;
  trade: MarkupTrade;
  title: string;
  description: string;
  author_name: string;
  author_id: string;
  position: [number, number, number];
  normal: [number, number, number];
  stroke_points: [number, number, number][];
  camera_viewpoint?: CameraViewpoint;
  color: string;
  status: MarkupStatus;
  created_at: string;
}

export interface CollabPresenceState {
  world_id: string;
  peer_count: number;
  peers: PeerPresence[];
}

export interface CollabMarkupsState {
  world_id: string;
  markup_count: number;
  markups: SpatialMarkup[];
}

/**
 * Return authoritative hex color for an architectural trade markup.
 */
export function getTradeColorHex(trade: MarkupTrade): string {
  switch (trade) {
    case "architecture":
      return "#38bdf8"; // Light Sky
    case "structure":
      return "#f59e0b"; // Amber
    case "mep":
      return "#10b981"; // Emerald
    case "safety":
      return "#ef4444"; // Crimson Red
    case "general":
    default:
      return "#a855f7"; // Purple
  }
}

/**
 * Return human-readable trade label.
 */
export function getTradeLabel(trade: MarkupTrade): string {
  switch (trade) {
    case "architecture":
      return "Architecture";
    case "structure":
      return "Structure";
    case "mep":
      return "MEP / Services";
    case "safety":
      return "Safety & Code";
    case "general":
    default:
      return "General Note";
  }
}

/**
 * Return human-readable status label.
 */
export function getStatusLabel(status: MarkupStatus): string {
  switch (status) {
    case "open":
      return "Open Issue";
    case "in_review":
      return "In Review";
    case "resolved":
      return "Resolved";
    default:
      return "Open";
  }
}

/**
 * Format 3D coordinate point to 2 decimal places.
 */
export function formatMarkupCoord(point: [number, number, number]): string {
  return `(${point[0].toFixed(2)}, ${point[1].toFixed(2)}, ${point[2].toFixed(2)}) m`;
}

/**
 * Format timestamp into concise relative or local string.
 */
export function formatRelativeTimestamp(isoDate: string): string {
  try {
    const d = new Date(isoDate);
    if (isNaN(d.getTime())) return "Recently";
    const deltaSec = Math.round((Date.now() - d.getTime()) / 1000);
    if (deltaSec < 60) return "just now";
    if (deltaSec < 3600) return `${Math.floor(deltaSec / 60)}m ago`;
    if (deltaSec < 86400) return `${Math.floor(deltaSec / 3600)}h ago`;
    return d.toLocaleDateString();
  } catch {
    return "Recently";
  }
}
