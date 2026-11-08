import { apiGet, apiPost } from "./client";

export interface RoomWaypoint {
  room_id: string;
  name: string;
  room_type: string;
  center: [number, number, number];
  bounds_min: [number, number, number];
  bounds_max: [number, number, number];
  floor_elevation_m: number;
  ceiling_height_m: number;
  area_sqm: number;
  openings: Array<{
    door_id: string;
    name: string;
    position: [number, number, number];
    width_m: number;
  }>;
}

export interface SpatialBookmark {
  bookmark_id: string;
  name: string;
  room_id: string | null;
  room_name: string | null;
  position: [number, number, number];
  yaw_rad: number;
  pitch_rad: number;
  fov_deg: number;
  tags: string[];
  created_at: string;
}

export interface TourKeyframe {
  keyframe_id: string;
  room_id: string;
  room_name: string;
  position: [number, number, number];
  yaw_rad: number;
  pitch_rad: number;
  dwell_seconds: number;
  transition_seconds: number;
  narration: string;
  kind: "entry" | "panorama" | "feature" | "transition";
}

export interface ArchitecturalTour {
  tour_id: string;
  title: string;
  description: string;
  total_duration_seconds: number;
  keyframes: TourKeyframe[];
  waypoint_path: [number, number, number][];
  rooms_visited: string[];
}

export interface NavigationBounds {
  min_x: number;
  max_x: number;
  min_z: number;
  max_z: number;
}

export interface NavigationData {
  world_id: string;
  rooms: RoomWaypoint[];
  doors: Array<{
    door_id: string;
    name: string;
    position: [number, number, number];
    width_m: number;
  }>;
  recommended_tour: ArchitecturalTour;
  suggested_bookmarks: SpatialBookmark[];
  bounds: NavigationBounds;
}

export interface BookmarkCreateInput {
  name: string;
  room_id?: string | null;
  room_name?: string | null;
  position: [number, number, number];
  yaw_rad: number;
  pitch_rad: number;
  fov_deg?: number;
  tags?: string[];
}

export interface PathResult {
  world_id: string;
  start_room: string;
  target_room: string;
  distance_m: number;
  waypoints: [number, number, number][];
}

/** Fetch full navigation manifest, rooms, doors, and automated tour for a world */
export function fetchWorldNavigation(worldId: string): Promise<NavigationData> {
  return apiGet<NavigationData>(`/api/worlds/${worldId}/navigation`);
}

/** List saved spatial bookmarks for a world */
export function fetchWorldBookmarks(worldId: string): Promise<SpatialBookmark[]> {
  return apiGet<SpatialBookmark[]>(`/api/worlds/${worldId}/bookmarks`);
}

/** Create and persist a new spatial perspective bookmark */
export function createWorldBookmark(
  worldId: string,
  input: BookmarkCreateInput,
): Promise<SpatialBookmark> {
  return apiPost<SpatialBookmark>(`/api/worlds/${worldId}/bookmarks`, input);
}

/** Calculate collision-avoiding route between coordinates/rooms */
export function calculateNavigationPath(
  worldId: string,
  startPos: [number, number, number],
  targetRoomId?: string,
  targetPos?: [number, number, number],
): Promise<PathResult> {
  return apiPost<PathResult>(`/api/worlds/${worldId}/navigation/path`, {
    start_pos: startPos,
    target_room_id: targetRoomId,
    target_pos: targetPos,
  });
}
