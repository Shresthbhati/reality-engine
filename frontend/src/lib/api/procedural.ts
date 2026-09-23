import { apiPost } from "./client";

export interface WallOpeningIn {
  wall: "north" | "south" | "east" | "west";
  kind: "door" | "window";
  lateral_offset: number;
  width: number;
  bottom: number;
  top: number;
}

export interface RoomIn {
  name: string;
  width: number;
  depth: number;
  height: number;
  openings?: WallOpeningIn[];
  seed?: number;
  origin?: [number, number, number];
}

export interface RoomCreatedResult {
  version_id: string;
  room_entity_id: string;
  entity_ids_added: string[];
}

/** Creates a room via the real procedural/room_grammar.py generator,
 * committed as a new WorldStore version -- POST /api/worlds/{id}/rooms. */
export function createRoom(worldId: string, body: RoomIn): Promise<RoomCreatedResult> {
  return apiPost<RoomCreatedResult>(`/api/worlds/${encodeURIComponent(worldId)}/rooms`, body);
}
