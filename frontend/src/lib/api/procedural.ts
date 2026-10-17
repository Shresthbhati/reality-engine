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

export interface CorridorIn {
  name: string;
  length: number;
  width?: number;
  height?: number;
  axis?: "x" | "y";
  /** Existing room entity ids (e.g. "hall-room") this corridor connects. */
  connects?: string[];
  origin?: [number, number, number];
}

export interface CityIn {
  name: string;
  blocks_x?: number;
  blocks_y?: number;
  lot_size?: number;
  street_width?: number;
  min_height?: number;
  max_height?: number;
  seed?: number;
  origin?: [number, number, number];
}

export interface ProceduralCreatedResult {
  version_id: string;
  entity_ids_added: string[];
}

/** POST /api/worlds/{id}/corridors -- procedural/urban_grammar.generate_corridor. */
export function createCorridor(worldId: string, body: CorridorIn): Promise<ProceduralCreatedResult> {
  return apiPost<ProceduralCreatedResult>(`/api/worlds/${encodeURIComponent(worldId)}/corridors`, body);
}

/** POST /api/worlds/{id}/cities -- procedural/urban_grammar.generate_city. */
export function createCity(worldId: string, body: CityIn): Promise<ProceduralCreatedResult> {
  return apiPost<ProceduralCreatedResult>(`/api/worlds/${encodeURIComponent(worldId)}/cities`, body);
}
