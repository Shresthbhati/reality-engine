import { apiDelete, apiGet, apiPost, apiPut } from "./client";
import type { GeneratedItem } from "@/lib/procedural/generated-model";

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
  /** Flush to a generated room's side; the server computes the position. */
  attach?: { to: string; side: "east" | "west" | "north" | "south" };
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

export type GeneratedKind = "room" | "corridor" | "city";

const KIND_PATH: Record<GeneratedKind, string> = { room: "rooms", corridor: "corridors", city: "cities" };

function objectPath(worldId: string, kind: GeneratedKind, name: string): string {
  return `/api/worlds/${encodeURIComponent(worldId)}/${KIND_PATH[kind]}/${encodeURIComponent(name)}`;
}

/** GET /api/worlds/{id}/generated -- every stored generated object with its params and dependencies. */
export function listGenerated(worldId: string): Promise<GeneratedItem[]> {
  return apiGet<GeneratedItem[]>(`/api/worlds/${encodeURIComponent(worldId)}/generated`);
}

export interface UpdateResult {
  version_id: string;
  changed: number;
  removed: number;
  unchanged: number;
  entity_ids_changed: string[];
}

/** PUT .../{rooms|corridors|cities}/{name}: partial edit -> new version. Errors (incl. 409/422) carry the server text. */
export function updateGenerated(
  worldId: string,
  kind: GeneratedKind,
  name: string,
  patch: Record<string, unknown>,
): Promise<UpdateResult> {
  return apiPut<UpdateResult>(objectPath(worldId, kind, name), patch);
}

export interface DeleteResult {
  version_id: string;
  removed: number;
  entity_ids_removed: string[];
  cascaded: string[];
}

/** DELETE .../{name}[?cascade=true]. Without cascade the server answers 409 listing dependents. */
export function deleteGenerated(
  worldId: string,
  kind: GeneratedKind,
  name: string,
  cascade = false,
): Promise<DeleteResult> {
  return apiDelete<DeleteResult>(`${objectPath(worldId, kind, name)}${cascade ? "?cascade=true" : ""}`);
}

export interface RoomTemplate {
  name: string;
  width: number;
  depth: number;
  height: number;
}

/** GET /api/procedural/templates -- default dimensions only. */
export function listRoomTemplates(): Promise<{ label: string; templates: RoomTemplate[] }> {
  return apiGet("/api/procedural/templates");
}
