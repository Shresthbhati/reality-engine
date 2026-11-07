import { apiGet, apiPost } from "./client";
import type { LostConnection, RoomChangeSet, RoomProposals } from "@/lib/rooms/rooms-model";

export * from "@/lib/rooms/rooms-model";

export interface RoomsCommitResult {
  version_id: string;
  world_id: string;
  duplicate: boolean;
  operation?: string;
}

/** Read-only: door joins and possible second visits, each with a confidence. Nothing is applied by asking. */
export function getRoomProposals(worldId: string): Promise<RoomProposals> {
  return apiGet<RoomProposals>(`/api/worlds/${encodeURIComponent(worldId)}/rooms/proposals`);
}

/** Read-only: what differs between `oldRoom` and a possible second visit `newRoom`. */
export function getRoomChangeSet(worldId: string, oldRoom: string, newRoom: string): Promise<RoomChangeSet> {
  const q = new URLSearchParams({ old: oldRoom, new: newRoom });
  return apiGet<RoomChangeSet>(`/api/worlds/${encodeURIComponent(worldId)}/rooms/change-set?${q}`);
}

/** The user confirmed these two doors are one opening: the second door's room moves to meet the first. */
export function connectRooms(worldId: string, doorId: string, otherDoorId: string): Promise<RoomsCommitResult> {
  return apiPost<RoomsCommitResult>(`/api/worlds/${encodeURIComponent(worldId)}/rooms/connect`, { door_id: doorId, other_door_id: otherDoorId });
}

/** The user confirmed `newRoom` is a second visit to `oldRoom`: the new capture replaces it. */
export function acceptRevisit(worldId: string, oldRoom: string, newRoom: string): Promise<RoomsCommitResult & { lost_connections?: LostConnection[] }> {
  return apiPost(`/api/worlds/${encodeURIComponent(worldId)}/rooms/accept-revisit`, { old_room: oldRoom, new_room: newRoom });
}
