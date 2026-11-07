import { apiGet, apiPost } from "./client";
import type { WorldHistory } from "@/lib/history/history-model";

export * from "@/lib/history/history-model";

const base = (worldId: string) => `/api/worlds/${encodeURIComponent(worldId)}`;

/** What can be undone/redone for this world right now (saved edits only; survives a reload). */
export function getWorldHistory(worldId: string): Promise<WorldHistory> {
  return apiGet<WorldHistory>(`${base(worldId)}/history`);
}

/** Move the world back one saved edit. 409 when there is nothing to undo or a reconstruction is running. */
export function undoWorld(worldId: string): Promise<WorldHistory> {
  return apiPost<WorldHistory>(`${base(worldId)}/undo`, {});
}

export function redoWorld(worldId: string): Promise<WorldHistory> {
  return apiPost<WorldHistory>(`${base(worldId)}/redo`, {});
}
