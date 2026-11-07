/**
 * Studio undo/redo: the server keeps the edit history (every committed edit is a world version), so Undo/Redo ask it first.
 * The in-memory stack remains only for changes that were never committed (e.g. a gizmo drag in progress).
 * Pure functions, no I/O: decisions and wording live here so they can be tested.
 */

export interface WorldHistory {
  can_undo: boolean;
  can_redo: boolean;
  undo_depth: number;
  redo_depth: number;
  head: string | null;
  undo_label: string | null;
  redo_label: string | null;
}

export type HistorySource = "server" | "local" | "none";

/** Server history wins whenever it can act: a local entry for a committed edit would only change the screen, not the saved world. */
export function pickUndoSource(server: WorldHistory | null, localUndoCount: number): HistorySource {
  if (server?.can_undo) return "server";
  return localUndoCount > 0 ? "local" : "none";
}

export function pickRedoSource(server: WorldHistory | null, localRedoCount: number): HistorySource {
  if (server?.can_redo) return "server";
  return localRedoCount > 0 ? "local" : "none";
}

export function undoTitle(server: WorldHistory | null, localDescription: string | null): string {
  if (server?.can_undo) {
    const n = server.undo_depth;
    return `Undo last edit (Ctrl+Z) - ${n} saved edit${n === 1 ? "" : "s"} can be undone`;
  }
  return localDescription ? `Undo: ${localDescription} (Ctrl+Z)` : "Nothing to undo";
}

export function redoTitle(server: WorldHistory | null, localDescription: string | null): string {
  if (server?.can_redo) {
    const n = server.redo_depth;
    return `Redo (Ctrl+Y) - ${n} undone edit${n === 1 ? "" : "s"} can be restored`;
  }
  return localDescription ? `Redo: ${localDescription} (Ctrl+Y)` : "Nothing to redo";
}

/** The version the world is on now. After an undo the newest version is no longer the current one, so never assume list order. */
export function currentVersionOf(versions: { id: string; isCurrent?: boolean }[]): string | null {
  return versions.find((v) => v.isCurrent)?.id ?? versions[0]?.id ?? null;
}
