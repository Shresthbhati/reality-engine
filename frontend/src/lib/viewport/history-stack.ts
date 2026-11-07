/**
 * Undo/Redo history stack for Reality Studio geometry adjustments,
 * object repositioning, and entity property corrections (T-6.7 & T-4b.1).
 *
 * Provides persistent per-world undo/redo backed by the server-side edit
 * command API (/api/worlds/{id}/commands, /undo, /redo) so history is
 * preserved across page reloads and edits in world A never touch world B.
 */

export interface HistoryEntry<T = unknown> {
  id: string;
  type: "transform" | "geometry" | "dimension" | "classification" | "custom" | "furniture" | "surface_finish";
  description: string;
  entityId: string;
  timestamp: number;
  before: T;
  after: T;
  apply?: (state: T) => void;
  undone?: boolean;
  sequence?: number;
}

export type HistoryListener = () => void;

export class HistoryStack {
  private undoStack: HistoryEntry[] = [];
  private redoStack: HistoryEntry[] = [];
  private maxDepth: number;
  private listeners: Set<HistoryListener> = new Set();
  public worldId?: string;

  constructor(maxDepth = 50, worldId?: string) {
    this.maxDepth = maxDepth;
    this.worldId = worldId;
  }

  public push<T>(entry: Omit<HistoryEntry<T>, "id" | "timestamp"> & { id?: string; timestamp?: number }): HistoryEntry<T> {
    const fullEntry: HistoryEntry<T> = {
      ...entry,
      id: entry.id ?? `hist-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      timestamp: entry.timestamp ?? Date.now(),
    };

    this.undoStack.push(fullEntry as HistoryEntry<unknown>);
    if (this.undoStack.length > this.maxDepth) {
      this.undoStack.shift();
    }
    this.redoStack = [];
    this.notify();
    return fullEntry;
  }

  public undo(): HistoryEntry | null {
    const entry = this.undoStack.pop();
    if (!entry) return null;

    try {
      if (entry.apply) {
        entry.apply(entry.before);
      }
    } catch (err) {
      console.error(`Failed to undo action '${entry.description}':`, err);
    }

    this.redoStack.push(entry);
    this.notify();

    // Trigger asynchronous remote undo if backed by a world
    if (this.worldId) {
      this.remoteUndo().catch((err) => {
        console.warn(`Remote undo failed for world ${this.worldId}:`, err);
      });
    }

    return entry;
  }

  public redo(): HistoryEntry | null {
    const entry = this.redoStack.pop();
    if (!entry) return null;

    try {
      if (entry.apply) {
        entry.apply(entry.after);
      }
    } catch (err) {
      console.error(`Failed to redo action '${entry.description}':`, err);
    }

    this.undoStack.push(entry);
    this.notify();

    // Trigger asynchronous remote redo if backed by a world
    if (this.worldId) {
      this.remoteRedo().catch((err) => {
        console.warn(`Remote redo failed for world ${this.worldId}:`, err);
      });
    }

    return entry;
  }

  public async fetchRemoteHistory(): Promise<void> {
    if (!this.worldId) return;
    try {
      const res = await fetch(`/api/worlds/${this.worldId}/commands`);
      if (!res.ok) return;
      const data = await res.json();
      if (Array.isArray(data.commands)) {
        this.undoStack = [];
        this.redoStack = [];
        for (const cmd of data.commands) {
          const item: HistoryEntry = {
            id: cmd.id,
            type: (cmd.command_type as HistoryEntry["type"]) || "custom",
            description: cmd.description,
            entityId: cmd.entity_id,
            timestamp: cmd.created_at ? new Date(cmd.created_at).getTime() : Date.now(),
            before: null,
            after: null,
            undone: cmd.undone,
            sequence: cmd.sequence,
          };
          if (cmd.undone) {
            this.redoStack.push(item);
          } else {
            this.undoStack.push(item);
          }
        }
        this.notify();
      }
    } catch (err) {
      console.warn(`Failed to fetch remote commands for world ${this.worldId}:`, err);
    }
  }

  public async remoteUndo(): Promise<boolean> {
    if (!this.worldId) return false;
    try {
      const res = await fetch(`/api/worlds/${this.worldId}/undo`, { method: "POST" });
      return res.ok;
    } catch {
      return false;
    }
  }

  public async remoteRedo(): Promise<boolean> {
    if (!this.worldId) return false;
    try {
      const res = await fetch(`/api/worlds/${this.worldId}/redo`, { method: "POST" });
      return res.ok;
    } catch {
      return false;
    }
  }

  public canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  public canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  public getUndoCount(): number {
    return this.undoStack.length;
  }

  public getRedoCount(): number {
    return this.redoStack.length;
  }

  public peekUndo(): HistoryEntry | null {
    return this.undoStack.length > 0 ? this.undoStack[this.undoStack.length - 1] : null;
  }

  public peekRedo(): HistoryEntry | null {
    return this.redoStack.length > 0 ? this.redoStack[this.redoStack.length - 1] : null;
  }

  public clear(): void {
    this.undoStack = [];
    this.redoStack = [];
    this.notify();
  }

  public subscribe(listener: HistoryListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    for (const listener of this.listeners) {
      try {
        listener();
      } catch (err) {
        console.error("Error in history listener:", err);
      }
    }
  }
}

// Scoped per-world history stacks (T-6.7: undo in world A never touches B)
const worldHistories = new Map<string, HistoryStack>();

export function getWorldHistory(worldId: string): HistoryStack {
  if (!worldHistories.has(worldId)) {
    const stack = new HistoryStack(50, worldId);
    worldHistories.set(worldId, stack);
    stack.fetchRemoteHistory().catch((err) => {
      console.warn(`Could not sync initial history for world ${worldId}:`, err);
    });
  }
  return worldHistories.get(worldId)!;
}

// Default studio history instance (fallback for unparented UI / global preview)
let defaultStudioHistoryInstance: HistoryStack | null = null;

export function getStudioHistory(worldId?: string): HistoryStack {
  if (worldId) {
    return getWorldHistory(worldId);
  }
  if (!defaultStudioHistoryInstance) {
    defaultStudioHistoryInstance = new HistoryStack(50);
  }
  return defaultStudioHistoryInstance;
}
