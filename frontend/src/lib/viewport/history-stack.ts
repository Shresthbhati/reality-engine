/**
 * Undo/Redo history stack for Reality Studio geometry adjustments,
 * object repositioning, and entity property corrections (Tasks 8.4 & 8.10).
 */

export interface HistoryEntry<T = unknown> {
  id: string;
  type: "transform" | "geometry" | "dimension" | "classification" | "custom";
  description: string;
  entityId: string;
  timestamp: number;
  before: T;
  after: T;
  apply: (state: T) => void;
}

export type HistoryListener = () => void;

export class HistoryStack {
  private undoStack: HistoryEntry[] = [];
  private redoStack: HistoryEntry[] = [];
  private maxDepth: number;
  private listeners: Set<HistoryListener> = new Set();

  constructor(maxDepth = 50) {
    this.maxDepth = maxDepth;
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
      entry.apply(entry.before);
    } catch (err) {
      console.error(`Failed to undo action '${entry.description}':`, err);
    }

    this.redoStack.push(entry);
    this.notify();
    return entry;
  }

  public redo(): HistoryEntry | null {
    const entry = this.redoStack.pop();
    if (!entry) return null;

    try {
      entry.apply(entry.after);
    } catch (err) {
      console.error(`Failed to redo action '${entry.description}':`, err);
    }

    this.undoStack.push(entry);
    this.notify();
    return entry;
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

// Global studio history singleton
let studioHistoryInstance: HistoryStack | null = null;

export function getStudioHistory(): HistoryStack {
  if (!studioHistoryInstance) {
    studioHistoryInstance = new HistoryStack(50);
  }
  return studioHistoryInstance;
}
