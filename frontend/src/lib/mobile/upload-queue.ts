/**
 * Per-photo upload queue state machine for the mobile capture screen.
 *
 * Pure (no React, no fetch) so it can be contract-tested. Photos upload ONE AT A
 * TIME: the first accepted photo creates the world, every later photo must join
 * that SAME world, so a second upload may never start until the first has told
 * us the world id (or until the user removes the failed first photo).
 *
 * Nothing here is invented: `progress` only moves from real XHR upload events,
 * and `accepted` / `rejected` / `failed` come from the API's actual response.
 */

export type UploadStatus = "queued" | "uploading" | "accepted" | "rejected" | "failed";

export interface QueueItem {
  id: string;
  name: string;
  size: number;
  status: UploadStatus;
  /** 0..1 of bytes sent; reaching 1 means "sent, waiting for the server". */
  progress: number;
  attempts: number;
  error: string | null;
  /** false = retrying cannot help (e.g. the API said the request itself is invalid). */
  retryable: boolean;
  evidenceId: string | null;
  rejectedReason: string | null;
}

export interface QueueState {
  items: QueueItem[];
}

export type QueueAction =
  | { type: "add"; items: { id: string; name: string; size: number }[] }
  | { type: "start"; id: string }
  | { type: "progress"; id: string; loaded: number; total: number }
  | { type: "accepted"; id: string; evidenceId: string }
  | { type: "rejected"; id: string; reason: string }
  | { type: "failed"; id: string; error: string; retryable: boolean }
  | { type: "retry"; id: string }
  | { type: "remove"; id: string };

export const EMPTY_QUEUE: QueueState = { items: [] };

function update(state: QueueState, id: string, from: UploadStatus[], patch: (i: QueueItem) => QueueItem): QueueState {
  let changed = false;
  const items = state.items.map((i) => {
    if (i.id !== id || !from.includes(i.status)) return i;
    changed = true;
    return patch(i);
  });
  return changed ? { items } : state;
}

export function queueReducer(state: QueueState, action: QueueAction): QueueState {
  switch (action.type) {
    case "add": {
      const known = new Set(state.items.map((i) => i.id));
      const fresh = action.items
        .filter((i) => !known.has(i.id))
        .map<QueueItem>((i) => ({
          ...i,
          status: "queued",
          progress: 0,
          attempts: 0,
          error: null,
          retryable: true,
          evidenceId: null,
          rejectedReason: null,
        }));
      return fresh.length ? { items: [...state.items, ...fresh] } : state;
    }
    case "start":
      return update(state, action.id, ["queued"], (i) => ({
        ...i,
        status: "uploading",
        progress: 0,
        attempts: i.attempts + 1,
        error: null,
      }));
    case "progress": {
      const p = action.total > 0 ? Math.min(1, Math.max(0, action.loaded / action.total)) : 0;
      // Monotonic: a late/out-of-order event never moves the bar backwards.
      return update(state, action.id, ["uploading"], (i) => (p > i.progress ? { ...i, progress: p } : i));
    }
    case "accepted":
      return update(state, action.id, ["uploading"], (i) => ({
        ...i,
        status: "accepted",
        progress: 1,
        evidenceId: action.evidenceId,
      }));
    case "rejected":
      return update(state, action.id, ["uploading"], (i) => ({
        ...i,
        status: "rejected",
        progress: 1,
        rejectedReason: action.reason,
      }));
    case "failed":
      return update(state, action.id, ["uploading"], (i) => ({
        ...i,
        status: "failed",
        error: action.error,
        retryable: action.retryable,
      }));
    case "retry":
      return update(state, action.id, ["failed"], (i) => ({
        ...i,
        status: "queued",
        progress: 0,
        error: null,
      }));
    case "remove": {
      const items = state.items.filter((i) => !(i.id === action.id && i.status !== "uploading"));
      return items.length === state.items.length ? state : { items };
    }
  }
}

/**
 * The next photo to send, or null. At most one upload at a time. While no world
 * exists yet, a failed photo HOLDS the queue: otherwise the next photo would
 * create a second, separate world.
 */
export function nextToUpload(state: QueueState, worldId: string | null): QueueItem | null {
  if (state.items.some((i) => i.status === "uploading")) return null;
  if (worldId === null && state.items.some((i) => i.status === "failed")) return null;
  return state.items.find((i) => i.status === "queued") ?? null;
}

export interface QueueSummary {
  total: number;
  queued: number;
  uploading: number;
  accepted: number;
  rejected: number;
  failed: number;
  /** photos that still need the network */
  pending: number;
}

export function summarize(state: QueueState): QueueSummary {
  const s: QueueSummary = { total: state.items.length, queued: 0, uploading: 0, accepted: 0, rejected: 0, failed: 0, pending: 0 };
  for (const i of state.items) s[i.status] += 1;
  s.pending = s.queued + s.uploading;
  return s;
}

/** Structural view of POST /api/reconstructions' response (one file per request). */
export interface ReconstructionResponseLike {
  evidence: { id: string }[];
  rejected: { name: string; reason: string }[];
}

export type UploadOutcome = { kind: "accepted"; evidenceId: string } | { kind: "rejected"; reason: string };

/** What the API said about the single photo we sent. */
export function outcomeFromResponse(res: ReconstructionResponseLike): UploadOutcome {
  const accepted = res.evidence[0];
  if (accepted) return { kind: "accepted", evidenceId: accepted.id };
  const rejected = res.rejected[0];
  return {
    kind: "rejected",
    reason: rejected?.reason ?? "The engine neither accepted nor rejected this photo.",
  };
}
