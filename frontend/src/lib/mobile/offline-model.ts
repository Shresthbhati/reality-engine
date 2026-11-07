/**
 * Offline capture rules (plan T-5.3), pure so they are testable under `node --test`.
 *
 * A photo that still needs the network is kept in the device's IndexedDB; a finished one is not. After a reload the kept photos come
 * back as fresh queued items and upload again. A retryable failure is retried by itself with backoff, a few times, and again as soon as
 * the connection returns. Nothing is retried that the API said cannot succeed.
 */
import type { QueueItem, QueueState } from "./upload-queue";

export interface StoredPhoto {
  id: string;
  name: string;
  size: number;
}

export const MAX_AUTO_RETRIES = 5;
const BASE_DELAY_MS = 2000;
const MAX_DELAY_MS = 30_000;

export function keepInStore(item: Pick<QueueItem, "status" | "retryable">): boolean {
  if (item.status === "queued" || item.status === "uploading") return true;
  if (item.status === "failed") return item.retryable;
  return false;
}

export function restorePlan(stored: StoredPhoto[], queue: QueueState): { add: StoredPhoto[] } {
  const known = new Set(queue.items.map((i) => i.id));
  return { add: stored.filter((s) => !known.has(s.id)).map(({ id, name, size }) => ({ id, name, size })) };
}

export function restoredMessage(count: number): string | null {
  if (count <= 0) return null;
  return `${count} ${count === 1 ? "photo" : "photos"} from before the page closed ${count === 1 ? "was" : "were"} kept and will upload now.`;
}

export function idsToRetry(queue: QueueState): string[] {
  return queue.items.filter((i) => i.status === "failed" && i.retryable).map((i) => i.id);
}

/** Milliseconds to wait before retrying `item` by itself, or null when it must not be retried automatically. */
export function autoRetryDelayMs(item: Pick<QueueItem, "status" | "retryable" | "attempts">): number | null {
  if (item.status !== "failed" || !item.retryable || item.attempts > MAX_AUTO_RETRIES) return null;
  return Math.min(MAX_DELAY_MS, BASE_DELAY_MS * 2 ** Math.max(0, item.attempts - 1));
}
