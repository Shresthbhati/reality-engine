"use client";

/**
 * IndexedDB persistent offline queue for mobile photo capture.
 *
 * Ensures captured photos, timestamped GPS snapshots, and upload state
 * survive page reloads, tab crashes, or intermittent network drops.
 * Zero external dependencies; safely no-ops on SSR / environments without IndexedDB.
 */

import type { QueueItemLocation } from "./upload-queue";

export interface StoredCapturePhoto {
  id: string;
  name: string;
  size: number;
  type: string;
  blob: Blob;
  status: "queued" | "uploading" | "failed" | "accepted" | "rejected";
  worldId: string | null;
  sessionId?: string | null;
  location?: QueueItemLocation | null;
  attempts: number;
  error?: string | null;
  evidenceId?: string | null;
  createdAt: number;
}

const DB_NAME = "reality-engine-offline-capture";
const DB_VERSION = 1;
const STORE_NAME = "capture-photos";

function getIndexedDB(): IDBFactory | null {
  if (typeof window !== "undefined" && window.indexedDB) {
    return window.indexedDB;
  }
  if (typeof indexedDB !== "undefined") {
    return indexedDB;
  }
  return null;
}

export function openCaptureDb(): Promise<IDBDatabase | null> {
  const idb = getIndexedDB();
  if (!idb) return Promise.resolve(null);

  return new Promise((resolve) => {
    try {
      const request = idb.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
          store.createIndex("by_status", "status", { unique: false });
          store.createIndex("by_createdAt", "createdAt", { unique: false });
        }
      };

      request.onsuccess = (event) => {
        resolve((event.target as IDBOpenDBRequest).result);
      };

      request.onerror = () => {
        resolve(null);
      };
    } catch {
      resolve(null);
    }
  });
}

export async function persistCapturePhoto(item: StoredCapturePhoto): Promise<void> {
  const db = await openCaptureDb();
  if (!db) return;

  return new Promise((resolve, reject) => {
    try {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const req = store.put(item);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    } catch (err) {
      reject(err);
    }
  });
}

export async function updateCapturePhotoStatus(
  id: string,
  patch: Partial<Omit<StoredCapturePhoto, "id" | "blob">>,
): Promise<void> {
  const db = await openCaptureDb();
  if (!db) return;

  return new Promise((resolve, reject) => {
    try {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const getReq = store.get(id);

      getReq.onsuccess = () => {
        const existing = getReq.result as StoredCapturePhoto | undefined;
        if (!existing) {
          resolve();
          return;
        }
        const updated = { ...existing, ...patch };
        const putReq = store.put(updated);
        putReq.onsuccess = () => resolve();
        putReq.onerror = () => reject(putReq.error);
      };
      getReq.onerror = () => reject(getReq.error);
    } catch (err) {
      reject(err);
    }
  });
}

export async function loadPendingCapturePhotos(): Promise<StoredCapturePhoto[]> {
  const db = await openCaptureDb();
  if (!db) return [];

  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();

      req.onsuccess = () => {
        const results = (req.result as StoredCapturePhoto[]) || [];
        // Pending photos are those not accepted and not permanently rejected.
        // If an item was left as "uploading" when the browser closed/reloaded,
        // reset it to "queued" so the queue pump will naturally pick it up.
        const pending = results.filter((p) => p.status !== "accepted" && p.status !== "rejected");
        for (const item of pending) {
          if (item.status === "uploading") {
            item.status = "queued";
            store.put(item);
          }
        }
        pending.sort((a, b) => a.createdAt - b.createdAt);
        resolve(pending);
      };
      req.onerror = () => resolve([]);
    } catch {
      resolve([]);
    }
  });
}

export async function removeCapturePhoto(id: string): Promise<void> {
  const db = await openCaptureDb();
  if (!db) return;

  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const req = store.delete(id);
      req.onsuccess = () => resolve();
      req.onerror = () => resolve();
    } catch {
      resolve();
    }
  });
}

export async function clearCapturePhotos(): Promise<void> {
  const db = await openCaptureDb();
  if (!db) return;

  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const req = store.clear();
      req.onsuccess = () => resolve();
      req.onerror = () => resolve();
    } catch {
      resolve();
    }
  });
}
