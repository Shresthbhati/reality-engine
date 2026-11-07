"use client";

/**
 * Device-side store for photos that have not finished uploading (plan T-5.3): IndexedDB, so a closed tab or a lost connection does not
 * lose them. Every call fails soft: when IndexedDB is unavailable (private mode, blocked) the capture screen works exactly as before,
 * it just cannot restore photos after a reload, and `storeAvailable()` says so.
 */

const DB_NAME = "reality-capture";
const STORE = "photos";

interface PhotoRecord {
  id: string;
  name: string;
  size: number;
  type: string;
  lastModified: number;
  blob: Blob;
  addedAt: number;
}

export function storeAvailable(): boolean {
  return typeof indexedDB !== "undefined";
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE, { keyPath: "id" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx<T>(mode: IDBTransactionMode, run: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  try {
    return await new Promise<T>((resolve, reject) => {
      const t = db.transaction(STORE, mode);
      const r = run(t.objectStore(STORE));
      t.oncomplete = () => resolve(r.result);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error);
    });
  } finally {
    db.close();
  }
}

/** Keep `file` under `id` until it has uploaded. Returns false when it could not be stored. */
export async function savePhoto(id: string, file: File): Promise<boolean> {
  if (!storeAvailable()) return false;
  try {
    const rec: PhotoRecord = { id, name: file.name || "photo.jpg", size: file.size, type: file.type || "image/jpeg", lastModified: file.lastModified, blob: file, addedAt: Date.now() };
    await tx("readwrite", (s) => s.put(rec));
    return true;
  } catch {
    return false;
  }
}

export async function removePhoto(id: string): Promise<void> {
  if (!storeAvailable()) return;
  try {
    await tx("readwrite", (s) => s.delete(id));
  } catch {
    /* nothing to remove, or the store is unavailable */
  }
}

/** Every photo still waiting, oldest first, as real `File`s. */
export async function loadPhotos(): Promise<{ id: string; file: File }[]> {
  if (!storeAvailable()) return [];
  try {
    const all = await tx<PhotoRecord[]>("readonly", (s) => s.getAll() as IDBRequest<PhotoRecord[]>);
    return all
      .sort((a, b) => a.addedAt - b.addedAt)
      .map((r) => ({ id: r.id, file: new File([r.blob], r.name, { type: r.type, lastModified: r.lastModified }) }));
  } catch {
    return [];
  }
}
