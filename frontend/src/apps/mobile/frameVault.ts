'use client';

/**
 * FrameVault — durable on-device evidence storage (IndexedDB).
 *
 * The field app cannot assume a network, or that a desktop machine is
 * present, so captured pixels must survive a reload on the device itself.
 * Frames are keyed by their sha256 — the same content identity the engine
 * uses for sources (`folder:<hash16>`) — so vault storage, export and
 * desktop ingest agree by construction instead of by convention.
 *
 * Every failure path throws a descriptive Error. A missing vault is a state
 * the UI must be able to show; it is never swallowed into a silent success.
 */

const DB_NAME = 're-mobile-evidence';
const DB_VERSION = 1;
const STORE_NAME = 'frames';

export interface VaultEntry {
  /** sha256 hex of `blob` — storage key, and the frame's content identity. */
  sha256: string;
  blob: Blob;
  width: number;
  height: number;
  capturedAt: string;
  mime: string;
}

export interface VaultStats {
  count: number;
  bytes: number;
}

export function vaultAvailable(): boolean {
  return typeof indexedDB !== 'undefined';
}

/** Copy into a plain ArrayBuffer-backed view (safe as a Blob part everywhere). */
export function bytesToBlob(bytes: Uint8Array, type: string): Blob {
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return new Blob([copy], { type });
}

export async function blobToBytes(blob: Blob): Promise<Uint8Array> {
  return new Uint8Array(await blob.arrayBuffer());
}

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  if (!vaultAvailable()) {
    return Promise.reject(
      new Error('IndexedDB is unavailable in this browser: captured frames cannot be stored durably on this device'),
    );
  }
  dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'sha256' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () =>
      reject(new Error(`IndexedDB open failed: ${req.error?.message ?? 'unknown error'}`));
    req.onblocked = () =>
      reject(new Error('IndexedDB open blocked by another tab: close other Reality Capture tabs and retry'));
  });
  return dbPromise;
}

async function withStore<T>(
  mode: IDBTransactionMode,
  op: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openDb();
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const tx = db.transaction(STORE_NAME, mode);
    tx.onabort = () => {
      if (settled) return;
      settled = true;
      reject(new Error(`IndexedDB transaction aborted: ${tx.error?.message ?? 'unknown error'}`));
    };
    let req: IDBRequest<T>;
    try {
      req = op(tx.objectStore(STORE_NAME));
    } catch (err) {
      settled = true;
      reject(err instanceof Error ? err : new Error(String(err)));
      return;
    }
    req.onsuccess = () => {
      settled = true;
      resolve(req.result);
    };
    req.onerror = () => {
      settled = true;
      reject(new Error(`IndexedDB request failed: ${req.error?.message ?? 'unknown error'}`));
    };
  });
}

/** Store one frame's bytes. Idempotent: the key IS the content hash. */
export async function vaultPut(entry: VaultEntry): Promise<void> {
  await withStore('readwrite', (store) => store.put(entry));
}

export async function vaultGet(sha256: string): Promise<VaultEntry | null> {
  const found = await withStore<VaultEntry | undefined>('readonly', (store) => store.get(sha256));
  return found ?? null;
}

export async function vaultGetMany(sha256s: string[]): Promise<Map<string, VaultEntry>> {
  const out = new Map<string, VaultEntry>();
  if (sha256s.length === 0) return out;
  const db = await openDb();
  return new Promise<Map<string, VaultEntry>>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    tx.onabort = () => reject(new Error(`IndexedDB transaction aborted: ${tx.error?.message ?? 'unknown error'}`));
    tx.onerror = () => reject(new Error(`IndexedDB transaction failed: ${tx.error?.message ?? 'unknown error'}`));
    tx.oncomplete = () => resolve(out);
    for (const sha of sha256s) {
      const req = store.get(sha);
      req.onsuccess = () => {
        const entry = req.result as VaultEntry | undefined;
        if (entry) out.set(sha, entry);
      };
      req.onerror = () => reject(new Error(`IndexedDB read failed for ${sha}: ${req.error?.message ?? 'unknown error'}`));
    }
  });
}

export async function vaultHas(sha256: string): Promise<boolean> {
  const keys = await withStore<IDBValidKey[]>('readonly', (store) => store.getAllKeys());
  return keys.includes(sha256);
}

export async function vaultStats(): Promise<VaultStats> {
  const db = await openDb();
  return new Promise<VaultStats>((resolve, reject) => {
    let count = 0;
    let bytes = 0;
    const tx = db.transaction(STORE_NAME, 'readonly');
    const req = tx.objectStore(STORE_NAME).openCursor();
    tx.onabort = () => reject(new Error(`IndexedDB transaction aborted: ${tx.error?.message ?? 'unknown error'}`));
    req.onerror = () => reject(new Error(`IndexedDB cursor failed: ${req.error?.message ?? 'unknown error'}`));
    req.onsuccess = () => {
      const cursor = req.result;
      if (!cursor) return; // transaction completes -> resolve
      const entry = cursor.value as VaultEntry;
      count += 1;
      bytes += entry.blob?.size ?? 0;
      cursor.continue();
    };
    tx.oncomplete = () => resolve({ count, bytes });
  });
}

/** Remove frames the operator explicitly discarded. Absent keys are not an error. */
export async function vaultDelete(sha256s: string[]): Promise<void> {
  if (sha256s.length === 0) return;
  await withStore('readwrite', (store) => {
    for (const sha of sha256s) store.delete(sha);
    return store.count();
  });
}