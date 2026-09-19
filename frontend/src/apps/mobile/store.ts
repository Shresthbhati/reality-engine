'use client';

/**
 * Mobile field store — zustand + localStorage persistence, with pixel payloads
 * in IndexedDB (`frameVault`).
 *
 * Separate from the desktop `re-store`: the desktop store's capture slice is a
 * simulation. This store holds only real, locally-derived state — actual
 * captured frame blobs, actual measured quality, actual telemetry when the
 * device reports it, and per-frame durability status (`localCopy`).
 *
 * Split of responsibilities:
 *   localStorage — session index, per-frame measured metadata, tasks
 *   IndexedDB    — the frame pixels, keyed by sha256 (content identity)
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { CaptureTask, FieldSession, FrameRecord, FrameVerdict, SessionSummary } from './types';
import { assessFrame, verdictMessage, type AssessResult } from './quality';
import { bytesToBlob, vaultGet, vaultPut } from './frameVault';
import { MOBILE_TASK_SCHEMA, parseMobileBundle, sha256HexBytes } from './bundle';

export interface FrameCaptureInput {
  blob: Blob;
  width: number;
  height: number;
  origin: 'device_camera' | 'file_import';
  device: string;
  telemetry: FrameRecord['telemetry'];
}

export interface ImportBundleResult {
  sessionId: string;
  framesStored: number;
  skippedFrames: number;
  integrityFailures: number;
}

/** Open tasks for a session, newest first. */
export function openTasksFor(tasks: CaptureTask[], sessionId: string | null): CaptureTask[] {
  if (!sessionId) return [];
  return tasks.filter((t) => t.sessionId === sessionId && t.status === 'OPEN');
}

export interface MobileStore {
  sessions: FieldSession[];
  /** Capture tasks requested by the engine, keyed by their own taskId. */
  tasks: CaptureTask[];
  activeSessionId: string | null;
  /** Task the operator is currently shooting for (drives frame tagging). */
  activeTaskId: string | null;
  sync: { lastSyncAt: string | null; lastSyncError: string | null };
  lastAssessment: { frameId: string; verdict: FrameVerdict; message: string } | null;
  /** Latest device-storage failure; surfaced to the operator, never swallowed. */
  vaultError: string | null;
  /** True once previews have been resolved against the vault. */
  rehydrated: boolean;
  /** Gray buffer of the most recent frame (runtime only). */
  lastGray: { data: Uint8ClampedArray; w: number; h: number } | null;

  createSession: (name: string, intent: string) => string;
  selectSession: (id: string | null) => void;
  setActiveTask: (id: string | null) => void;
  addFrame: (input: FrameCaptureInput) => Promise<FrameRecord>;
  finishSession: (id: string) => void;
  markSynced: (at: string) => void;
  setSyncError: (err: string | null) => void;
  /** Re-resolve every session's previews from the device vault. */
  rehydrateFromVault: () => Promise<{ restored: number; missing: number }>;
  /** Import a verified `re.mobile-session-bundle/v1` file (phone → phone). */
  importBundle: (text: string) => Promise<ImportBundleResult>;
  /** Import a `re.mobile-capture-task/v1` file from the engine. */
  importTaskFile: (text: string) => number;
  setVaultError: (err: string | null) => void;
}
export const useMobileStore = create<MobileStore>()(
  persist(
    (set, get) => ({
      sessions: [],
      tasks: [],
      activeSessionId: null,
      activeTaskId: null,
      sync: { lastSyncAt: null, lastSyncError: null },
      lastAssessment: null,
      vaultError: null,
      rehydrated: false,
      lastGray: null,

      createSession: (name, intent) => {
        const id = crypto.randomUUID();
        const now = new Date().toISOString();
        set((s) => ({
          sessions: [
            { sessionId: id, name, intent, createdAt: now, updatedAt: now, status: 'CAPTURING', frames: [] },
            ...s.sessions,
          ],
          activeSessionId: id,
        }));
        return id;
      },

      selectSession: (id) => set({ activeSessionId: id, activeTaskId: null }),
      setActiveTask: (id) => set({ activeTaskId: id }),
      setVaultError: (err) => set({ vaultError: err }),

      addFrame: async ({ blob, width, height, origin, device, telemetry }) => {
        const session = get().sessions.find((s) => s.sessionId === get().activeSessionId);
        if (!session) throw new Error('No active session: frames cannot be attributed to a session');

        const hex = await sha256HexBytes(new Uint8Array(await blob.arrayBuffer()));
        const frameId = `frame:${hex.slice(0, 16)}`;
        const duplicate = session.frames.find((f) => f.frameId === frameId);
        if (duplicate) {
          // Same bytes already in this session: report it, never store twice.
          set({
            lastAssessment: {
              frameId,
              verdict: 'REDUNDANT',
              message: 'Identical bytes are already in this session — nothing new was added.',
            },
          });
          return duplicate;
        }

        let assess: AssessResult;
        try {
          assess = await assessFrame({ blob, previousGray: get().lastGray });
        } catch (err) {
          throw new Error(`Frame assessment failed: ${(err as Error).message}`);
        }
        set({ lastGray: assess.gray });

        const capturedAt = new Date().toISOString();
        const mime = blob.type || 'image/jpeg';

        // Durable storage first: a frame the operator believes is captured must
        // exist on the device, or say loudly that it does not.
        let localCopy: FrameRecord['localCopy'] = 'missing';
        let vaultError = get().vaultError;
        try {
          await vaultPut({ sha256: hex, blob, width, height, capturedAt, mime });
          localCopy = 'vault';
        } catch (err) {
          vaultError = (err as Error).message;
        }

        const activeTaskId = get().activeTaskId;
        const record: FrameRecord = {
          frameId,
          contentSha256: hex,
          previewUrl: URL.createObjectURL(blob),
          capturedAt,
          width,
          height,
          bytes: blob.size,
          mime,
          verdict: assess.verdict,
          quality: assess.quality,
          reasons: assess.reasons,
          telemetry,
          provenance: { origin, device, capturedBy: 'operator', ingestion: 'mobile_local' },
          taskId: activeTaskId,
          localCopy,
        };

        set((s) => ({
          sessions: s.sessions.map((sess) =>
            sess.sessionId === session.sessionId
              ? { ...sess, frames: [...sess.frames, record], updatedAt: record.capturedAt }
              : sess,
          ),
          // A task is serviced only by evidence that is actually usable.
          tasks: activeTaskId
            ? s.tasks.map((t) =>
                t.taskId === activeTaskId && t.status === 'OPEN' && assess.verdict === 'USEFUL'
                  ? {
                      ...t,
                      status: 'DONE' as const,
                      completedByFrameIds: [...t.completedByFrameIds, frameId],
                      completedAt: capturedAt,
                    }
                  : t,
              )
            : s.tasks,
          lastAssessment: {
            frameId,
            verdict: assess.verdict,
            message:
              localCopy === 'missing' && vaultError
                ? `Captured, but not stored on this device — ${vaultError}`
                : verdictMessage(assess.verdict, assess.reasons),
          },
          vaultError,
        }));
        return record;
      },

      finishSession: (id) =>
        set((s) => ({
          sessions: s.sessions.map((sess) =>
            sess.sessionId === id
              ? { ...sess, status: 'READY_TO_SYNC', updatedAt: new Date().toISOString() }
              : sess,
          ),
        })),
      markSynced: (at) => set({ sync: { lastSyncAt: at, lastSyncError: null } }),
      setSyncError: (err) => set((s) => ({ sync: { ...s.sync, lastSyncError: err } })),

      rehydrateFromVault: async () => {
        let restored = 0;
        let missing = 0;
        let vaultError: string | null = null;
        const nextSessions: FieldSession[] = [];
        for (const sess of get().sessions) {
          const frames: FrameRecord[] = [];
          for (const frame of sess.frames) {
            let entry = null;
            try {
              entry = await vaultGet(frame.contentSha256);
            } catch (err) {
              vaultError = (err as Error).message;
            }
            if (!entry) {
              // The pixels are gone: say so, and do not pretend the frame can
              // still be handed off. A live object URL may still be displayable.
              missing += 1;
              frames.push({ ...frame, localCopy: 'missing' });
              continue;
            }
            restored += 1;
            frames.push({
              ...frame,
              localCopy: 'vault',
              previewUrl: frame.previewUrl || URL.createObjectURL(entry.blob),
            });
          }
          nextSessions.push({ ...sess, frames });
        }
        set((s) => ({ sessions: nextSessions, rehydrated: true, vaultError: vaultError ?? s.vaultError }));
        return { restored, missing };
      },

      importBundle: async (text) => {
        const parsed = await parseMobileBundle(text);
        let framesStored = 0;
        let vaultError = get().vaultError;
        const imported: FrameRecord[] = [];
        for (const f of parsed.frames) {
          const blob = bytesToBlob(f.bytes, f.record.mime);
          let localCopy: FrameRecord['localCopy'] = 'missing';
          try {
            await vaultPut({
              sha256: f.record.contentSha256,
              blob,
              width: f.record.width,
              height: f.record.height,
              capturedAt: f.record.capturedAt,
              mime: f.record.mime,
            });
            localCopy = 'vault';
            framesStored += 1;
          } catch (err) {
            vaultError = (err as Error).message;
          }
          imported.push({ ...f.record, localCopy, previewUrl: URL.createObjectURL(blob) });
        }
        const existing = get().sessions.find((s) => s.sessionId === parsed.session.sessionId);
        const mergedFrames = existing
          ? [
              ...existing.frames,
              ...imported.filter((f) => !existing.frames.some((e) => e.frameId === f.frameId)),
            ]
          : imported;
        const session: FieldSession = existing
          ? { ...existing, frames: mergedFrames, updatedAt: new Date().toISOString() }
          : { ...parsed.session, frames: mergedFrames };
        set((s) => ({
          sessions: [session, ...s.sessions.filter((x) => x.sessionId !== session.sessionId)],
          activeSessionId: session.sessionId,
          vaultError,
        }));
        return {
          sessionId: session.sessionId,
          framesStored,
          skippedFrames: parsed.skippedFrames.length,
          integrityFailures: parsed.integrityFailures.length,
        };
      },

      importTaskFile: (text) => {
        let raw: unknown;
        try {
          raw = JSON.parse(text);
        } catch (err) {
          throw new Error(`Task file is not valid JSON: ${(err as Error).message}`);
        }
        const root = raw as { schema?: unknown; tasks?: unknown };
        if (root.schema !== MOBILE_TASK_SCHEMA) {
          throw new Error(`Unsupported task schema ${String(root.schema)} (expected ${MOBILE_TASK_SCHEMA})`);
        }
        if (!Array.isArray(root.tasks)) throw new Error("Task file is missing a 'tasks' array");
        const incoming: CaptureTask[] = (root.tasks as Record<string, unknown>[])
          .map((t) => ({
            taskId: String(t.taskId ?? ''),
            sessionId: String(t.sessionId ?? ''),
            createdAt: String(t.createdAt ?? ''),
            createdBy: String(t.createdBy ?? 'engine'),
            targetFrameIds: Array.isArray(t.targetFrameIds) ? (t.targetFrameIds as string[]) : [],
            reasons: (t.reasons ?? []) as CaptureTask['reasons'],
            guidance: String(t.guidance ?? ''),
            status: (t.status ?? 'OPEN') as CaptureTask['status'],
            completedByFrameIds: [],
            completedAt: null,
          }))
          .filter((t) => t.taskId !== '' && t.sessionId !== '');
        set((s) => ({
          tasks: [...incoming.filter((t) => !s.tasks.some((x) => x.taskId === t.taskId)), ...s.tasks],
        }));
        return incoming.length;
      },
    }),
    {
      name: 're-mobile-field',
      partialize: (s) => ({
        sessions: s.sessions.map((sess) => ({
          ...sess,
          // Object URLs are per-browsing-context; the pixels live in the vault.
          frames: sess.frames.map((f) => ({ ...f, previewUrl: '' })),
        })),
        tasks: s.tasks,
        activeSessionId: s.activeSessionId,
        sync: s.sync,
      }),
    },
  ),
);

/**
 * Smallest arc that covers every measured heading (degrees), or null when there
 * are fewer than two measurements. Compass headings are circular: a naive
 * max-minus-min reports 359° for two frames one degree apart across north, so
 * the largest gap is subtracted instead.
 */
export function circularSpreadDeg(values: number[]): number | null {
  if (values.length < 2) return null;
  const sorted = values.map((v) => ((v % 360) + 360) % 360).sort((a, b) => a - b);
  let largestGap = 360 - (sorted[sorted.length - 1] - sorted[0]);
  for (let i = 1; i < sorted.length; i++) largestGap = Math.max(largestGap, sorted[i] - sorted[i - 1]);
  return 360 - largestGap;
}

/** Aggregates over real frame records only — nothing is estimated or filled in. */
export function summarizeSession(session: FieldSession): SessionSummary {
  const frames = session.frames;
  const headings = frames
    .map((f) => f.telemetry.headingDeg)
    .filter((h): h is number => typeof h === 'number');
  const positions = frames
    .map((f) => f.telemetry.geolocation)
    .filter((g): g is { latitude: number; longitude: number; accuracyM: number } => g != null);
  const times = frames
    .map((f) => Date.parse(f.capturedAt))
    .filter((t) => Number.isFinite(t))
    .sort((a, b) => a - b);

  return {
    total: frames.length,
    useful: frames.filter((f) => f.verdict === 'USEFUL').length,
    rejected: frames.filter((f) => f.verdict === 'REJECTED').length,
    redundant: frames.filter((f) => f.verdict === 'REDUNDANT').length,
    unassessed: frames.filter((f) => f.verdict === 'UNASSESSED').length,
    bytes: frames.reduce((a, f) => a + f.bytes, 0),
    missingLocalCopy: frames.filter((f) => f.localCopy === 'missing').length,
    headingSpreadDeg: circularSpreadDeg(headings),
    gps:
      positions.length > 0
        ? {
            minLat: Math.min(...positions.map((p) => p.latitude)),
            maxLat: Math.max(...positions.map((p) => p.latitude)),
            minLon: Math.min(...positions.map((p) => p.longitude)),
            maxLon: Math.max(...positions.map((p) => p.longitude)),
          }
        : null,
    timeSpanSec: times.length >= 2 ? (times[times.length - 1] - times[0]) / 1000 : null,
  };
}