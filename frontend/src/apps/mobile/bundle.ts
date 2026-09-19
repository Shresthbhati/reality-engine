'use client';

/**
 * Mobile session bundle — the real handoff artifact between the field app and
 * the engine.
 *
 * A bundle is a single self-contained JSON file carrying
 *   1. the session record,
 *   2. every captured frame **with its pixels** (base64), and
 *   3. the on-device triage (verdict + reasons + measured quality),
 * keyed by `sha256(bytes)` so the receiving side verifies content identity
 * instead of trusting the file.
 *
 * Guarantees (mirrored by the authoritative Python reader in
 * `apps/cli/mobile_bridge.py`):
 *   - frame identity is `frame:<sha256[0:16]>` of the payload; a mismatch is a
 *     hard integrity failure, never a warning-then-accept;
 *   - frames whose bytes are not on this device are listed in `skippedFrames` —
 *     never silently dropped, never exported as metadata-only "evidence";
 *   - mobile verdicts travel as *advisory* triage; the engine re-derives its
 *     own judgment during ingestion.
 *
 * Schema: docs/mobile/MOBILE_BUNDLE.md
 */

import type { CaptureTask, CaptureTaskOutcome, FieldSession, FrameRecord } from './types';

export const MOBILE_BUNDLE_SCHEMA = 're.mobile-session-bundle/v1';
/** Engine → device capture requests; see `derive_capture_tasks` in the CLI bridge. */
export const MOBILE_TASK_SCHEMA = 're.mobile-capture-task/v1';

export class MobileBundleFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MobileBundleFormatError';
  }
}

export interface BundleFramePayload {
  record: FrameRecord;
  bytes: Uint8Array;
}

export interface MobileBundleBuildResult {
  text: string;
  bundleId: string;
  includedFrameIds: string[];
  /** Frames that exist in the session but have no local pixels. */
  skippedFrames: { frameId: string; reason: 'local_copy_missing' }[];
  payloadBytes: number;
}

export interface MobileBundleParseResult {
  bundleId: string;
  exportedAt: string;
  session: FieldSession;
  frames: BundleFramePayload[];
  skippedFrames: { frameId: string; reason: string }[];
  taskOutcomes: CaptureTaskOutcome[];
  /** Frames present in the file but failing sha256 verification. */
  integrityFailures: { frameId: string; reason: string }[];
}

// ---------------------------------------------------------------- primitives

export async function sha256HexBytes(bytes: Uint8Array): Promise<string> {
  const view = new Uint8Array(bytes.length);
  view.set(bytes);
  const digest = await crypto.subtle.digest('SHA-256', view);
  let hex = '';
  for (const b of new Uint8Array(digest)) hex += b.toString(16).padStart(2, '0');
  return hex;
}

const BASE64_CHUNK = 0x8000;

export function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += BASE64_CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + BASE64_CHUNK));
  }
  return btoa(binary);
}

export function base64ToBytes(b64: string): Uint8Array {
  if (typeof b64 !== 'string' || b64.length === 0) {
    throw new MobileBundleFormatError('payload is empty or not a string');
  }
  let binary: string;
  try {
    binary = atob(b64);
  } catch {
    throw new MobileBundleFormatError('payload is not valid base64');
  }
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

const MIME_EXTENSIONS: Record<string, string> = {
  'image/jpeg': 'jpg',
  'image/png': 'png',
  'image/webp': 'webp',
  'video/mp4': 'mp4',
};

export function mimeExtension(mime: string): string {
  return MIME_EXTENSIONS[mime] ?? 'bin';
}

/** Deterministic bundle identity: session + ordered frame content hashes + tasks. */
async function computeBundleId(
  sessionId: string,
  frames: { frameId: string; contentSha256: string }[],
  taskOutcomes: CaptureTaskOutcome[],
): Promise<string> {
  const manifest = JSON.stringify({
    schema: MOBILE_BUNDLE_SCHEMA,
    sessionId,
    frames: frames.map((f) => [f.frameId, f.contentSha256]),
    tasks: taskOutcomes
      .map((t) => [t.taskId, t.status])
      .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)),
  });
  const hex = await sha256HexBytes(new TextEncoder().encode(manifest));
  return `bundle:${hex.slice(0, 16)}`;
}

/** The single declared device label when every frame agrees; otherwise "". */
function deviceLabelOf(frames: FrameRecord[]): string {
  if (frames.length === 0) return '';
  const first = frames[0].provenance.device;
  return frames.every((f) => f.provenance.device === first) ? first : '';
}

// ------------------------------------------------------------------- writing

export interface BuildBundleInput {
  session: FieldSession;
  /** Payload bytes keyed by contentSha256 (read from the device vault). */
  payloads: Map<string, Uint8Array>;
  /** Capture tasks belonging to this session; their outcomes travel with it. */
  tasks: CaptureTask[];
  /** Injected for determinism/testing; defaults to now. */
  exportedAt?: string;
}

export async function buildMobileBundle(input: BuildBundleInput): Promise<MobileBundleBuildResult> {
  const { session, payloads, tasks } = input;
  const exportedAt = input.exportedAt ?? new Date().toISOString();

  const skippedFrames: { frameId: string; reason: 'local_copy_missing' }[] = [];
  const included: { frame: FrameRecord; payloadBase64: string }[] = [];
  let payloadBytes = 0;

  for (const frame of session.frames) {
    const bytes = payloads.get(frame.contentSha256);
    if (!bytes || bytes.length === 0) {
      skippedFrames.push({ frameId: frame.frameId, reason: 'local_copy_missing' });
      continue;
    }
    const sha = await sha256HexBytes(bytes);
    if (sha !== frame.contentSha256) {
      throw new MobileBundleFormatError(
        `frame ${frame.frameId}: local bytes hash to ${sha.slice(0, 16)}… but the record claims ` +
          `${frame.contentSha256.slice(0, 16)}… — refusing to export evidence whose identity does not verify`,
      );
    }
    included.push({ frame, payloadBase64: bytesToBase64(bytes) });
    payloadBytes += bytes.length;
  }

  const taskOutcomes: CaptureTaskOutcome[] = tasks
    .filter((t) => t.sessionId === session.sessionId)
    .map((t) => ({
      taskId: t.taskId,
      status: t.status,
      frameIds: t.completedByFrameIds,
      completedAt: t.completedAt,
    }));

  const bundleId = await computeBundleId(
    session.sessionId,
    included.map((i) => ({ frameId: i.frame.frameId, contentSha256: i.frame.contentSha256 })),
    taskOutcomes,
  );

  const payload = {
    schema: MOBILE_BUNDLE_SCHEMA,
    bundleId,
    exportedAt,
    device: deviceLabelOf(session.frames),
    session: {
      sessionId: session.sessionId,
      name: session.name,
      intent: session.intent,
      createdAt: session.createdAt,
      updatedAt: session.updatedAt,
      status: session.status,
    },
    frames: included.map(({ frame, payloadBase64 }) => ({
      frameId: frame.frameId,
      contentSha256: frame.contentSha256,
      capturedAt: frame.capturedAt,
      width: frame.width,
      height: frame.height,
      bytes: frame.bytes,
      mime: frame.mime,
      verdict: frame.verdict,
      reasons: frame.reasons,
      quality: frame.quality,
      telemetry: frame.telemetry,
      provenance: frame.provenance,
      taskId: frame.taskId,
      payloadBase64,
    })),
    skippedFrames,
    taskOutcomes,
  };

  return {
    text: JSON.stringify(payload, null, 2),
    bundleId,
    includedFrameIds: included.map((i) => i.frame.frameId),
    skippedFrames,
    payloadBytes,
  };
}

// ------------------------------------------------------------------- reading

interface RawBundleFrame {
  frameId?: unknown;
  contentSha256?: unknown;
  capturedAt?: unknown;
  width?: unknown;
  height?: unknown;
  bytes?: unknown;
  mime?: unknown;
  verdict?: unknown;
  reasons?: unknown;
  quality?: unknown;
  telemetry?: unknown;
  provenance?: unknown;
  taskId?: unknown;
  payloadBase64?: unknown;
}

function asRecord(value: unknown, field: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new MobileBundleFormatError(`bundle field '${field}' must be an object`);
  }
  return value as Record<string, unknown>;
}

/**
 * Parse + verify a bundle. Per-frame verification failures are returned, never
 * converted into accepted evidence; structural problems throw.
 */
export async function parseMobileBundle(text: string): Promise<MobileBundleParseResult> {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (err) {
    throw new MobileBundleFormatError(`bundle is not valid JSON: ${(err as Error).message}`);
  }
  const root = asRecord(raw, 'root');
  if (root.schema !== MOBILE_BUNDLE_SCHEMA) {
    throw new MobileBundleFormatError(
      `unsupported bundle schema ${String(root.schema)} (expected ${MOBILE_BUNDLE_SCHEMA})`,
    );
  }
  const sessionRaw = asRecord(root.session, 'session');
  const sessionId = String(sessionRaw.sessionId ?? '');
  if (!sessionId) throw new MobileBundleFormatError("bundle session is missing 'sessionId'");
  if (!Array.isArray(root.frames)) throw new MobileBundleFormatError("bundle is missing a 'frames' array");

  const frames: BundleFramePayload[] = [];
  const integrityFailures: { frameId: string; reason: string }[] = [];

  for (const entry of root.frames as RawBundleFrame[]) {
    const rawFrame = asRecord(entry, 'frames[]');
    const frameId = String(rawFrame.frameId ?? '');
    const contentSha256 = String(rawFrame.contentSha256 ?? '');
    let bytes: Uint8Array;
    try {
      bytes = base64ToBytes(String(rawFrame.payloadBase64 ?? ''));
    } catch (err) {
      integrityFailures.push({ frameId, reason: (err as Error).message });
      continue;
    }
    const sha = await sha256HexBytes(bytes);
    if (sha !== contentSha256) {
      integrityFailures.push({
        frameId,
        reason: `payload sha256 ${sha.slice(0, 16)}… does not match declared ${contentSha256.slice(0, 16)}…`,
      });
      continue;
    }
    if (frameId !== `frame:${sha.slice(0, 16)}`) {
      integrityFailures.push({
        frameId,
        reason: `frame id does not match content: expected frame:${sha.slice(0, 16)}`,
      });
      continue;
    }
    frames.push({
      bytes,
      record: {
        frameId,
        contentSha256,
        previewUrl: '',
        capturedAt: String(rawFrame.capturedAt ?? ''),
        width: Number(rawFrame.width ?? 0),
        height: Number(rawFrame.height ?? 0),
        bytes: Number(rawFrame.bytes ?? bytes.length),
        mime: String(rawFrame.mime ?? 'image/jpeg'),
        verdict: (rawFrame.verdict ?? 'UNASSESSED') as FrameRecord['verdict'],
        reasons: (rawFrame.reasons ?? []) as FrameRecord['reasons'],
        quality: (rawFrame.quality ?? {
          sharpness: 0,
          exposure: 0,
          clippedRatio: 0,
          diffVsPrevious: null,
        }) as FrameRecord['quality'],
        telemetry: (rawFrame.telemetry ?? {}) as FrameRecord['telemetry'],
        provenance: (rawFrame.provenance ?? {
          origin: 'file_import',
          device: 'imported bundle',
          capturedBy: 'operator',
          ingestion: 'mobile_local',
        }) as FrameRecord['provenance'],
        taskId: (rawFrame.taskId ?? null) as string | null,
        localCopy: 'vault',
      },
    });
  }

  const skippedRaw = Array.isArray(root.skippedFrames)
    ? (root.skippedFrames as Record<string, unknown>[])
    : [];
  const outcomesRaw = Array.isArray(root.taskOutcomes)
    ? (root.taskOutcomes as Record<string, unknown>[])
    : [];

  return {
    bundleId: String(root.bundleId ?? ''),
    exportedAt: String(root.exportedAt ?? ''),
    session: {
      sessionId,
      name: String(sessionRaw.name ?? 'Imported session'),
      intent: String(sessionRaw.intent ?? ''),
      createdAt: String(sessionRaw.createdAt ?? ''),
      updatedAt: String(sessionRaw.updatedAt ?? ''),
      status: (sessionRaw.status ?? 'CAPTURING') as FieldSession['status'],
      frames: frames.map((f) => f.record),
    },
    frames,
    skippedFrames: skippedRaw.map((s) => ({
      frameId: String(s.frameId ?? ''),
      reason: String(s.reason ?? 'unknown'),
    })),
    taskOutcomes: outcomesRaw.map((o) => ({
      taskId: String(o.taskId ?? ''),
      status: (o.status ?? 'OPEN') as CaptureTaskOutcome['status'],
      frameIds: Array.isArray(o.frameIds) ? (o.frameIds as string[]) : [],
      completedAt: (o.completedAt ?? null) as string | null,
    })),
    integrityFailures,
  };
}